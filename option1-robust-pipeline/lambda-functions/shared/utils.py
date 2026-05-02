import json
import logging
import boto3
from typing import Any, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger()

def get_parameter(ssm_client: boto3.client, parameter_name: str) -> str:
    """
    Retrieve parameter from AWS Systems Manager Parameter Store
    
    Args:
        ssm_client: Boto3 SSM client
        parameter_name: Name of the parameter to retrieve
        
    Returns:
        Parameter value as string
        
    Raises:
        Exception: If parameter retrieval fails
    """
    try:
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        logger.error(f"Failed to retrieve parameter {parameter_name}: {str(e)}")
        raise

def send_to_sqs(sqs_client: boto3.client, queue_url: str, message_body: str, 
                message_group_id: Optional[str] = None) -> bool:
    """
    Send message to SQS queue with error handling
    
    Args:
        sqs_client: Boto3 SQS client
        queue_url: URL of the target SQS queue
        message_body: Message content as string
        message_group_id: Optional message group ID for FIFO queues
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        send_params = {
            'QueueUrl': queue_url,
            'MessageBody': message_body,
            'MessageAttributes': {
                'timestamp': {
                    'StringValue': datetime.now(timezone.utc).isoformat(),
                    'DataType': 'String'
                },
                'source': {
                    'StringValue': 'bitcoin-prediction-pipeline',
                    'DataType': 'String'
                }
            }
        }
        
        if message_group_id:
            send_params['MessageGroupId'] = message_group_id
            send_params['MessageDeduplicationId'] = f"{message_group_id}-{int(datetime.now().timestamp())}"
        
        response = sqs_client.send_message(**send_params)
        
        if response.get('MessageId'):
            logger.debug(f"Successfully sent message to SQS: {response['MessageId']}")
            return True
        else:
            logger.error("Failed to send message to SQS: No MessageId in response")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send message to SQS: {str(e)}")
        return False

def send_batch_to_sqs(sqs_client: boto3.client, queue_url: str, 
                     messages: list, message_group_id_prefix: Optional[str] = None) -> Dict[str, int]:
    """
    Send multiple messages to SQS in batches
    
    Args:
        sqs_client: Boto3 SQS client
        queue_url: URL of the target SQS queue
        messages: List of message bodies (strings)
        message_group_id_prefix: Optional prefix for message group IDs
        
    Returns:
        Dict with success and failure counts
    """
    batch_size = 10  # SQS batch limit
    successful = 0
    failed = 0
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        entries = []
        
        for j, message in enumerate(batch):
            entry = {
                'Id': str(i + j),
                'MessageBody': message,
                'MessageAttributes': {
                    'timestamp': {
                        'StringValue': datetime.now(timezone.utc).isoformat(),
                        'DataType': 'String'
                    },
                    'batch_id': {
                        'StringValue': f"batch-{i // batch_size}",
                        'DataType': 'String'
                    }
                }
            }
            
            if message_group_id_prefix:
                entry['MessageGroupId'] = f"{message_group_id_prefix}-{i // batch_size}"
                entry['MessageDeduplicationId'] = f"{message_group_id_prefix}-{i + j}-{int(datetime.now().timestamp())}"
            
            entries.append(entry)
        
        try:
            response = sqs_client.send_message_batch(
                QueueUrl=queue_url,
                Entries=entries
            )
            
            successful += len(response.get('Successful', []))
            failed += len(response.get('Failed', []))
            
            if response.get('Failed'):
                for failure in response['Failed']:
                    logger.error(f"Failed to send message {failure['Id']}: {failure.get('Message', 'Unknown error')}")
                    
        except Exception as e:
            logger.error(f"Failed to send batch to SQS: {str(e)}")
            failed += len(batch)
    
    return {'successful': successful, 'failed': failed}

def create_success_response(message: str, data: Any = None) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        message: Success message
        data: Optional additional data
        
    Returns:
        Standardized response dictionary
    """
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'success',
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    }
    
    if data:
        body = json.loads(response['body'])
        body['data'] = data
        response['body'] = json.dumps(body)
    
    return response

def create_error_response(error_type: str, error_message: str, 
                         status_code: int = 500) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_type: Type of error
        error_message: Error message
        status_code: HTTP status code
        
    Returns:
        Standardized error response dictionary
    """
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'status': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    }

def validate_api_response(response_data: Dict, required_fields: list) -> bool:
    """
    Validate API response contains required fields
    
    Args:
        response_data: Response data dictionary
        required_fields: List of required field names
        
    Returns:
        bool: True if all required fields present
    """
    for field in required_fields:
        if field not in response_data:
            logger.error(f"Missing required field in API response: {field}")
            return False
    return True

def calculate_exponential_backoff(attempt: int, base_delay: float = 1.0, 
                                max_delay: float = 60.0, jitter: bool = True) -> float:
    """
    Calculate exponential backoff delay with optional jitter
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    import random
    
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add up to 25% random jitter
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount
    
    return delay

def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Sanitize string for safe storage and processing
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length-3] + "..."
    
    return sanitized.strip()

def format_timestamp(timestamp: Any) -> str:
    """
    Format various timestamp formats to ISO string
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        ISO formatted timestamp string
    """
    if isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    elif isinstance(timestamp, datetime):
        return timestamp.isoformat()
    else:
        return datetime.now(timezone.utc).isoformat()

def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split list into chunks of specified size
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely load JSON string with default fallback
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {str(e)}")
        return default

def get_current_timestamp() -> float:
    """Get current Unix timestamp"""
    return datetime.now(timezone.utc).timestamp()

def get_current_iso_timestamp() -> str:
    """Get current ISO formatted timestamp"""
    return datetime.now(timezone.utc).isoformat()