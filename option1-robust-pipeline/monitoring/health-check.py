#!/usr/bin/env python3
"""
Bitcoin Prediction Pipeline Health Check Script

This script performs comprehensive health checks on all components
of the Bitcoin prediction pipeline and provides detailed diagnostics.
"""

import boto3
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BitcoinPipelineHealthChecker:
    def __init__(self, region: str = 'eu-west-2'):
        """
        Initialize health checker
        
        Args:
            region: AWS region
        """
        self.region = region
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sqs = boto3.client('sqs', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.events = boto3.client('events', region_name=region)
        self.ssm = boto3.client('ssm', region_name=region)
        
        # Component configurations
        self.lambda_functions = [
            'bitcoin-newsapi-collector',
            'bitcoin-cryptocompare-collector', 
            'bitcoin-coinapi-collector',
            'bitcoin-sentiment-analyzer',
            'bitcoin-feature-engineer',
            'bitcoin-predictor'
        ]
        
        self.sqs_queues = [
            'bitcoin-data-processing',
            'bitcoin-data-processing-dlq',
            'bitcoin-prediction',
            'bitcoin-prediction-dlq'
        ]
        
        self.dynamodb_tables = [
            'bitcoin-predictions',
            'bitcoin-raw-data'
        ]
        
        self.s3_bucket = 'bitcoin-prediction-data-654654488711'
        
        self.event_rules = [
            'bitcoin-data-collection-schedule',
            'bitcoin-prediction-schedule'
        ]
        
        self.ssm_parameters = [
            '/bitcoin/newsapi/key',
            '/bitcoin/cryptocompare/key'
        ]

    def check_lambda_functions(self) -> Dict[str, Any]:
        """Check health of all Lambda functions"""
        logger.info("Checking Lambda functions...")
        
        results = {
            'status': 'healthy',
            'functions': {},
            'issues': []
        }
        
        for function_name in self.lambda_functions:
            try:
                # Get function configuration
                response = self.lambda_client.get_function(FunctionName=function_name)
                
                function_info = {
                    'exists': True,
                    'state': response['Configuration']['State'],
                    'last_modified': response['Configuration']['LastModified'],
                    'runtime': response['Configuration']['Runtime'],
                    'timeout': response['Configuration']['Timeout'],
                    'memory_size': response['Configuration']['MemorySize']
                }
                
                # Check recent invocations and errors
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=1)
                
                try:
                    # Get invocation metrics
                    invocation_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/Lambda',
                        MetricName='Invocations',
                        Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    # Get error metrics
                    error_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/Lambda',
                        MetricName='Errors',
                        Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    # Get duration metrics
                    duration_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/Lambda',
                        MetricName='Duration',
                        Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Average', 'Maximum']
                    )
                    
                    invocations = sum(point['Sum'] for point in invocation_metrics['Datapoints'])
                    errors = sum(point['Sum'] for point in error_metrics['Datapoints'])
                    
                    function_info['invocations_last_hour'] = invocations
                    function_info['errors_last_hour'] = errors
                    function_info['error_rate'] = (errors / invocations * 100) if invocations > 0 else 0
                    
                    if duration_metrics['Datapoints']:
                        avg_duration = sum(point['Average'] for point in duration_metrics['Datapoints']) / len(duration_metrics['Datapoints'])
                        max_duration = max(point['Maximum'] for point in duration_metrics['Datapoints'])
                        function_info['avg_duration_ms'] = avg_duration
                        function_info['max_duration_ms'] = max_duration
                    
                    # Health assessment
                    if function_info['state'] != 'Active':
                        function_info['health'] = 'unhealthy'
                        results['issues'].append(f"{function_name}: Function state is {function_info['state']}")
                    elif function_info['error_rate'] > 50:
                        function_info['health'] = 'degraded'
                        results['issues'].append(f"{function_name}: High error rate ({function_info['error_rate']:.1f}%)")
                    elif function_info.get('max_duration_ms', 0) > function_info['timeout'] * 800:  # 80% of timeout
                        function_info['health'] = 'degraded'
                        results['issues'].append(f"{function_name}: High duration approaching timeout")
                    else:
                        function_info['health'] = 'healthy'
                
                except Exception as e:
                    logger.warning(f"Failed to get metrics for {function_name}: {str(e)}")
                    function_info['health'] = 'unknown'
                    function_info['metrics_error'] = str(e)
                
                results['functions'][function_name] = function_info
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    results['functions'][function_name] = {
                        'exists': False,
                        'health': 'missing'
                    }
                    results['issues'].append(f"{function_name}: Function does not exist")
                else:
                    results['functions'][function_name] = {
                        'exists': 'unknown',
                        'health': 'error',
                        'error': str(e)
                    }
                    results['issues'].append(f"{function_name}: Error checking function - {str(e)}")
        
        # Overall status
        unhealthy_functions = [f for f, info in results['functions'].items() 
                             if info.get('health') in ['unhealthy', 'missing', 'error']]
        degraded_functions = [f for f, info in results['functions'].items() 
                            if info.get('health') == 'degraded']
        
        if unhealthy_functions:
            results['status'] = 'unhealthy'
        elif degraded_functions:
            results['status'] = 'degraded'
        
        logger.info(f"Lambda functions check completed: {results['status']}")
        return results

    def check_sqs_queues(self) -> Dict[str, Any]:
        """Check health of SQS queues"""
        logger.info("Checking SQS queues...")
        
        results = {
            'status': 'healthy',
            'queues': {},
            'issues': []
        }
        
        for queue_name in self.sqs_queues:
            try:
                # Get queue URL
                queue_url_response = self.sqs.get_queue_url(QueueName=queue_name)
                queue_url = queue_url_response['QueueUrl']
                
                # Get queue attributes
                attributes = self.sqs.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=['All']
                )['Attributes']
                
                queue_info = {
                    'exists': True,
                    'messages_available': int(attributes.get('ApproximateNumberOfMessages', 0)),
                    'messages_in_flight': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                    'messages_delayed': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                    'created_timestamp': attributes.get('CreatedTimestamp'),
                    'last_modified_timestamp': attributes.get('LastModifiedTimestamp'),
                    'visibility_timeout': int(attributes.get('VisibilityTimeout', 0)),
                    'message_retention_period': int(attributes.get('MessageRetentionPeriod', 0))
                }
                
                # Health assessment
                if 'dlq' in queue_name.lower():
                    # Dead letter queues should ideally be empty
                    if queue_info['messages_available'] > 0:
                        queue_info['health'] = 'degraded'
                        results['issues'].append(f"{queue_name}: {queue_info['messages_available']} messages in DLQ")
                    else:
                        queue_info['health'] = 'healthy'
                else:
                    # Regular queues - check for excessive backlog
                    total_messages = queue_info['messages_available'] + queue_info['messages_in_flight']
                    if total_messages > 1000:
                        queue_info['health'] = 'degraded'
                        results['issues'].append(f"{queue_name}: High message backlog ({total_messages} messages)")
                    else:
                        queue_info['health'] = 'healthy'
                
                results['queues'][queue_name] = queue_info
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
                    results['queues'][queue_name] = {
                        'exists': False,
                        'health': 'missing'
                    }
                    results['issues'].append(f"{queue_name}: Queue does not exist")
                else:
                    results['queues'][queue_name] = {
                        'exists': 'unknown',
                        'health': 'error',
                        'error': str(e)
                    }
                    results['issues'].append(f"{queue_name}: Error checking queue - {str(e)}")
        
        # Overall status
        unhealthy_queues = [q for q, info in results['queues'].items() 
                          if info.get('health') in ['missing', 'error']]
        degraded_queues = [q for q, info in results['queues'].items() 
                         if info.get('health') == 'degraded']
        
        if unhealthy_queues:
            results['status'] = 'unhealthy'
        elif degraded_queues:
            results['status'] = 'degraded'
        
        logger.info(f"SQS queues check completed: {results['status']}")
        return results

    def check_dynamodb_tables(self) -> Dict[str, Any]:
        """Check health of DynamoDB tables"""
        logger.info("Checking DynamoDB tables...")
        
        results = {
            'status': 'healthy',
            'tables': {},
            'issues': []
        }
        
        for table_name in self.dynamodb_tables:
            try:
                table = self.dynamodb.Table(table_name)
                
                # Get table description
                description = table.meta.client.describe_table(TableName=table_name)['Table']
                
                table_info = {
                    'exists': True,
                    'status': description['TableStatus'],
                    'item_count': description.get('ItemCount', 0),
                    'table_size_bytes': description.get('TableSizeBytes', 0),
                    'billing_mode': description.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED'),
                    'creation_date': description['CreationDateTime'].isoformat() if 'CreationDateTime' in description else None
                }
                
                # Check recent activity
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=1)
                
                try:
                    # Get read/write metrics
                    read_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/DynamoDB',
                        MetricName='ConsumedReadCapacityUnits',
                        Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    write_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/DynamoDB',
                        MetricName='ConsumedWriteCapacityUnits',
                        Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    # Get throttling metrics
                    read_throttle_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/DynamoDB',
                        MetricName='ReadThrottleCount',
                        Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    write_throttle_metrics = self.cloudwatch.get_metric_statistics(
                        Namespace='AWS/DynamoDB',
                        MetricName='WriteThrottleCount',
                        Dimensions=[{'Name': 'TableName', 'Value': table_name}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Sum']
                    )
                    
                    table_info['read_capacity_consumed_last_hour'] = sum(point['Sum'] for point in read_metrics['Datapoints'])
                    table_info['write_capacity_consumed_last_hour'] = sum(point['Sum'] for point in write_metrics['Datapoints'])
                    table_info['read_throttles_last_hour'] = sum(point['Sum'] for point in read_throttle_metrics['Datapoints'])
                    table_info['write_throttles_last_hour'] = sum(point['Sum'] for point in write_throttle_metrics['Datapoints'])
                    
                except Exception as e:
                    logger.warning(f"Failed to get metrics for table {table_name}: {str(e)}")
                    table_info['metrics_error'] = str(e)
                
                # Health assessment
                if table_info['status'] != 'ACTIVE':
                    table_info['health'] = 'unhealthy'
                    results['issues'].append(f"{table_name}: Table status is {table_info['status']}")
                elif table_info.get('read_throttles_last_hour', 0) > 0 or table_info.get('write_throttles_last_hour', 0) > 0:
                    table_info['health'] = 'degraded'
                    results['issues'].append(f"{table_name}: Throttling detected")
                else:
                    table_info['health'] = 'healthy'
                
                results['tables'][table_name] = table_info
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    results['tables'][table_name] = {
                        'exists': False,
                        'health': 'missing'
                    }
                    results['issues'].append(f"{table_name}: Table does not exist")
                else:
                    results['tables'][table_name] = {
                        'exists': 'unknown',
                        'health': 'error',
                        'error': str(e)
                    }
                    results['issues'].append(f"{table_name}: Error checking table - {str(e)}")
        
        # Overall status
        unhealthy_tables = [t for t, info in results['tables'].items() 
                          if info.get('health') in ['unhealthy', 'missing', 'error']]
        degraded_tables = [t for t, info in results['tables'].items() 
                         if info.get('health') == 'degraded']
        
        if unhealthy_tables:
            results['status'] = 'unhealthy'
        elif degraded_tables:
            results['status'] = 'degraded'
        
        logger.info(f"DynamoDB tables check completed: {results['status']}")
        return results

    def check_s3_bucket(self) -> Dict[str, Any]:
        """Check health of S3 bucket"""
        logger.info("Checking S3 bucket...")
        
        results = {
            'status': 'healthy',
            'bucket': {},
            'issues': []
        }
        
        try:
            # Check if bucket exists and is accessible
            self.s3.head_bucket(Bucket=self.s3_bucket)
            
            # Get bucket location
            location = self.s3.get_bucket_location(Bucket=self.s3_bucket)
            
            # List recent objects
            objects = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                MaxKeys=100
            )
            
            bucket_info = {
                'exists': True,
                'location': location.get('LocationConstraint', 'us-east-1'),
                'object_count': objects.get('KeyCount', 0),
                'is_truncated': objects.get('IsTruncated', False)
            }
            
            # Check for recent model uploads
            model_objects = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='models/',
                MaxKeys=10
            )
            
            if model_objects.get('KeyCount', 0) > 0:
                latest_model = max(model_objects['Contents'], key=lambda x: x['LastModified'])
                bucket_info['latest_model'] = {
                    'key': latest_model['Key'],
                    'last_modified': latest_model['LastModified'].isoformat(),
                    'size': latest_model['Size']
                }
                
                # Check if model is recent (within last 7 days)
                model_age = datetime.now(timezone.utc) - latest_model['LastModified']
                if model_age.days > 7:
                    bucket_info['health'] = 'degraded'
                    results['issues'].append(f"S3: Model is {model_age.days} days old")
                else:
                    bucket_info['health'] = 'healthy'
            else:
                bucket_info['health'] = 'degraded'
                results['issues'].append("S3: No model files found")
            
            results['bucket'] = bucket_info
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                results['bucket'] = {
                    'exists': False,
                    'health': 'missing'
                }
                results['issues'].append("S3: Bucket does not exist")
                results['status'] = 'unhealthy'
            else:
                results['bucket'] = {
                    'exists': 'unknown',
                    'health': 'error',
                    'error': str(e)
                }
                results['issues'].append(f"S3: Error checking bucket - {str(e)}")
                results['status'] = 'unhealthy'
        
        if results['bucket'].get('health') == 'degraded':
            results['status'] = 'degraded'
        
        logger.info(f"S3 bucket check completed: {results['status']}")
        return results

    def check_event_rules(self) -> Dict[str, Any]:
        """Check EventBridge rules"""
        logger.info("Checking EventBridge rules...")
        
        results = {
            'status': 'healthy',
            'rules': {},
            'issues': []
        }
        
        for rule_name in self.event_rules:
            try:
                # Get rule details
                rule_response = self.events.describe_rule(Name=rule_name)
                
                # Get rule targets
                targets_response = self.events.list_targets_by_rule(Rule=rule_name)
                
                rule_info = {
                    'exists': True,
                    'state': rule_response['State'],
                    'schedule_expression': rule_response.get('ScheduleExpression'),
                    'description': rule_response.get('Description'),
                    'target_count': len(targets_response['Targets']),
                    'targets': [target['Arn'] for target in targets_response['Targets']]
                }
                
                # Health assessment
                if rule_info['state'] != 'ENABLED':
                    rule_info['health'] = 'degraded'
                    results['issues'].append(f"{rule_name}: Rule is {rule_info['state']}")
                elif rule_info['target_count'] == 0:
                    rule_info['health'] = 'degraded'
                    results['issues'].append(f"{rule_name}: No targets configured")
                else:
                    rule_info['health'] = 'healthy'
                
                results['rules'][rule_name] = rule_info
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    results['rules'][rule_name] = {
                        'exists': False,
                        'health': 'missing'
                    }
                    results['issues'].append(f"{rule_name}: Rule does not exist")
                else:
                    results['rules'][rule_name] = {
                        'exists': 'unknown',
                        'health': 'error',
                        'error': str(e)
                    }
                    results['issues'].append(f"{rule_name}: Error checking rule - {str(e)}")
        
        # Overall status
        unhealthy_rules = [r for r, info in results['rules'].items() 
                         if info.get('health') in ['missing', 'error']]
        degraded_rules = [r for r, info in results['rules'].items() 
                        if info.get('health') == 'degraded']
        
        if unhealthy_rules:
            results['status'] = 'unhealthy'
        elif degraded_rules:
            results['status'] = 'degraded'
        
        logger.info(f"EventBridge rules check completed: {results['status']}")
        return results

    def check_ssm_parameters(self) -> Dict[str, Any]:
        """Check SSM parameters"""
        logger.info("Checking SSM parameters...")
        
        results = {
            'status': 'healthy',
            'parameters': {},
            'issues': []
        }
        
        for param_name in self.ssm_parameters:
            try:
                # Get parameter (without decryption for security)
                param_response = self.ssm.get_parameter(Name=param_name)
                
                param_info = {
                    'exists': True,
                    'type': param_response['Parameter']['Type'],
                    'last_modified_date': param_response['Parameter']['LastModifiedDate'].isoformat(),
                    'version': param_response['Parameter']['Version'],
                    'has_value': len(param_response['Parameter']['Value']) > 0
                }
                
                # Health assessment
                if not param_info['has_value']:
                    param_info['health'] = 'degraded'
                    results['issues'].append(f"{param_name}: Parameter has empty value")
                else:
                    param_info['health'] = 'healthy'
                
                results['parameters'][param_name] = param_info
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ParameterNotFound':
                    results['parameters'][param_name] = {
                        'exists': False,
                        'health': 'missing'
                    }
                    results['issues'].append(f"{param_name}: Parameter does not exist")
                else:
                    results['parameters'][param_name] = {
                        'exists': 'unknown',
                        'health': 'error',
                        'error': str(e)
                    }
                    results['issues'].append(f"{param_name}: Error checking parameter - {str(e)}")
        
        # Overall status
        unhealthy_params = [p for p, info in results['parameters'].items() 
                          if info.get('health') in ['missing', 'error']]
        degraded_params = [p for p, info in results['parameters'].items() 
                         if info.get('health') == 'degraded']
        
        if unhealthy_params:
            results['status'] = 'unhealthy'
        elif degraded_params:
            results['status'] = 'degraded'
        
        logger.info(f"SSM parameters check completed: {results['status']}")
        return results

    def check_data_freshness(self) -> Dict[str, Any]:
        """Check data freshness in the pipeline"""
        logger.info("Checking data freshness...")
        
        results = {
            'status': 'healthy',
            'data_freshness': {},
            'issues': []
        }
        
        try:
            # Check latest predictions
            predictions_table = self.dynamodb.Table('bitcoin-predictions')
            
            # Query for latest predictions
            response = predictions_table.query(
                IndexName='prediction-type-timestamp-index',
                KeyConditionExpression='prediction_type = :pt',
                ExpressionAttributeValues={':pt': 'bitcoin_price'},
                ScanIndexForward=False,
                Limit=1
            )
            
            if response['Items']:
                latest_prediction = response['Items'][0]
                prediction_time = datetime.fromtimestamp(latest_prediction['timestamp'], timezone.utc)
                prediction_age = datetime.now(timezone.utc) - prediction_time
                
                results['data_freshness']['latest_prediction'] = {
                    'timestamp': prediction_time.isoformat(),
                    'age_minutes': prediction_age.total_seconds() / 60,
                    'current_price': latest_prediction.get('current_price'),
                    'confidence_score': latest_prediction.get('confidence_score')
                }
                
                # Check if prediction is too old
                if prediction_age.total_seconds() > 3600 * 2:  # 2 hours
                    results['issues'].append(f"Latest prediction is {prediction_age.total_seconds() / 3600:.1f} hours old")
                    results['status'] = 'degraded'
            else:
                results['issues'].append("No predictions found in database")
                results['status'] = 'unhealthy'
            
            # Check feature freshness
            feature_response = predictions_table.query(
                IndexName='prediction-type-timestamp-index',
                KeyConditionExpression='prediction_type = :pt',
                ExpressionAttributeValues={':pt': 'feature_vector'},
                ScanIndexForward=False,
                Limit=1
            )
            
            if feature_response['Items']:
                latest_features = feature_response['Items'][0]
                feature_time = datetime.fromtimestamp(latest_features['timestamp'], timezone.utc)
                feature_age = datetime.now(timezone.utc) - feature_time
                
                results['data_freshness']['latest_features'] = {
                    'timestamp': feature_time.isoformat(),
                    'age_minutes': feature_age.total_seconds() / 60,
                    'feature_count': latest_features.get('feature_count'),
                    'data_quality_score': latest_features.get('data_quality_score')
                }
                
                # Check if features are too old
                if feature_age.total_seconds() > 3600:  # 1 hour
                    results['issues'].append(f"Latest features are {feature_age.total_seconds() / 3600:.1f} hours old")
                    if results['status'] == 'healthy':
                        results['status'] = 'degraded'
            else:
                results['issues'].append("No feature vectors found in database")
                results['status'] = 'unhealthy'
            
        except Exception as e:
            logger.error(f"Failed to check data freshness: {str(e)}")
            results['data_freshness']['error'] = str(e)
            results['issues'].append(f"Error checking data freshness: {str(e)}")
            results['status'] = 'error'
        
        logger.info(f"Data freshness check completed: {results['status']}")
        return results

    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all components"""
        logger.info("Starting comprehensive health check...")
        
        start_time = time.time()
        
        # Run all health checks
        lambda_results = self.check_lambda_functions()
        sqs_results = self.check_sqs_queues()
        dynamodb_results = self.check_dynamodb_tables()
        s3_results = self.check_s3_bucket()
        events_results = self.check_event_rules()
        ssm_results = self.check_ssm_parameters()
        data_freshness_results = self.check_data_freshness()
        
        # Aggregate results
        all_results = {
            'lambda_functions': lambda_results,
            'sqs_queues': sqs_results,
            'dynamodb_tables': dynamodb_results,
            's3_bucket': s3_results,
            'event_rules': events_results,
            'ssm_parameters': ssm_results,
            'data_freshness': data_freshness_results
        }
        
        # Calculate overall health
        component_statuses = [result['status'] for result in all_results.values()]
        
        if 'unhealthy' in component_statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in component_statuses:
            overall_status = 'degraded'
        elif 'error' in component_statuses:
            overall_status = 'error'
        else:
            overall_status = 'healthy'
        
        # Collect all issues
        all_issues = []
        for component, result in all_results.items():
            if 'issues' in result:
                all_issues.extend([f"{component}: {issue}" for issue in result['issues']])
        
        # Create summary
        summary = {
            'overall_status': overall_status,
            'check_timestamp': datetime.now(timezone.utc).isoformat(),
            'check_duration_seconds': time.time() - start_time,
            'component_summary': {
                component: result['status'] for component, result in all_results.items()
            },
            'total_issues': len(all_issues),
            'issues': all_issues,
            'recommendations': self._generate_recommendations(all_results)
        }
        
        # Final results
        final_results = {
            'summary': summary,
            'detailed_results': all_results
        }
        
        logger.info(f"Health check completed: {overall_status} ({len(all_issues)} issues)")
        return final_results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        # Lambda recommendations
        if results['lambda_functions']['status'] != 'healthy':
            recommendations.append("Review Lambda function logs and consider increasing memory or timeout settings")
        
        # SQS recommendations
        if results['sqs_queues']['status'] != 'healthy':
            recommendations.append("Check SQS dead letter queues and redrive failed messages")
        
        # DynamoDB recommendations
        if results['dynamodb_tables']['status'] != 'healthy':
            recommendations.append("Monitor DynamoDB throttling and consider increasing capacity")
        
        # Data freshness recommendations
        if results['data_freshness']['status'] != 'healthy':
            recommendations.append("Check EventBridge schedules and ensure data collection is running")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Bitcoin Prediction Pipeline Health Check')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    parser.add_argument('--output', choices=['json', 'summary'], default='summary', help='Output format')
    parser.add_argument('--full-check', action='store_true', help='Run comprehensive health check')
    
    args = parser.parse_args()
    
    # Initialize health checker
    health_checker = BitcoinPipelineHealthChecker(region=args.region)
    
    try:
        if args.full_check:
            results = health_checker.run_comprehensive_health_check()
        else:
            # Quick check - just run data freshness and Lambda functions
            lambda_results = health_checker.check_lambda_functions()
            data_freshness_results = health_checker.check_data_freshness()
            
            results = {
                'summary': {
                    'overall_status': 'degraded' if lambda_results['status'] != 'healthy' or data_freshness_results['status'] != 'healthy' else 'healthy',
                    'check_timestamp': datetime.now(timezone.utc).isoformat(),
                    'lambda_status': lambda_results['status'],
                    'data_freshness_status': data_freshness_results['status']
                },
                'lambda_functions': lambda_results,
                'data_freshness': data_freshness_results
            }
        
        # Output results
        if args.output == 'json':
            print(json.dumps(results, indent=2, default=str))
        else:
            # Summary output
            summary = results['summary']
            print(f"\n=== Bitcoin Prediction Pipeline Health Check ===")
            print(f"Overall Status: {summary['overall_status'].upper()}")
            print(f"Check Time: {summary['check_timestamp']}")
            
            if 'component_summary' in summary:
                print(f"\nComponent Status:")
                for component, status in summary['component_summary'].items():
                    print(f"  {component}: {status}")
            
            if summary.get('total_issues', 0) > 0:
                print(f"\nIssues ({summary['total_issues']}):")
                for issue in summary.get('issues', []):
                    print(f"  - {issue}")
            
            if 'recommendations' in summary:
                print(f"\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
            
            print(f"\nHealth check completed in {summary.get('check_duration_seconds', 0):.1f} seconds")
        
        # Exit with appropriate code
        if results['summary']['overall_status'] in ['unhealthy', 'error']:
            exit(1)
        elif results['summary']['overall_status'] == 'degraded':
            exit(2)
        else:
            exit(0)
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        print(f"ERROR: Health check failed - {str(e)}")
        exit(3)

if __name__ == "__main__":
    main()