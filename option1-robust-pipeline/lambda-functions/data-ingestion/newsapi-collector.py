import json
import boto3
import requests
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import os
import sys

# Add shared utilities to path
sys.path.append('/opt/python/shared')
from circuit_breaker import CircuitBreaker
from retry_handler import RetryHandler
from utils import send_to_sqs, get_parameter, create_error_response, create_success_response

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
SQS_QUEUE_URL = os.environ['SQS_QUEUE_URL']
NEWSAPI_PARAM_NAME = os.environ['NEWSAPI_PARAM_NAME']

# Initialize AWS clients
ssm = boto3.client('ssm')
sqs = boto3.client('sqs')
cloudwatch = boto3.client('cloudwatch')

# Initialize circuit breaker and retry handler
circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=300,  # 5 minutes
    expected_exception=requests.RequestException
)
retry_handler = RetryHandler(max_retries=3, base_delay=1.0)

class NewsAPICollector:
    def __init__(self):
        self.api_key = None
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Bitcoin-Prediction-Pipeline/1.0'
        })

    def get_api_key(self) -> str:
        """Retrieve API key from Parameter Store"""
        if not self.api_key:
            try:
                self.api_key = get_parameter(ssm, NEWSAPI_PARAM_NAME)
                logger.info("Successfully retrieved NewsAPI key from Parameter Store")
            except Exception as e:
                logger.error(f"Failed to retrieve API key: {str(e)}")
                raise
        return self.api_key

    @circuit_breaker
    @retry_handler
    def fetch_bitcoin_news(self) -> List[Dict[str, Any]]:
        """Fetch Bitcoin-related news from NewsAPI"""
        api_key = self.get_api_key()
        
        # Search parameters for Bitcoin news
        params = {
            'q': 'bitcoin OR cryptocurrency OR crypto OR BTC',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': api_key
        }
        
        url = f"{self.base_url}/everything"
        
        logger.info("Fetching Bitcoin news from NewsAPI")
        start_time = time.time()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if request was successful
            if data.get('status') != 'ok':
                raise requests.RequestException(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            
            articles = data.get('articles', [])
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('NewsAPI.RequestDuration', duration, 'Seconds')
            self.send_custom_metric('NewsAPI.ArticlesCount', len(articles), 'Count')
            
            logger.info(f"Successfully fetched {len(articles)} articles in {duration:.2f} seconds")
            
            return self.process_articles(articles)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request failed: {str(e)}")
            self.send_custom_metric('NewsAPI.RequestFailures', 1, 'Count')
            raise
        except Exception as e:
            logger.error(f"Unexpected error in NewsAPI fetch: {str(e)}")
            self.send_custom_metric('NewsAPI.UnexpectedErrors', 1, 'Count')
            raise

    def process_articles(self, articles: List[Dict]) -> List[Dict[str, Any]]:
        """Process and clean article data"""
        processed_articles = []
        current_time = datetime.now(timezone.utc)
        
        for article in articles:
            try:
                # Skip articles without essential data
                if not article.get('title') or not article.get('publishedAt'):
                    continue
                
                # Parse publication date
                published_at = self.parse_date(article['publishedAt'])
                if not published_at:
                    continue
                
                # Skip articles older than 24 hours
                age_hours = (current_time - published_at).total_seconds() / 3600
                if age_hours > 24:
                    continue
                
                processed_article = {
                    'source': 'newsapi',
                    'data_type': 'news',
                    'timestamp': current_time.isoformat(),
                    'published_at': published_at.isoformat(),
                    'title': article.get('title', '').strip(),
                    'description': article.get('description', '').strip(),
                    'content': article.get('content', '').strip(),
                    'url': article.get('url', ''),
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'relevance_score': self.calculate_relevance(article),
                    'collection_timestamp': current_time.timestamp()
                }
                
                processed_articles.append(processed_article)
                
            except Exception as e:
                logger.warning(f"Failed to process article: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_articles)} articles out of {len(articles)} raw articles")
        return processed_articles

    def parse_date(self, date_string: str) -> datetime:
        """Parse ISO date string to datetime object"""
        try:
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Failed to parse date: {date_string}")
            return None

    def calculate_relevance(self, article: Dict) -> float:
        """Calculate relevance score based on Bitcoin keywords"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        # Bitcoin-specific keywords with weights
        keywords = {
            'bitcoin': 1.0,
            'btc': 0.9,
            'cryptocurrency': 0.7,
            'crypto': 0.6,
            'blockchain': 0.5,
            'satoshi': 0.8,
            'mining': 0.4,
            'wallet': 0.3,
            'exchange': 0.4,
            'halving': 0.9,
            'price': 0.5,
            'bull': 0.6,
            'bear': 0.6
        }
        
        score = 0.0
        for keyword, weight in keywords.items():
            if keyword in text:
                score += weight
        
        # Normalize score (0-1 range)
        return min(score / 5.0, 1.0)

    def send_custom_metric(self, metric_name: str, value: float, unit: str):
        """Send custom metrics to CloudWatch"""
        try:
            cloudwatch.put_metric_data(
                Namespace='Bitcoin/Prediction',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.now(timezone.utc)
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to send metric {metric_name}: {str(e)}")

def lambda_handler(event, context):
    """Main Lambda handler"""
    collector = NewsAPICollector()
    
    try:
        logger.info("Starting NewsAPI data collection")
        
        # Check circuit breaker status
        if circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, skipping NewsAPI call")
            return create_error_response(
                "Circuit breaker open",
                "NewsAPI circuit breaker is currently open due to previous failures"
            )
        
        # Fetch news data
        articles = collector.fetch_bitcoin_news()
        
        if not articles:
            logger.warning("No relevant articles found")
            return create_success_response(
                "No articles found",
                {"articles_processed": 0}
            )
        
        # Send data to SQS for processing
        messages_sent = 0
        batch_size = 10  # SQS batch limit
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            message_body = {
                'source': 'newsapi',
                'data_type': 'news_batch',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'articles': batch,
                'batch_info': {
                    'batch_number': i // batch_size + 1,
                    'batch_size': len(batch),
                    'total_articles': len(articles)
                }
            }
            
            success = send_to_sqs(
                sqs,
                SQS_QUEUE_URL,
                json.dumps(message_body),
                f"newsapi-batch-{i // batch_size + 1}"
            )
            
            if success:
                messages_sent += 1
            else:
                logger.error(f"Failed to send batch {i // batch_size + 1} to SQS")
        
        # Send final metrics
        collector.send_custom_metric('NewsAPI.MessagesSuccessfullySent', messages_sent, 'Count')
        collector.send_custom_metric('NewsAPI.TotalArticlesCollected', len(articles), 'Count')
        
        logger.info(f"Successfully processed {len(articles)} articles and sent {messages_sent} batches to SQS")
        
        return create_success_response(
            "NewsAPI collection completed",
            {
                "articles_collected": len(articles),
                "messages_sent": messages_sent,
                "circuit_breaker_status": "closed" if not circuit_breaker.is_open() else "open"
            }
        )
        
    except Exception as e:
        logger.error(f"NewsAPI collection failed: {str(e)}")
        collector.send_custom_metric('NewsAPI.CollectionFailures', 1, 'Count')
        
        return create_error_response(
            "NewsAPI collection failed",
            str(e)
        )