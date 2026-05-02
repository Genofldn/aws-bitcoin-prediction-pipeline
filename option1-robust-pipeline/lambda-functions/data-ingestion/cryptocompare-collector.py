import json
import boto3
import requests
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
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
CRYPTOCOMPARE_PARAM_NAME = os.environ['CRYPTOCOMPARE_PARAM_NAME']

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

class CryptoCompareCollector:
    def __init__(self):
        self.api_key = None
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Bitcoin-Prediction-Pipeline/1.0'
        })

    def get_api_key(self) -> str:
        """Retrieve API key from Parameter Store"""
        if not self.api_key:
            try:
                self.api_key = get_parameter(ssm, CRYPTOCOMPARE_PARAM_NAME)
                logger.info("Successfully retrieved CryptoCompare API key from Parameter Store")
            except Exception as e:
                logger.error(f"Failed to retrieve API key: {str(e)}")
                raise
        return self.api_key

    @circuit_breaker
    @retry_handler
    def fetch_bitcoin_price_data(self) -> Dict[str, Any]:
        """Fetch Bitcoin price data from CryptoCompare"""
        api_key = self.get_api_key()
        
        # Get current price data
        price_url = f"{self.base_url}/price"
        price_params = {
            'fsym': 'BTC',
            'tsyms': 'USD,EUR,GBP',
            'api_key': api_key
        }
        
        logger.info("Fetching Bitcoin price data from CryptoCompare")
        start_time = time.time()
        
        try:
            response = self.session.get(price_url, params=price_params, timeout=30)
            response.raise_for_status()
            
            price_data = response.json()
            
            if 'Response' in price_data and price_data['Response'] == 'Error':
                raise requests.RequestException(f"CryptoCompare API error: {price_data.get('Message', 'Unknown error')}")
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('CryptoCompare.PriceRequestDuration', duration, 'Seconds')
            
            logger.info(f"Successfully fetched price data in {duration:.2f} seconds")
            
            return {
                'source': 'cryptocompare',
                'data_type': 'price',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'prices': price_data,
                'collection_timestamp': time.time()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CryptoCompare price request failed: {str(e)}")
            self.send_custom_metric('CryptoCompare.PriceRequestFailures', 1, 'Count')
            raise

    @circuit_breaker
    @retry_handler
    def fetch_bitcoin_historical_data(self) -> Dict[str, Any]:
        """Fetch Bitcoin historical data (last 24 hours)"""
        api_key = self.get_api_key()
        
        # Get hourly data for last 24 hours
        hist_url = f"{self.base_url}/v2/histohour"
        hist_params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': 24,  # Last 24 hours
            'api_key': api_key
        }
        
        logger.info("Fetching Bitcoin historical data from CryptoCompare")
        start_time = time.time()
        
        try:
            response = self.session.get(hist_url, params=hist_params, timeout=30)
            response.raise_for_status()
            
            hist_data = response.json()
            
            if hist_data.get('Response') == 'Error':
                raise requests.RequestException(f"CryptoCompare API error: {hist_data.get('Message', 'Unknown error')}")
            
            # Process historical data
            processed_data = self.process_historical_data(hist_data.get('Data', {}).get('Data', []))
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('CryptoCompare.HistoricalRequestDuration', duration, 'Seconds')
            self.send_custom_metric('CryptoCompare.HistoricalDataPoints', len(processed_data), 'Count')
            
            logger.info(f"Successfully fetched {len(processed_data)} historical data points in {duration:.2f} seconds")
            
            return {
                'source': 'cryptocompare',
                'data_type': 'historical',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'historical_data': processed_data,
                'collection_timestamp': time.time()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CryptoCompare historical request failed: {str(e)}")
            self.send_custom_metric('CryptoCompare.HistoricalRequestFailures', 1, 'Count')
            raise

    @circuit_breaker
    @retry_handler
    def fetch_social_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch Bitcoin social statistics"""
        api_key = self.get_api_key()
        
        # Get social stats
        social_url = f"{self.base_url}/social/coin/histo/day"
        social_params = {
            'coinId': '1182',  # Bitcoin ID
            'limit': 1,  # Just latest
            'api_key': api_key
        }
        
        logger.info("Fetching Bitcoin social stats from CryptoCompare")
        start_time = time.time()
        
        try:
            response = self.session.get(social_url, params=social_params, timeout=30)
            response.raise_for_status()
            
            social_data = response.json()
            
            if social_data.get('Response') == 'Error':
                logger.warning(f"CryptoCompare social API error: {social_data.get('Message', 'Unknown error')}")
                return None
            
            # Process social data
            data_points = social_data.get('Data', [])
            if not data_points:
                return None
            
            latest_social = data_points[-1]
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('CryptoCompare.SocialRequestDuration', duration, 'Seconds')
            
            logger.info(f"Successfully fetched social stats in {duration:.2f} seconds")
            
            return {
                'source': 'cryptocompare',
                'data_type': 'social',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'social_stats': {
                    'comments': latest_social.get('comments', 0),
                    'posts': latest_social.get('posts', 0),
                    'followers': latest_social.get('followers', 0),
                    'points': latest_social.get('points', 0),
                    'overview_page_views': latest_social.get('overview_page_views', 0),
                    'analysis_page_views': latest_social.get('analysis_page_views', 0),
                    'markets_page_views': latest_social.get('markets_page_views', 0),
                    'charts_page_views': latest_social.get('charts_page_views', 0),
                    'trades_page_views': latest_social.get('trades_page_views', 0),
                    'forum_page_views': latest_social.get('forum_page_views', 0)
                },
                'collection_timestamp': time.time()
            }
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"CryptoCompare social request failed (non-critical): {str(e)}")
            self.send_custom_metric('CryptoCompare.SocialRequestFailures', 1, 'Count')
            return None

    def process_historical_data(self, data_points: List[Dict]) -> List[Dict[str, Any]]:
        """Process historical data points"""
        processed_data = []
        
        for point in data_points:
            try:
                processed_point = {
                    'timestamp': point.get('time', 0),
                    'open': point.get('open', 0),
                    'high': point.get('high', 0),
                    'low': point.get('low', 0),
                    'close': point.get('close', 0),
                    'volume_from': point.get('volumefrom', 0),
                    'volume_to': point.get('volumeto', 0),
                    'conversion_type': point.get('conversionType', ''),
                    'conversion_symbol': point.get('conversionSymbol', '')
                }
                
                # Calculate additional metrics
                if processed_point['high'] > 0 and processed_point['low'] > 0:
                    processed_point['price_range'] = processed_point['high'] - processed_point['low']
                    processed_point['price_volatility'] = processed_point['price_range'] / processed_point['close'] if processed_point['close'] > 0 else 0
                
                processed_data.append(processed_point)
                
            except Exception as e:
                logger.warning(f"Failed to process historical data point: {str(e)}")
                continue
        
        return processed_data

    def calculate_technical_indicators(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if len(historical_data) < 2:
            return {}
        
        # Sort by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x['timestamp'])
        closes = [point['close'] for point in sorted_data if point['close'] > 0]
        
        if len(closes) < 2:
            return {}
        
        indicators = {}
        
        # Simple Moving Average (SMA)
        if len(closes) >= 24:
            sma_24 = sum(closes[-24:]) / 24
            indicators['sma_24h'] = sma_24
        
        if len(closes) >= 12:
            sma_12 = sum(closes[-12:]) / 12
            indicators['sma_12h'] = sma_12
        
        # Price change
        if len(closes) >= 2:
            price_change = closes[-1] - closes[-2]
            price_change_pct = (price_change / closes[-2]) * 100 if closes[-2] > 0 else 0
            indicators['price_change'] = price_change
            indicators['price_change_pct'] = price_change_pct
        
        # 24h high/low
        if len(sorted_data) >= 24:
            recent_data = sorted_data[-24:]
            highs = [point['high'] for point in recent_data if point['high'] > 0]
            lows = [point['low'] for point in recent_data if point['low'] > 0]
            
            if highs and lows:
                indicators['24h_high'] = max(highs)
                indicators['24h_low'] = min(lows)
                indicators['24h_range'] = indicators['24h_high'] - indicators['24h_low']
        
        # Volume analysis
        volumes = [point['volume_to'] for point in sorted_data[-24:] if point['volume_to'] > 0]
        if volumes:
            indicators['avg_volume_24h'] = sum(volumes) / len(volumes)
            indicators['current_volume'] = volumes[-1] if volumes else 0
        
        return indicators

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
    collector = CryptoCompareCollector()
    
    try:
        logger.info("Starting CryptoCompare data collection")
        
        # Check circuit breaker status
        if circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, skipping CryptoCompare call")
            return create_error_response(
                "Circuit breaker open",
                "CryptoCompare circuit breaker is currently open due to previous failures"
            )
        
        # Collect all data types
        collected_data = []
        
        # 1. Price data (critical)
        try:
            price_data = collector.fetch_bitcoin_price_data()
            collected_data.append(price_data)
        except Exception as e:
            logger.error(f"Failed to fetch price data: {str(e)}")
            # Price data is critical, so fail if we can't get it
            raise
        
        # 2. Historical data (critical)
        try:
            historical_data = collector.fetch_bitcoin_historical_data()
            
            # Calculate technical indicators
            technical_indicators = collector.calculate_technical_indicators(
                historical_data.get('historical_data', [])
            )
            
            if technical_indicators:
                historical_data['technical_indicators'] = technical_indicators
            
            collected_data.append(historical_data)
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            # Historical data is critical for predictions
            raise
        
        # 3. Social stats (non-critical)
        try:
            social_data = collector.fetch_social_stats()
            if social_data:
                collected_data.append(social_data)
        except Exception as e:
            logger.warning(f"Failed to fetch social data (non-critical): {str(e)}")
            # Social data is nice-to-have, don't fail the whole process
        
        if not collected_data:
            logger.error("No data collected from CryptoCompare")
            return create_error_response(
                "No data collected",
                "Failed to collect any data from CryptoCompare API"
            )
        
        # Send data to SQS for processing
        messages_sent = 0
        
        for data_item in collected_data:
            message_body = json.dumps(data_item)
            success = send_to_sqs(
                sqs,
                SQS_QUEUE_URL,
                message_body,
                f"cryptocompare-{data_item['data_type']}"
            )
            
            if success:
                messages_sent += 1
            else:
                logger.error(f"Failed to send {data_item['data_type']} data to SQS")
        
        # Send final metrics
        collector.send_custom_metric('CryptoCompare.MessagesSuccessfullySent', messages_sent, 'Count')
        collector.send_custom_metric('CryptoCompare.TotalDataTypesCollected', len(collected_data), 'Count')
        
        logger.info(f"Successfully collected {len(collected_data)} data types and sent {messages_sent} messages to SQS")
        
        return create_success_response(
            "CryptoCompare collection completed",
            {
                "data_types_collected": [item['data_type'] for item in collected_data],
                "messages_sent": messages_sent,
                "circuit_breaker_status": "closed" if not circuit_breaker.is_open() else "open"
            }
        )
        
    except Exception as e:
        logger.error(f"CryptoCompare collection failed: {str(e)}")
        collector.send_custom_metric('CryptoCompare.CollectionFailures', 1, 'Count')
        
        return create_error_response(
            "CryptoCompare collection failed",
            str(e)
        )