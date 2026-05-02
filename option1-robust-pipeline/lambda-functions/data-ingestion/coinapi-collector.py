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
COINAPI_PARAM_NAME = os.environ.get('COINAPI_PARAM_NAME', '/bitcoin/coinapi/key')

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

class CoinAPICollector:
    def __init__(self):
        self.api_key = None
        self.base_url = "https://rest.coinapi.io/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Bitcoin-Prediction-Pipeline/1.0'
        })

    def get_api_key(self) -> Optional[str]:
        """Retrieve API key from Parameter Store"""
        if not self.api_key:
            try:
                self.api_key = get_parameter(ssm, COINAPI_PARAM_NAME)
                logger.info("Successfully retrieved CoinAPI key from Parameter Store")
            except Exception as e:
                logger.warning(f"Failed to retrieve CoinAPI key: {str(e)}")
                logger.warning("CoinAPI will be skipped - consider adding the key to Parameter Store")
                return None
        return self.api_key

    @circuit_breaker
    @retry_handler
    def fetch_bitcoin_current_rate(self) -> Dict[str, Any]:
        """Fetch current Bitcoin exchange rate"""
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("CoinAPI key not available")
        
        # Get current BTC/USD rate
        url = f"{self.base_url}/exchangerate/BTC/USD"
        headers = {'X-CoinAPI-Key': api_key}
        
        logger.info("Fetching Bitcoin current rate from CoinAPI")
        start_time = time.time()
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            rate_data = response.json()
            
            if 'error' in rate_data:
                raise requests.RequestException(f"CoinAPI error: {rate_data['error']}")
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('CoinAPI.RateRequestDuration', duration, 'Seconds')
            
            logger.info(f"Successfully fetched current rate in {duration:.2f} seconds")
            
            return {
                'source': 'coinapi',
                'data_type': 'current_rate',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'rate_data': {
                    'asset_id_base': rate_data.get('asset_id_base', 'BTC'),
                    'asset_id_quote': rate_data.get('asset_id_quote', 'USD'),
                    'rate': rate_data.get('rate', 0),
                    'time': rate_data.get('time', datetime.now(timezone.utc).isoformat()),
                    'src_side_base': rate_data.get('src_side_base', []),
                    'src_side_quote': rate_data.get('src_side_quote', [])
                },
                'collection_timestamp': time.time()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinAPI rate request failed: {str(e)}")
            self.send_custom_metric('CoinAPI.RateRequestFailures', 1, 'Count')
            raise

    @circuit_breaker
    @retry_handler
    def fetch_bitcoin_ohlcv_data(self) -> Dict[str, Any]:
        """Fetch Bitcoin OHLCV data from multiple exchanges"""
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("CoinAPI key not available")
        
        # Get OHLCV data for last 24 hours
        period_id = "1HRS"  # 1 hour periods
        limit = 24  # Last 24 hours
        
        url = f"{self.base_url}/ohlcv/BTC/USD/history"
        headers = {'X-CoinAPI-Key': api_key}
        params = {
            'period_id': period_id,
            'limit': limit
        }
        
        logger.info("Fetching Bitcoin OHLCV data from CoinAPI")
        start_time = time.time()
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            ohlcv_data = response.json()
            
            if isinstance(ohlcv_data, dict) and 'error' in ohlcv_data:
                raise requests.RequestException(f"CoinAPI error: {ohlcv_data['error']}")
            
            # Process OHLCV data
            processed_data = self.process_ohlcv_data(ohlcv_data)
            
            # Log metrics
            duration = time.time() - start_time
            self.send_custom_metric('CoinAPI.OHLCVRequestDuration', duration, 'Seconds')
            self.send_custom_metric('CoinAPI.OHLCVDataPoints', len(processed_data), 'Count')
            
            logger.info(f"Successfully fetched {len(processed_data)} OHLCV data points in {duration:.2f} seconds")
            
            return {
                'source': 'coinapi',
                'data_type': 'ohlcv',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'ohlcv_data': processed_data,
                'collection_timestamp': time.time()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinAPI OHLCV request failed: {str(e)}")
            self.send_custom_metric('CoinAPI.OHLCVRequestFailures', 1, 'Count')
            raise

    @circuit_breaker
    @retry_handler
    def fetch_exchange_rates(self) -> Dict[str, Any]:
        """Fetch Bitcoin rates from multiple exchanges"""
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("CoinAPI key not available")
        
        # Get rates from major exchanges
        exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'BITSTAMP']
        all_rates = []
        
        headers = {'X-CoinAPI-Key': api_key}
        
        logger.info("Fetching Bitcoin exchange rates from CoinAPI")
        start_time = time.time()
        
        successful_exchanges = 0
        
        for exchange in exchanges:
            try:
                url = f"{self.base_url}/exchangerate/{exchange}_SPOT_BTC_USD"
                
                response = self.session.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                rate_data = response.json()
                
                if 'error' not in rate_data and 'rate' in rate_data:
                    all_rates.append({
                        'exchange': exchange,
                        'rate': rate_data.get('rate', 0),
                        'time': rate_data.get('time', datetime.now(timezone.utc).isoformat()),
                        'asset_id_base': rate_data.get('asset_id_base', 'BTC'),
                        'asset_id_quote': rate_data.get('asset_id_quote', 'USD')
                    })
                    successful_exchanges += 1
                
            except Exception as e:
                logger.warning(f"Failed to fetch rate from {exchange}: {str(e)}")
                continue
        
        if not all_rates:
            raise requests.RequestException("Failed to fetch rates from any exchange")
        
        # Calculate aggregated metrics
        rates = [r['rate'] for r in all_rates if r['rate'] > 0]
        
        aggregated_data = {
            'average_rate': sum(rates) / len(rates) if rates else 0,
            'min_rate': min(rates) if rates else 0,
            'max_rate': max(rates) if rates else 0,
            'rate_spread': max(rates) - min(rates) if rates else 0,
            'rate_spread_pct': ((max(rates) - min(rates)) / sum(rates) * len(rates)) * 100 if rates else 0,
            'exchange_count': len(all_rates)
        }
        
        # Log metrics
        duration = time.time() - start_time
        self.send_custom_metric('CoinAPI.ExchangeRateRequestDuration', duration, 'Seconds')
        self.send_custom_metric('CoinAPI.SuccessfulExchanges', successful_exchanges, 'Count')
        
        logger.info(f"Successfully fetched rates from {successful_exchanges} exchanges in {duration:.2f} seconds")
        
        return {
            'source': 'coinapi',
            'data_type': 'exchange_rates',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'exchange_rates': all_rates,
            'aggregated_metrics': aggregated_data,
            'collection_timestamp': time.time()
        }

    def process_ohlcv_data(self, ohlcv_data: List[Dict]) -> List[Dict[str, Any]]:
        """Process OHLCV data points"""
        processed_data = []
        
        for point in ohlcv_data:
            try:
                processed_point = {
                    'time_period_start': point.get('time_period_start', ''),
                    'time_period_end': point.get('time_period_end', ''),
                    'time_open': point.get('time_open', ''),
                    'time_close': point.get('time_close', ''),
                    'price_open': point.get('price_open', 0),
                    'price_high': point.get('price_high', 0),
                    'price_low': point.get('price_low', 0),
                    'price_close': point.get('price_close', 0),
                    'volume_traded': point.get('volume_traded', 0),
                    'trades_count': point.get('trades_count', 0)
                }
                
                # Calculate additional metrics
                if processed_point['price_high'] > 0 and processed_point['price_low'] > 0:
                    processed_point['price_range'] = processed_point['price_high'] - processed_point['price_low']
                    
                    if processed_point['price_close'] > 0:
                        processed_point['price_volatility'] = processed_point['price_range'] / processed_point['price_close']
                
                if processed_point['price_open'] > 0 and processed_point['price_close'] > 0:
                    price_change = processed_point['price_close'] - processed_point['price_open']
                    processed_point['price_change'] = price_change
                    processed_point['price_change_pct'] = (price_change / processed_point['price_open']) * 100
                
                processed_data.append(processed_point)
                
            except Exception as e:
                logger.warning(f"Failed to process OHLCV data point: {str(e)}")
                continue
        
        return processed_data

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
    collector = CoinAPICollector()
    
    try:
        logger.info("Starting CoinAPI data collection")
        
        # Check if API key is available
        if not collector.get_api_key():
            logger.warning("CoinAPI key not available, skipping collection")
            return create_success_response(
                "CoinAPI collection skipped",
                {
                    "reason": "API key not available",
                    "message": "Add CoinAPI key to Parameter Store at /bitcoin/coinapi/key to enable collection"
                }
            )
        
        # Check circuit breaker status
        if circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, skipping CoinAPI call")
            return create_error_response(
                "Circuit breaker open",
                "CoinAPI circuit breaker is currently open due to previous failures"
            )
        
        # Collect all data types
        collected_data = []
        
        # 1. Current rate (critical)
        try:
            current_rate = collector.fetch_bitcoin_current_rate()
            collected_data.append(current_rate)
        except Exception as e:
            logger.error(f"Failed to fetch current rate: {str(e)}")
            # Current rate is important but not critical enough to fail everything
        
        # 2. OHLCV data (important)
        try:
            ohlcv_data = collector.fetch_bitcoin_ohlcv_data()
            collected_data.append(ohlcv_data)
        except Exception as e:
            logger.warning(f"Failed to fetch OHLCV data: {str(e)}")
            # OHLCV is useful but not critical
        
        # 3. Exchange rates (nice-to-have)
        try:
            exchange_rates = collector.fetch_exchange_rates()
            collected_data.append(exchange_rates)
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rates: {str(e)}")
            # Exchange rate comparison is nice-to-have
        
        if not collected_data:
            logger.warning("No data collected from CoinAPI")
            return create_success_response(
                "CoinAPI collection completed with no data",
                {
                    "data_types_collected": [],
                    "messages_sent": 0,
                    "warning": "No data was successfully collected from CoinAPI"
                }
            )
        
        # Send data to SQS for processing
        messages_sent = 0
        
        for data_item in collected_data:
            message_body = json.dumps(data_item)
            success = send_to_sqs(
                sqs,
                SQS_QUEUE_URL,
                message_body,
                f"coinapi-{data_item['data_type']}"
            )
            
            if success:
                messages_sent += 1
            else:
                logger.error(f"Failed to send {data_item['data_type']} data to SQS")
        
        # Send final metrics
        collector.send_custom_metric('CoinAPI.MessagesSuccessfullySent', messages_sent, 'Count')
        collector.send_custom_metric('CoinAPI.TotalDataTypesCollected', len(collected_data), 'Count')
        
        logger.info(f"Successfully collected {len(collected_data)} data types and sent {messages_sent} messages to SQS")
        
        return create_success_response(
            "CoinAPI collection completed",
            {
                "data_types_collected": [item['data_type'] for item in collected_data],
                "messages_sent": messages_sent,
                "circuit_breaker_status": "closed" if not circuit_breaker.is_open() else "open"
            }
        )
        
    except Exception as e:
        logger.error(f"CoinAPI collection failed: {str(e)}")
        collector.send_custom_metric('CoinAPI.CollectionFailures', 1, 'Count')
        
        return create_error_response(
            "CoinAPI collection failed",
            str(e)
        )