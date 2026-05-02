import json
import boto3
import logging
import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add shared utilities to path
sys.path.append('/opt/python/shared')
from utils import create_error_response, create_success_response

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_RAW_TABLE = os.environ['DYNAMODB_RAW_TABLE']
DYNAMODB_PREDICTIONS_TABLE = os.environ['DYNAMODB_PREDICTIONS_TABLE']

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Get table references
raw_data_table = dynamodb.Table(DYNAMODB_RAW_TABLE)
predictions_table = dynamodb.Table(DYNAMODB_PREDICTIONS_TABLE)

class FeatureEngineer:
    def __init__(self):
        self.feature_cache = {}
        
    def fetch_recent_data(self, hours_back: int = 24) -> Dict[str, List[Dict]]:
        """
        Fetch recent data from DynamoDB for feature engineering
        
        Args:
            hours_back: How many hours of data to fetch
            
        Returns:
            Dictionary with data organized by source
        """
        cutoff_timestamp = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())
        
        try:
            # Scan for recent data (in production, use GSI for better performance)
            response = raw_data_table.scan(
                FilterExpression='#ts >= :cutoff',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_timestamp}
            )
            
            # Organize data by source
            organized_data = {}
            for item in response.get('Items', []):
                source = item.get('source', 'unknown')
                if source not in organized_data:
                    organized_data[source] = []
                organized_data[source].append(item)
            
            # Sort each source's data by timestamp
            for source in organized_data:
                organized_data[source].sort(key=lambda x: x.get('timestamp', 0))
            
            logger.info(f"Fetched {len(response.get('Items', []))} data points from {len(organized_data)} sources")
            return organized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch recent data: {str(e)}")
            return {}

    def extract_price_features(self, price_data: List[Dict]) -> Dict[str, float]:
        """
        Extract price-based features from historical data
        
        Args:
            price_data: List of price data points
            
        Returns:
            Dictionary of price features
        """
        features = {}
        
        if not price_data:
            return features
        
        try:
            # Extract prices from different sources
            cryptocompare_prices = []
            coinapi_prices = []
            current_prices = []
            
            for data_point in price_data:
                if data_point.get('source') == 'cryptocompare':
                    # Handle different CryptoCompare data types
                    if 'prices' in data_point:
                        usd_price = data_point['prices'].get('USD', 0)
                        if usd_price > 0:
                            cryptocompare_prices.append({
                                'price': usd_price,
                                'timestamp': data_point.get('timestamp', 0)
                            })
                    elif 'historical_data' in data_point:
                        for hist_point in data_point['historical_data']:
                            if hist_point.get('close', 0) > 0:
                                cryptocompare_prices.append({
                                    'price': hist_point['close'],
                                    'timestamp': hist_point.get('timestamp', 0)
                                })
                
                elif data_point.get('source') == 'coinapi':
                    if 'rate_data' in data_point:
                        rate = data_point['rate_data'].get('rate', 0)
                        if rate > 0:
                            coinapi_prices.append({
                                'price': rate,
                                'timestamp': data_point.get('timestamp', 0)
                            })
                    elif 'ohlcv_data' in data_point:
                        for ohlcv_point in data_point['ohlcv_data']:
                            if ohlcv_point.get('price_close', 0) > 0:
                                coinapi_prices.append({
                                    'price': ohlcv_point['price_close'],
                                    'timestamp': data_point.get('timestamp', 0)
                                })
            
            # Combine all prices
            all_prices = cryptocompare_prices + coinapi_prices
            all_prices.sort(key=lambda x: x['timestamp'])
            
            if len(all_prices) < 2:
                return features
            
            # Calculate price features
            prices = [p['price'] for p in all_prices]
            recent_prices = prices[-24:] if len(prices) >= 24 else prices  # Last 24 hours or available
            
            # Basic price statistics
            features['current_price'] = prices[-1]
            features['price_24h_ago'] = prices[-24] if len(prices) >= 24 else prices[0]
            features['price_12h_ago'] = prices[-12] if len(prices) >= 12 else prices[len(prices)//2]
            features['price_6h_ago'] = prices[-6] if len(prices) >= 6 else prices[max(1, len(prices)//4)]
            
            # Price changes
            features['price_change_24h'] = features['current_price'] - features['price_24h_ago']
            features['price_change_12h'] = features['current_price'] - features['price_12h_ago']
            features['price_change_6h'] = features['current_price'] - features['price_6h_ago']
            
            # Percentage changes
            features['price_change_pct_24h'] = (features['price_change_24h'] / features['price_24h_ago']) * 100 if features['price_24h_ago'] > 0 else 0
            features['price_change_pct_12h'] = (features['price_change_12h'] / features['price_12h_ago']) * 100 if features['price_12h_ago'] > 0 else 0
            features['price_change_pct_6h'] = (features['price_change_6h'] / features['price_6h_ago']) * 100 if features['price_6h_ago'] > 0 else 0
            
            # Statistical features
            features['price_mean_24h'] = statistics.mean(recent_prices)
            features['price_median_24h'] = statistics.median(recent_prices)
            features['price_std_24h'] = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
            features['price_min_24h'] = min(recent_prices)
            features['price_max_24h'] = max(recent_prices)
            features['price_range_24h'] = features['price_max_24h'] - features['price_min_24h']
            
            # Volatility indicators
            features['price_volatility'] = features['price_std_24h'] / features['price_mean_24h'] if features['price_mean_24h'] > 0 else 0
            features['price_coefficient_variation'] = features['price_volatility']
            
            # Trend indicators
            if len(prices) >= 3:
                # Simple trend calculation
                recent_trend = []
                for i in range(1, min(6, len(prices))):  # Last 5 changes
                    change = prices[-i] - prices[-i-1]
                    recent_trend.append(1 if change > 0 else -1 if change < 0 else 0)
                
                features['trend_direction'] = sum(recent_trend) / len(recent_trend)
                features['trend_strength'] = abs(features['trend_direction'])
            
            # Moving averages approximation
            if len(prices) >= 12:
                features['sma_12'] = sum(prices[-12:]) / 12
                features['price_vs_sma_12'] = (features['current_price'] - features['sma_12']) / features['sma_12'] * 100
            
            if len(prices) >= 6:
                features['sma_6'] = sum(prices[-6:]) / 6
                features['price_vs_sma_6'] = (features['current_price'] - features['sma_6']) / features['sma_6'] * 100
            
            logger.info(f"Extracted {len(features)} price features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract price features: {str(e)}")
            return {}

    def extract_sentiment_features(self, sentiment_data: List[Dict]) -> Dict[str, float]:
        """
        Extract sentiment-based features
        
        Args:
            sentiment_data: List of sentiment analysis results
            
        Returns:
            Dictionary of sentiment features
        """
        features = {}
        
        if not sentiment_data:
            return features
        
        try:
            # Filter for sentiment analysis data
            sentiment_points = []
            for data_point in sentiment_data:
                if data_point.get('source', '').startswith('sentiment-'):
                    sentiment_points.append(data_point)
            
            if not sentiment_points:
                return features
            
            # Sort by timestamp
            sentiment_points.sort(key=lambda x: x.get('timestamp', 0))
            
            # Extract sentiment scores
            sentiment_scores = []
            confidence_scores = []
            article_counts = []
            
            for point in sentiment_points:
                if 'sentiment_score' in point:
                    sentiment_scores.append(point['sentiment_score'])
                    confidence_scores.append(point.get('confidence', 0))
                    article_counts.append(point.get('article_count', 0))
            
            if not sentiment_scores:
                return features
            
            # Current sentiment features
            features['current_sentiment'] = sentiment_scores[-1]
            features['current_sentiment_confidence'] = confidence_scores[-1] if confidence_scores else 0
            features['current_article_count'] = article_counts[-1] if article_counts else 0
            
            # Historical sentiment features
            if len(sentiment_scores) > 1:
                features['sentiment_change'] = sentiment_scores[-1] - sentiment_scores[-2]
                features['sentiment_trend'] = features['sentiment_change']
            
            # Aggregated sentiment features
            features['avg_sentiment_24h'] = statistics.mean(sentiment_scores)
            features['sentiment_volatility'] = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            features['max_sentiment_24h'] = max(sentiment_scores)
            features['min_sentiment_24h'] = min(sentiment_scores)
            features['sentiment_range'] = features['max_sentiment_24h'] - features['min_sentiment_24h']
            
            # Sentiment distribution features
            positive_count = sum(1 for s in sentiment_scores if s > 0.1)
            negative_count = sum(1 for s in sentiment_scores if s < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            total = len(sentiment_scores)
            features['positive_sentiment_ratio'] = positive_count / total
            features['negative_sentiment_ratio'] = negative_count / total
            features['neutral_sentiment_ratio'] = neutral_count / total
            
            # Confidence-weighted sentiment
            if confidence_scores:
                weighted_sentiment = sum(s * c for s, c in zip(sentiment_scores, confidence_scores))
                total_confidence = sum(confidence_scores)
                features['weighted_sentiment'] = weighted_sentiment / total_confidence if total_confidence > 0 else 0
                features['avg_confidence'] = statistics.mean(confidence_scores)
            
            # Article volume features
            if article_counts:
                features['total_articles_24h'] = sum(article_counts)
                features['avg_articles_per_period'] = statistics.mean(article_counts)
                features['article_volume_trend'] = article_counts[-1] - article_counts[0] if len(article_counts) > 1 else 0
            
            logger.info(f"Extracted {len(features)} sentiment features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract sentiment features: {str(e)}")
            return {}

    def extract_volume_features(self, price_data: List[Dict]) -> Dict[str, float]:
        """
        Extract volume and market activity features
        
        Args:
            price_data: List of price/market data
            
        Returns:
            Dictionary of volume features
        """
        features = {}
        
        try:
            volumes = []
            trades_counts = []
            
            for data_point in price_data:
                # Extract volume data from different sources
                if data_point.get('source') == 'cryptocompare' and 'historical_data' in data_point:
                    for hist_point in data_point['historical_data']:
                        volume = hist_point.get('volume_to', 0)
                        if volume > 0:
                            volumes.append(volume)
                
                elif data_point.get('source') == 'coinapi':
                    if 'ohlcv_data' in data_point:
                        for ohlcv_point in data_point['ohlcv_data']:
                            volume = ohlcv_point.get('volume_traded', 0)
                            trades = ohlcv_point.get('trades_count', 0)
                            if volume > 0:
                                volumes.append(volume)
                            if trades > 0:
                                trades_counts.append(trades)
            
            if volumes:
                features['current_volume'] = volumes[-1] if volumes else 0
                features['avg_volume_24h'] = statistics.mean(volumes)
                features['volume_std_24h'] = statistics.stdev(volumes) if len(volumes) > 1 else 0
                features['max_volume_24h'] = max(volumes)
                features['min_volume_24h'] = min(volumes)
                features['volume_range_24h'] = features['max_volume_24h'] - features['min_volume_24h']
                
                # Volume trend
                if len(volumes) >= 2:
                    features['volume_change'] = volumes[-1] - volumes[-2]
                    features['volume_change_pct'] = (features['volume_change'] / volumes[-2]) * 100 if volumes[-2] > 0 else 0
                
                # Volume relative to average
                features['volume_vs_avg'] = (features['current_volume'] / features['avg_volume_24h']) if features['avg_volume_24h'] > 0 else 0
            
            if trades_counts:
                features['current_trades'] = trades_counts[-1] if trades_counts else 0
                features['avg_trades_24h'] = statistics.mean(trades_counts)
                features['trades_trend'] = trades_counts[-1] - trades_counts[0] if len(trades_counts) > 1 else 0
            
            logger.info(f"Extracted {len(features)} volume features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract volume features: {str(e)}")
            return {}

    def extract_temporal_features(self) -> Dict[str, float]:
        """
        Extract time-based features
        
        Returns:
            Dictionary of temporal features
        """
        try:
            now = datetime.now(timezone.utc)
            
            features = {
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),  # 0 = Monday
                'day_of_month': now.day,
                'month': now.month,
                'quarter': (now.month - 1) // 3 + 1,
                'is_weekend': 1 if now.weekday() >= 5 else 0,
                'is_business_hours_utc': 1 if 8 <= now.hour <= 17 else 0,
                'is_us_business_hours': 1 if 13 <= now.hour <= 22 else 0,  # 9 AM - 6 PM EST in UTC
                'is_asia_business_hours': 1 if 0 <= now.hour <= 9 else 0,   # Approximate Asia business hours
            }
            
            # Cyclical encoding for periodic features
            import math
            features['hour_sin'] = math.sin(2 * math.pi * now.hour / 24)
            features['hour_cos'] = math.cos(2 * math.pi * now.hour / 24)
            features['day_sin'] = math.sin(2 * math.pi * now.weekday() / 7)
            features['day_cos'] = math.cos(2 * math.pi * now.weekday() / 7)
            features['month_sin'] = math.sin(2 * math.pi * now.month / 12)
            features['month_cos'] = math.cos(2 * math.pi * now.month / 12)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract temporal features: {str(e)}")
            return {}

    def calculate_technical_indicators(self, price_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate technical indicators
        
        Args:
            price_data: Price data points
            
        Returns:
            Dictionary of technical indicators
        """
        features = {}
        
        try:
            # Extract price series
            prices = []
            volumes = []
            highs = []
            lows = []
            
            for data_point in price_data:
                if data_point.get('source') == 'cryptocompare' and 'historical_data' in data_point:
                    for hist_point in data_point['historical_data']:
                        if hist_point.get('close', 0) > 0:
                            prices.append(hist_point['close'])
                            highs.append(hist_point.get('high', hist_point['close']))
                            lows.append(hist_point.get('low', hist_point['close']))
                            volumes.append(hist_point.get('volume_to', 0))
            
            if len(prices) < 14:  # Need at least 14 periods for RSI
                return features
            
            # RSI (Relative Strength Index)
            features.update(self._calculate_rsi(prices))
            
            # Bollinger Bands
            features.update(self._calculate_bollinger_bands(prices))
            
            # MACD approximation
            features.update(self._calculate_macd(prices))
            
            # Support/Resistance levels
            features.update(self._calculate_support_resistance(highs, lows))
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Dict[str, float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return {}
        
        try:
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))
            
            if len(gains) < period:
                return {}
            
            # Average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            return {
                'rsi': rsi,
                'rsi_overbought': 1 if rsi > 70 else 0,
                'rsi_oversold': 1 if rsi < 30 else 0
            }
        except:
            return {}

    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {}
        
        try:
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / period
            variance = sum((p - sma) ** 2 for p in recent_prices) / period
            std = variance ** 0.5
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current_price = prices[-1]
            
            return {
                'bb_upper': upper_band,
                'bb_lower': lower_band,
                'bb_middle': sma,
                'bb_width': upper_band - lower_band,
                'bb_position': (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
            }
        except:
            return {}

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD approximation"""
        if len(prices) < 26:
            return {}
        
        try:
            # Simple approximation of MACD
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            if ema_12 is not None and ema_26 is not None:
                macd = ema_12 - ema_26
                return {
                    'macd': macd,
                    'macd_signal': 1 if macd > 0 else -1
                }
        except:
            pass
        
        return {}

    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        try:
            multiplier = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
        except:
            return None

    def _calculate_support_resistance(self, highs: List[float], lows: List[float]) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            if not highs or not lows:
                return {}
            
            # Simple support/resistance calculation
            recent_highs = highs[-10:] if len(highs) >= 10 else highs
            recent_lows = lows[-10:] if len(lows) >= 10 else lows
            
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            return {
                'resistance_level': resistance,
                'support_level': support,
                'support_resistance_range': resistance - support
            }
        except:
            return {}

    def create_feature_vector(self, all_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Create final feature vector with metadata
        
        Args:
            all_features: Combined features from all sources
            
        Returns:
            Feature vector with metadata
        """
        current_time = datetime.now(timezone.utc)
        
        feature_vector = {
            'timestamp': current_time.isoformat(),
            'unix_timestamp': int(current_time.timestamp()),
            'feature_count': len(all_features),
            'features': all_features,
            'feature_metadata': {
                'created_at': current_time.isoformat(),
                'version': '1.0',
                'data_quality_score': self._calculate_data_quality_score(all_features)
            }
        }
        
        return feature_vector

    def _calculate_data_quality_score(self, features: Dict[str, float]) -> float:
        """Calculate data quality score based on available features"""
        try:
            total_possible_features = 100  # Estimated maximum features
            available_features = len(features)
            
            # Check for critical features
            critical_features = ['current_price', 'current_sentiment']
            critical_score = sum(1 for f in critical_features if f in features) / len(critical_features)
            
            # Coverage score
            coverage_score = min(available_features / total_possible_features, 1.0)
            
            # Combined score
            quality_score = (critical_score * 0.7) + (coverage_score * 0.3)
            
            return round(quality_score, 3)
        except:
            return 0.5

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
    """Main Lambda handler for feature engineering"""
    engineer = FeatureEngineer()
    
    try:
        logger.info("Starting feature engineering process")
        start_time = time.time()
        
        # Fetch recent data
        recent_data = engineer.fetch_recent_data(hours_back=24)
        
        if not recent_data:
            logger.warning("No recent data available for feature engineering")
            return create_success_response(
                "Feature engineering completed with no data",
                {"features_extracted": 0}
            )
        
        # Combine all data for processing
        all_data = []
        for source_data in recent_data.values():
            all_data.extend(source_data)
        
        # Extract features from different data types
        all_features = {}
        
        # Price features
        price_features = engineer.extract_price_features(all_data)
        all_features.update(price_features)
        
        # Sentiment features
        sentiment_features = engineer.extract_sentiment_features(all_data)
        all_features.update(sentiment_features)
        
        # Volume features
        volume_features = engineer.extract_volume_features(all_data)
        all_features.update(volume_features)
        
        # Temporal features
        temporal_features = engineer.extract_temporal_features()
        all_features.update(temporal_features)
        
        # Technical indicators
        technical_features = engineer.calculate_technical_indicators(all_data)
        all_features.update(technical_features)
        
        # Create final feature vector
        feature_vector = engineer.create_feature_vector(all_features)
        
        # Store feature vector in DynamoDB
        try:
            feature_item = {
                'id': f"features-{int(datetime.now(timezone.utc).timestamp())}",
                'timestamp': feature_vector['unix_timestamp'],
                'data_type': 'feature_vector',
                'features': feature_vector['features'],
                'feature_count': feature_vector['feature_count'],
                'data_quality_score': feature_vector['feature_metadata']['data_quality_score'],
                'created_at': feature_vector['timestamp'],
                'ttl': int((datetime.now(timezone.utc).timestamp() + 30 * 24 * 3600))  # 30 days TTL
            }
            
            predictions_table.put_item(Item=feature_item)
            logger.info("Successfully stored feature vector in DynamoDB")
            
        except Exception as e:
            logger.error(f"Failed to store feature vector: {str(e)}")
        
        # Send metrics
        duration = time.time() - start_time
        engineer.send_custom_metric('FeatureEngineering.ProcessingDuration', duration, 'Seconds')
        engineer.send_custom_metric('FeatureEngineering.FeaturesExtracted', len(all_features), 'Count')
        engineer.send_custom_metric('FeatureEngineering.DataQualityScore', feature_vector['feature_metadata']['data_quality_score'], 'None')
        engineer.send_custom_metric('FeatureEngineering.DataSources', len(recent_data), 'Count')
        
        logger.info(f"Feature engineering completed: {len(all_features)} features extracted in {duration:.2f} seconds")
        
        return create_success_response(
            "Feature engineering completed",
            {
                "features_extracted": len(all_features),
                "data_quality_score": feature_vector['feature_metadata']['data_quality_score'],
                "processing_duration": duration,
                "data_sources": len(recent_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        engineer.send_custom_metric('FeatureEngineering.ProcessingFailures', 1, 'Count')
        
        return create_error_response(
            "Feature engineering failed",
            str(e)
        )