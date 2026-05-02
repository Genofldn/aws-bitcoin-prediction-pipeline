import json
import boto3
import requests
import time
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import logging
import os
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
S3_BUCKET = os.environ['S3_BUCKET']
NEWSAPI_KEY = os.environ['NEWSAPI_KEY']
CRYPTOCOMPARE_KEY = os.environ['CRYPTOCOMPARE_KEY']
COINAPI_KEY = os.environ.get('COINAPI_KEY', '')

# Initialize AWS clients
s3 = boto3.client('s3')
ssm = boto3.client('ssm')

class SimpleBitcoinPredictor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Simple-Bitcoin-Predictor/1.0'
        })
        
        # Simple sentiment keywords
        self.positive_words = {
            'bull', 'bullish', 'surge', 'rally', 'moon', 'buy', 'pump', 'gains',
            'profit', 'breakthrough', 'adoption', 'institutional', 'approved'
        }
        
        self.negative_words = {
            'bear', 'bearish', 'crash', 'dump', 'sell', 'fear', 'panic', 'bubble',
            'scam', 'banned', 'regulation', 'decline', 'fall', 'drop', 'loss'
        }

    def get_api_key(self, param_name: str, default: str = '') -> str:
        """Get API key from environment or SSM Parameter Store"""
        try:
            # Try environment variable first
            env_key = os.environ.get(param_name.split('/')[-1].upper().replace('-', '_'), '')
            if env_key:
                return env_key
            
            # Fallback to SSM Parameter Store
            response = ssm.get_parameter(Name=param_name, WithDecryption=True)
            return response['Parameter']['Value']
        except Exception as e:
            logger.warning(f"Failed to get API key {param_name}: {str(e)}")
            return default

    def fetch_bitcoin_price(self) -> Optional[Dict]:
        """Fetch current Bitcoin price from CryptoCompare"""
        try:
            url = "https://min-api.cryptocompare.com/data/price"
            params = {
                'fsym': 'BTC',
                'tsyms': 'USD',
                'api_key': CRYPTOCOMPARE_KEY
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'USD' in data:
                return {
                    'price': data['USD'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'cryptocompare'
                }
            
        except Exception as e:
            logger.error(f"Failed to fetch price data: {str(e)}")
        
        return None

    def fetch_price_history(self) -> List[Dict]:
        """Fetch 24h price history"""
        try:
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': 24,
                'api_key': CRYPTOCOMPARE_KEY
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Response') == 'Success':
                history = []
                for point in data['Data']['Data']:
                    history.append({
                        'timestamp': point['time'],
                        'open': point['open'],
                        'high': point['high'], 
                        'low': point['low'],
                        'close': point['close'],
                        'volume': point['volumeto']
                    })
                return history
            
        except Exception as e:
            logger.error(f"Failed to fetch price history: {str(e)}")
        
        return []

    def fetch_bitcoin_news(self) -> List[Dict]:
        """Fetch Bitcoin news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency OR crypto OR BTC',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': NEWSAPI_KEY
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"NewsAPI response status: {data.get('status')}, articles count: {len(data.get('articles', []))}")
            
            if data.get('status') == 'ok':
                news = []
                for article in data.get('articles', []):
                    if article.get('title') and article.get('publishedAt'):
                        # Include recent articles (last 48 hours for better coverage)
                        try:
                            pub_time = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                            age_hours = (datetime.now(timezone.utc) - pub_time).total_seconds() / 3600
                        except:
                            age_hours = 0  # Include if parsing fails
                        
                        if age_hours <= 48:
                            news.append({
                                'title': article['title'],
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'published_at': article['publishedAt'],
                                'source': article.get('source', {}).get('name', ''),
                                'sentiment_score': self.analyze_sentiment(article['title'] + ' ' + article.get('description', ''))
                            })
                
                logger.info(f"Filtered news articles: {len(news)} articles from last 48 hours")
                return news
            
        except Exception as e:
            logger.error(f"Failed to fetch news: {str(e)}")
        
        return []

    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Normalize sentiment score (-1 to 1)
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment * 10))  # Amplify small differences

    def calculate_technical_indicators(self, price_history: List[Dict]) -> Dict:
        """Calculate basic technical indicators"""
        if len(price_history) < 2:
            return {}
        
        # Extract closing prices
        closes = [point['close'] for point in price_history if point['close'] > 0]
        highs = [point['high'] for point in price_history if point['high'] > 0]
        lows = [point['low'] for point in price_history if point['low'] > 0]
        volumes = [point['volume'] for point in price_history if point['volume'] > 0]
        
        if not closes:
            return {}
        
        indicators = {}
        
        # Current vs historical
        current_price = closes[-1]
        price_24h_ago = closes[0] if len(closes) > 1 else current_price
        
        # Price changes
        indicators['price_change_24h'] = current_price - price_24h_ago
        indicators['price_change_pct'] = (indicators['price_change_24h'] / price_24h_ago * 100) if price_24h_ago > 0 else 0
        
        # Volatility
        if len(closes) > 1:
            indicators['volatility'] = statistics.stdev(closes) / statistics.mean(closes)
        
        # Moving averages
        if len(closes) >= 12:
            sma_12 = sum(closes[-12:]) / 12
            indicators['sma_12'] = sma_12
            indicators['price_vs_sma'] = (current_price - sma_12) / sma_12 * 100
        
        # High/Low analysis
        if highs and lows:
            indicators['24h_high'] = max(highs)
            indicators['24h_low'] = min(lows)
            indicators['24h_range'] = indicators['24h_high'] - indicators['24h_low']
        
        # Volume analysis
        if volumes:
            indicators['avg_volume'] = statistics.mean(volumes)
            indicators['current_volume'] = volumes[-1]
        
        return indicators

    def make_simple_prediction(self, current_price: float, price_history: List[Dict], 
                             news_sentiment: float, technical_indicators: Dict) -> Dict:
        """Make simple Bitcoin price prediction using linear model"""
        
        # Base prediction on multiple factors
        prediction_factors = {
            'price_momentum': 0.0,
            'sentiment_factor': 0.0,
            'technical_factor': 0.0,
            'volume_factor': 0.0
        }
        
        # 1. Price momentum factor
        price_change_pct = technical_indicators.get('price_change_pct', 0)
        prediction_factors['price_momentum'] = price_change_pct * 0.3  # Momentum continuation
        
        # 2. Sentiment factor
        prediction_factors['sentiment_factor'] = news_sentiment * 2.0  # Sentiment impact
        
        # 3. Technical factor
        price_vs_sma = technical_indicators.get('price_vs_sma', 0)
        if price_vs_sma > 5:  # Price well above SMA
            prediction_factors['technical_factor'] = -1.0  # Mean reversion
        elif price_vs_sma < -5:  # Price well below SMA
            prediction_factors['technical_factor'] = 1.0   # Bounce back
        
        # 4. Volume factor
        if 'current_volume' in technical_indicators and 'avg_volume' in technical_indicators:
            volume_ratio = technical_indicators['current_volume'] / technical_indicators['avg_volume']
            if volume_ratio > 1.5:  # High volume
                prediction_factors['volume_factor'] = 0.5 if price_change_pct > 0 else -0.5
        
        # Combine factors for final prediction
        total_factor = sum(prediction_factors.values())
        
        # Convert to price change (limit to reasonable range)
        max_change_pct = 10.0  # Maximum 10% change prediction
        predicted_change_pct = max(-max_change_pct, min(max_change_pct, total_factor))
        
        # Generate predictions for different horizons
        predictions = {
            '1h': current_price * (1 + predicted_change_pct * 0.1 / 100),     # 10% of daily change
            '6h': current_price * (1 + predicted_change_pct * 0.25 / 100),    # 25% of daily change
            '24h': current_price * (1 + predicted_change_pct / 100),          # Full daily change
            '7d': current_price * (1 + predicted_change_pct * 2.0 / 100)      # Extrapolated weekly
        }
        
        # Calculate percentage changes
        changes = {}
        for horizon, price in predictions.items():
            changes[f'{horizon}_pct'] = ((price - current_price) / current_price) * 100
        
        # Calculate confidence based on data availability
        confidence_factors = [
            1.0 if current_price > 0 else 0.0,
            1.0 if len(price_history) >= 12 else 0.5,
            1.0 if abs(news_sentiment) > 0.1 else 0.7,
            1.0 if technical_indicators else 0.8
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors) * 0.7  # Conservative confidence
        
        return {
            'predictions': predictions,
            'changes': changes,
            'confidence': round(confidence, 3),
            'factors': prediction_factors,
            'method': 'simple_linear',
            'model_version': 'simple-v1.0'
        }

    def save_prediction_to_s3(self, prediction_data: Dict) -> bool:
        """Save prediction data to S3"""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create file path with date partitioning
            file_key = f"predictions/{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/{timestamp.hour:02d}/prediction-{int(timestamp.timestamp())}.json"
            
            # Upload to S3
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=file_key,
                Body=json.dumps(prediction_data, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Saved prediction to S3: {file_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save prediction to S3: {str(e)}")
            return False

    def get_historical_predictions(self, hours_back: int = 24) -> List[Dict]:
        """Get recent predictions from S3 for comparison"""
        try:
            now = datetime.now(timezone.utc)
            historical_data = []
            
            # Check last 24 hours of predictions
            for hour_offset in range(hours_back):
                check_time = now - timedelta(hours=hour_offset)
                prefix = f"predictions/{check_time.year}/{check_time.month:02d}/{check_time.day:02d}/{check_time.hour:02d}/"
                
                try:
                    response = s3.list_objects_v2(
                        Bucket=S3_BUCKET,
                        Prefix=prefix,
                        MaxKeys=5
                    )
                    
                    for obj in response.get('Contents', []):
                        file_response = s3.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                        file_content = json.loads(file_response['Body'].read())
                        historical_data.append(file_content)
                        
                except Exception:
                    continue  # Skip missing hours
            
            return historical_data[-10:]  # Return last 10 predictions
            
        except Exception as e:
            logger.warning(f"Failed to get historical predictions: {str(e)}")
            return []

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        logger.info("Starting simple Bitcoin prediction")
        start_time = time.time()
        
        # Initialize predictor
        predictor = SimpleBitcoinPredictor()
        
        # Collect data from all sources
        logger.info("Fetching data from APIs...")
        
        # Get current price and history
        current_price_data = predictor.fetch_bitcoin_price()
        if not current_price_data:
            logger.error("Failed to fetch current price")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to fetch current price'})
            }
        
        current_price = current_price_data['price']
        price_history = predictor.fetch_price_history()
        
        # Get news and sentiment
        news_articles = predictor.fetch_bitcoin_news()
        
        # Calculate overall sentiment
        if news_articles:
            sentiment_scores = [article['sentiment_score'] for article in news_articles]
            overall_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.0
        else:
            overall_sentiment = 0.0
        
        # Calculate technical indicators
        technical_indicators = predictor.calculate_technical_indicators(price_history)
        
        # Make prediction
        logger.info("Generating prediction...")
        prediction_result = predictor.make_simple_prediction(
            current_price=current_price,
            price_history=price_history,
            news_sentiment=overall_sentiment,
            technical_indicators=technical_indicators
        )
        
        # Get historical predictions for context
        historical_predictions = predictor.get_historical_predictions()
        
        # Create comprehensive result
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_data': {
                'price': current_price,
                'price_source': current_price_data['source'],
                'price_history_points': len(price_history),
                'news_articles_count': len(news_articles),
                'overall_sentiment': round(overall_sentiment, 3)
            },
            'technical_indicators': technical_indicators,
            'prediction': prediction_result,
            'news_articles': news_articles[:5],  # Include top 5 articles
            'historical_context': {
                'recent_predictions_count': len(historical_predictions),
                'processing_time_seconds': round(time.time() - start_time, 2)
            },
            'metadata': {
                'pipeline_version': 'simple-v1.0',
                'data_sources': ['NewsAPI', 'CryptoCompare'],
                'prediction_horizons': ['1h', '6h', '24h', '7d']
            }
        }
        
        # Save to S3
        save_success = predictor.save_prediction_to_s3(result)
        
        # Log results
        logger.info(f"Prediction completed in {result['historical_context']['processing_time_seconds']} seconds")
        logger.info(f"Current price: ${current_price:,.2f}")
        logger.info(f"24h prediction: ${prediction_result['predictions']['24h']:,.2f} ({prediction_result['changes']['24h_pct']:+.2f}%)")
        logger.info(f"Confidence: {prediction_result['confidence']:.1%}")
        logger.info(f"S3 save: {'SUCCESS' if save_success else 'FAILED'}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'current_price': current_price,
                'predictions': prediction_result['predictions'],
                'confidence': prediction_result['confidence'],
                's3_saved': save_success,
                'processing_time': result['historical_context']['processing_time_seconds']
            })
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }