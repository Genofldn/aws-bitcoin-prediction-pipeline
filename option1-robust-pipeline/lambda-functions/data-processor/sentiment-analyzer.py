import json
import boto3
import logging
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os
import sys

# Add shared utilities to path
sys.path.append('/opt/python/shared')
from utils import send_to_sqs, create_error_response, create_success_response

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
SQS_QUEUE_URL = os.environ['SQS_QUEUE_URL']

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')
cloudwatch = boto3.client('cloudwatch')

# Get table reference
table = dynamodb.Table(DYNAMODB_TABLE)

class SentimentAnalyzer:
    def __init__(self):
        # Bitcoin-specific sentiment keywords
        self.positive_keywords = {
            'bull', 'bullish', 'surge', 'rally', 'moon', 'lambo', 'hodl', 'buy',
            'pump', 'rocket', 'gains', 'profit', 'breakout', 'resistance',
            'adoption', 'institutional', 'etf', 'approved', 'breakthrough',
            'innovation', 'revolutionary', 'future', 'digital gold', 'store of value',
            'mainstream', 'acceptance', 'partnership', 'integration', 'upgrade',
            'halving', 'scarcity', 'demand', 'investment', 'portfolio'
        }
        
        self.negative_keywords = {
            'bear', 'bearish', 'crash', 'dump', 'sell', 'fear', 'panic',
            'bubble', 'scam', 'fraud', 'hack', 'stolen', 'banned', 'regulation',
            'crackdown', 'illegal', 'ponzi', 'volatile', 'risky', 'speculation',
            'manipulation', 'whale', 'sell-off', 'correction', 'decline',
            'dip', 'fall', 'plunge', 'tumble', 'drop', 'lose', 'loss',
            'uncertainty', 'concern', 'warning', 'caution', 'skeptical'
        }
        
        self.neutral_keywords = {
            'stable', 'sideways', 'consolidation', 'range', 'trading',
            'analysis', 'technical', 'fundamental', 'price', 'chart',
            'volume', 'market', 'blockchain', 'cryptocurrency', 'bitcoin',
            'btc', 'transaction', 'wallet', 'mining', 'node', 'protocol'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'massive': 2.2,
            'huge': 1.8, 'enormous': 2.0, 'tremendous': 1.9, 'incredible': 2.1,
            'amazing': 1.7, 'fantastic': 1.8, 'excellent': 1.6, 'outstanding': 1.9,
            'terrible': -2.0, 'horrible': -1.8, 'awful': -1.7, 'devastating': -2.2,
            'catastrophic': -2.5, 'disastrous': -2.3
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'none'}

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using keyword-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'positive_score': 0.0, 'negative_score': 0.0}
        
        # Clean and prepare text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'positive_score': 0.0, 'negative_score': 0.0}
        
        positive_score = 0.0
        negative_score = 0.0
        sentiment_words = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            elif i > 1 and words[i-2] in self.negations:
                is_negated = True
            
            # Check for intensifiers
            intensity_multiplier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity_multiplier = abs(self.intensifiers[words[i-1]])
            
            # Calculate sentiment for this word
            word_score = 0.0
            
            if word in self.positive_keywords:
                word_score = 1.0 * intensity_multiplier
                sentiment_words += 1
            elif word in self.negative_keywords:
                word_score = -1.0 * intensity_multiplier
                sentiment_words += 1
            elif word in self.neutral_keywords:
                sentiment_words += 1  # Count but don't add to score
            
            # Apply negation
            if is_negated and word_score != 0:
                word_score *= -0.8  # Reverse and slightly diminish
            
            # Add to appropriate score
            if word_score > 0:
                positive_score += word_score
            elif word_score < 0:
                negative_score += abs(word_score)
            
            i += 1
        
        # Calculate final sentiment score
        total_sentiment_words = max(sentiment_words, 1)
        normalized_positive = positive_score / total_sentiment_words
        normalized_negative = negative_score / total_sentiment_words
        
        # Final sentiment score (-1 to 1)
        sentiment_score = (normalized_positive - normalized_negative)
        
        # Confidence based on number of sentiment words found
        confidence = min(sentiment_words / len(words), 1.0)
        
        return {
            'sentiment_score': max(-1.0, min(1.0, sentiment_score)),
            'confidence': confidence,
            'positive_score': normalized_positive,
            'negative_score': normalized_negative,
            'sentiment_words_found': sentiment_words,
            'total_words': len(words)
        }

    def analyze_news_batch(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment for a batch of news articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Aggregated sentiment analysis
        """
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0
            }
        
        article_sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            try:
                # Combine title and description for analysis
                text_parts = []
                
                if article.get('title'):
                    text_parts.append(article['title'])
                if article.get('description'):
                    text_parts.append(article['description'])
                if article.get('content'):
                    # Limit content length to avoid processing very long articles
                    content = article['content'][:1000]
                    text_parts.append(content)
                
                combined_text = ' '.join(text_parts)
                
                # Analyze sentiment
                sentiment_result = self.analyze_text_sentiment(combined_text)
                
                # Weight by relevance score if available
                relevance_weight = article.get('relevance_score', 1.0)
                weighted_sentiment = sentiment_result['sentiment_score'] * relevance_weight
                
                article_sentiments.append({
                    'sentiment_score': sentiment_result['sentiment_score'],
                    'weighted_sentiment': weighted_sentiment,
                    'confidence': sentiment_result['confidence'],
                    'relevance_weight': relevance_weight,
                    'positive_score': sentiment_result['positive_score'],
                    'negative_score': sentiment_result['negative_score'],
                    'article_title': article.get('title', '')[:100]  # Truncate for storage
                })
                
                # Count sentiment categories
                if sentiment_result['sentiment_score'] > 0.1:
                    positive_count += 1
                elif sentiment_result['sentiment_score'] < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to analyze article sentiment: {str(e)}")
                continue
        
        if not article_sentiments:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0
            }
        
        # Calculate overall sentiment
        total_weighted_sentiment = sum(a['weighted_sentiment'] for a in article_sentiments)
        total_weight = sum(a['relevance_weight'] for a in article_sentiments)
        
        overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on number of articles and their individual confidences
        avg_confidence = sum(a['confidence'] for a in article_sentiments) / len(article_sentiments)
        count_factor = min(len(article_sentiments) / 10, 1.0)  # More articles = higher confidence
        overall_confidence = avg_confidence * count_factor
        
        return {
            'overall_sentiment': max(-1.0, min(1.0, overall_sentiment)),
            'confidence': overall_confidence,
            'article_count': len(article_sentiments),
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': neutral_count,
            'sentiment_distribution': {
                'positive_pct': (positive_count / len(article_sentiments)) * 100,
                'negative_pct': (negative_count / len(article_sentiments)) * 100,
                'neutral_pct': (neutral_count / len(article_sentiments)) * 100
            },
            'individual_sentiments': article_sentiments[:10]  # Store top 10 for debugging
        }

    def store_sentiment_data(self, sentiment_data: Dict[str, Any], source_data: Dict[str, Any]) -> bool:
        """
        Store sentiment analysis results in DynamoDB
        
        Args:
            sentiment_data: Processed sentiment analysis
            source_data: Original source data
            
        Returns:
            Success boolean
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            item = {
                'source': f"sentiment-{source_data.get('source', 'unknown')}",
                'timestamp': int(current_time.timestamp()),
                'id': f"sentiment-{source_data.get('source', 'unknown')}-{int(current_time.timestamp())}",
                'data_type': 'sentiment_analysis',
                'sentiment_score': sentiment_data['overall_sentiment'],
                'confidence': sentiment_data['confidence'],
                'article_count': sentiment_data['article_count'],
                'positive_articles': sentiment_data['positive_articles'],
                'negative_articles': sentiment_data['negative_articles'],
                'neutral_articles': sentiment_data['neutral_articles'],
                'sentiment_distribution': sentiment_data['sentiment_distribution'],
                'created_at': current_time.isoformat(),
                'ttl': int((current_time.timestamp() + 7 * 24 * 3600))  # 7 days TTL
            }
            
            # Add source-specific data
            if 'individual_sentiments' in sentiment_data:
                item['individual_sentiments'] = sentiment_data['individual_sentiments']
            
            table.put_item(Item=item)
            
            logger.info(f"Successfully stored sentiment data for {source_data.get('source', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store sentiment data: {str(e)}")
            return False

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
    """Main Lambda handler for sentiment analysis"""
    analyzer = SentimentAnalyzer()
    
    try:
        logger.info("Starting sentiment analysis processing")
        
        # Process SQS records
        processed_messages = 0
        failed_messages = 0
        
        for record in event.get('Records', []):
            try:
                # Parse message body
                message_body = json.loads(record['body'])
                
                # Skip non-news data
                if message_body.get('data_type') != 'news_batch' and message_body.get('data_type') != 'news':
                    logger.debug(f"Skipping non-news message: {message_body.get('data_type')}")
                    continue
                
                start_time = time.time()
                
                # Extract articles
                articles = []
                if message_body.get('data_type') == 'news_batch':
                    articles = message_body.get('articles', [])
                elif message_body.get('data_type') == 'news':
                    # Single article
                    articles = [message_body]
                
                if not articles:
                    logger.warning("No articles found in message")
                    continue
                
                # Analyze sentiment
                sentiment_result = analyzer.analyze_news_batch(articles)
                
                # Store in DynamoDB
                storage_success = analyzer.store_sentiment_data(sentiment_result, message_body)
                
                if storage_success:
                    # Send processed data to next stage
                    processed_data = {
                        'source': 'sentiment_analysis',
                        'data_type': 'processed_sentiment',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'sentiment_analysis': sentiment_result,
                        'original_source': message_body.get('source', 'unknown'),
                        'processing_duration': time.time() - start_time
                    }
                    
                    success = send_to_sqs(
                        sqs,
                        SQS_QUEUE_URL,
                        json.dumps(processed_data),
                        f"sentiment-{message_body.get('source', 'unknown')}"
                    )
                    
                    if success:
                        processed_messages += 1
                        logger.info(f"Successfully processed sentiment for {len(articles)} articles from {message_body.get('source', 'unknown')}")
                    else:
                        failed_messages += 1
                        logger.error("Failed to send processed sentiment data to SQS")
                else:
                    failed_messages += 1
                    logger.error("Failed to store sentiment data")
                
                # Send metrics
                duration = time.time() - start_time
                analyzer.send_custom_metric('SentimentAnalysis.ProcessingDuration', duration, 'Seconds')
                analyzer.send_custom_metric('SentimentAnalysis.ArticlesProcessed', len(articles), 'Count')
                analyzer.send_custom_metric('SentimentAnalysis.SentimentScore', sentiment_result['overall_sentiment'], 'None')
                
            except Exception as e:
                failed_messages += 1
                logger.error(f"Failed to process sentiment analysis message: {str(e)}")
                continue
        
        # Send final metrics
        analyzer.send_custom_metric('SentimentAnalysis.ProcessedMessages', processed_messages, 'Count')
        analyzer.send_custom_metric('SentimentAnalysis.FailedMessages', failed_messages, 'Count')
        
        logger.info(f"Sentiment analysis completed: {processed_messages} processed, {failed_messages} failed")
        
        return create_success_response(
            "Sentiment analysis completed",
            {
                "processed_messages": processed_messages,
                "failed_messages": failed_messages
            }
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis processing failed: {str(e)}")
        analyzer.send_custom_metric('SentimentAnalysis.ProcessingFailures', 1, 'Count')
        
        return create_error_response(
            "Sentiment analysis processing failed",
            str(e)
        )