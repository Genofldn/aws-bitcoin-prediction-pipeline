import json
import boto3
import logging
import time
import pickle
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import sys
import numpy as np

# Add shared utilities to path
sys.path.append('/opt/python/shared')
from utils import create_error_response, create_success_response

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_PREDICTIONS_TABLE = os.environ['DYNAMODB_PREDICTIONS_TABLE']
S3_BUCKET = os.environ['S3_BUCKET']

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

# Get table reference
predictions_table = dynamodb.Table(DYNAMODB_PREDICTIONS_TABLE)

class BitcoinPredictor:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.model_version = "1.0"
        
    def load_model_from_s3(self) -> bool:
        """
        Load trained model from S3
        
        Returns:
            Success boolean
        """
        try:
            # Try to load the model from S3
            model_key = "models/bitcoin_predictor_model.pkl"
            
            try:
                response = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
                model_data = response['Body'].read()
                
                # Load model components
                model_components = pickle.loads(model_data)
                self.model = model_components.get('model')
                self.feature_scaler = model_components.get('feature_scaler')
                self.target_scaler = model_components.get('target_scaler')
                self.feature_columns = model_components.get('feature_columns', [])
                self.model_version = model_components.get('version', '1.0')
                
                logger.info(f"Successfully loaded model version {self.model_version} from S3")
                return True
                
            except s3.exceptions.NoSuchKey:
                logger.warning("No trained model found in S3, using fallback model")
                return self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"Failed to load model from S3: {str(e)}")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> bool:
        """
        Create a simple fallback model for basic predictions
        
        Returns:
            Success boolean
        """
        try:
            logger.info("Creating fallback linear regression model")
            
            # Simple feature list for fallback
            self.feature_columns = [
                'current_price', 'price_change_24h', 'price_change_pct_24h',
                'current_sentiment', 'avg_sentiment_24h', 'price_volatility',
                'hour_of_day', 'day_of_week', 'is_weekend'
            ]
            
            # Create a simple model (coefficients for linear combination)
            self.model = {
                'type': 'linear_fallback',
                'coefficients': {
                    'current_price': 0.0001,  # Small positive trend
                    'price_change_24h': 0.3,  # Recent changes matter
                    'price_change_pct_24h': 0.2,
                    'current_sentiment': 1000.0,  # Sentiment impact
                    'avg_sentiment_24h': 500.0,
                    'price_volatility': -500.0,  # High volatility negative
                    'hour_of_day': 0.0,
                    'day_of_week': 0.0,
                    'is_weekend': -100.0,  # Weekend slightly negative
                    'intercept': 0.0
                }
            }
            
            self.feature_scaler = None  # No scaling for fallback
            self.target_scaler = None
            self.model_version = "fallback-1.0"
            
            logger.info("Fallback model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {str(e)}")
            return False

    def fetch_latest_features(self) -> Optional[Dict[str, float]]:
        """
        Fetch the latest feature vector from DynamoDB
        
        Returns:
            Feature dictionary or None
        """
        try:
            # Query for the latest feature vector
            response = predictions_table.query(
                IndexName='prediction-type-timestamp-index',
                KeyConditionExpression='prediction_type = :pt',
                ExpressionAttributeValues={':pt': 'feature_vector'},
                ScanIndexForward=False,  # Latest first
                Limit=1
            )
            
            items = response.get('Items', [])
            if not items:
                logger.warning("No feature vectors found in database")
                return None
            
            latest_features = items[0].get('features', {})
            logger.info(f"Retrieved {len(latest_features)} features from latest feature vector")
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Failed to fetch latest features: {str(e)}")
            return None

    def prepare_features(self, raw_features: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Prepare features for model prediction
        
        Args:
            raw_features: Raw feature dictionary
            
        Returns:
            Prepared feature array or None
        """
        try:
            # Extract features in the correct order
            feature_values = []
            
            for feature_name in self.feature_columns:
                value = raw_features.get(feature_name, 0.0)
                
                # Handle missing values with defaults
                if value is None or (isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value))):
                    value = self._get_default_value(feature_name)
                
                feature_values.append(float(value))
            
            # Convert to numpy array
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Apply scaling if available
            if self.feature_scaler is not None:
                feature_array = self.feature_scaler.transform(feature_array)
            
            logger.info(f"Prepared feature array with {len(feature_values)} features")
            return feature_array
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {str(e)}")
            return None

    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for missing features"""
        defaults = {
            'current_price': 50000.0,
            'price_change_24h': 0.0,
            'price_change_pct_24h': 0.0,
            'current_sentiment': 0.0,
            'avg_sentiment_24h': 0.0,
            'price_volatility': 0.02,
            'hour_of_day': 12.0,
            'day_of_week': 2.0,
            'is_weekend': 0.0
        }
        return defaults.get(feature_name, 0.0)

    def make_predictions(self, features: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        Make Bitcoin price predictions
        
        Args:
            features: Prepared feature array
            current_price: Current Bitcoin price
            
        Returns:
            Dictionary with predictions
        """
        try:
            predictions = {}
            
            if self.model['type'] == 'linear_fallback':
                # Use fallback linear model
                predictions = self._make_fallback_predictions(features, current_price)
            else:
                # Use trained ML model
                predictions = self._make_ml_predictions(features, current_price)
            
            # Add metadata
            predictions.update({
                'model_version': self.model_version,
                'prediction_timestamp': datetime.now(timezone.utc).isoformat(),
                'confidence_score': self._calculate_confidence(features, current_price),
                'current_price': current_price
            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {str(e)}")
            return self._create_safe_predictions(current_price)

    def _make_fallback_predictions(self, features: np.ndarray, current_price: float) -> Dict[str, Any]:
        """Make predictions using fallback linear model"""
        try:
            coefficients = self.model['coefficients']
            feature_array = features.flatten()
            
            # Calculate base prediction using linear combination
            prediction_change = coefficients['intercept']
            
            for i, feature_name in enumerate(self.feature_columns):
                if i < len(feature_array):
                    prediction_change += feature_array[i] * coefficients.get(feature_name, 0.0)
            
            # Generate predictions for different time horizons
            predictions = {
                '1h': current_price + (prediction_change * 0.1),  # 10% of daily change for 1h
                '6h': current_price + (prediction_change * 0.25), # 25% of daily change for 6h
                '24h': current_price + prediction_change,         # Full change for 24h
                '7d': current_price + (prediction_change * 3.5)   # Extrapolate for 7 days
            }
            
            # Calculate percentage changes
            changes = {}
            for horizon, price in predictions.items():
                change_pct = ((price - current_price) / current_price) * 100
                changes[f'{horizon}_pct'] = change_pct
            
            return {
                'predictions': predictions,
                'changes': changes,
                'method': 'fallback_linear'
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {str(e)}")
            return self._create_safe_predictions(current_price)

    def _make_ml_predictions(self, features: np.ndarray, current_price: float) -> Dict[str, Any]:
        """Make predictions using trained ML model"""
        try:
            # This would use the actual trained model
            # For now, using a sophisticated fallback
            raw_predictions = self.model.predict(features)
            
            # If model predicts changes rather than absolute prices
            if self.target_scaler:
                raw_predictions = self.target_scaler.inverse_transform(raw_predictions.reshape(-1, 1))
            
            # Generate multi-horizon predictions
            base_change = float(raw_predictions[0])
            
            predictions = {
                '1h': current_price + (base_change * 0.1),
                '6h': current_price + (base_change * 0.3),
                '24h': current_price + base_change,
                '7d': current_price + (base_change * 4.0)
            }
            
            # Calculate percentage changes
            changes = {}
            for horizon, price in predictions.items():
                change_pct = ((price - current_price) / current_price) * 100
                changes[f'{horizon}_pct'] = change_pct
            
            return {
                'predictions': predictions,
                'changes': changes,
                'method': 'ml_model'
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            return self._make_fallback_predictions(features, current_price)

    def _create_safe_predictions(self, current_price: float) -> Dict[str, Any]:
        """Create safe default predictions when all else fails"""
        # Very conservative predictions - minimal change
        predictions = {
            '1h': current_price * 1.001,   # 0.1% increase
            '6h': current_price * 1.002,   # 0.2% increase
            '24h': current_price * 1.005,  # 0.5% increase
            '7d': current_price * 1.02     # 2% increase
        }
        
        changes = {}
        for horizon, price in predictions.items():
            change_pct = ((price - current_price) / current_price) * 100
            changes[f'{horizon}_pct'] = change_pct
        
        return {
            'predictions': predictions,
            'changes': changes,
            'method': 'safe_default',
            'warning': 'Using conservative default predictions due to model unavailability'
        }

    def _calculate_confidence(self, features: np.ndarray, current_price: float) -> float:
        """Calculate prediction confidence score"""
        try:
            # Base confidence on data availability and model type
            base_confidence = 0.7 if self.model['type'] != 'linear_fallback' else 0.4
            
            # Adjust based on feature completeness
            feature_completeness = np.sum(features != 0) / len(features.flatten())
            confidence = base_confidence * feature_completeness
            
            # Adjust for price volatility (lower confidence for high volatility)
            if len(features.flatten()) > 5:  # If volatility feature is available
                volatility_penalty = min(features.flatten()[5] * 0.5, 0.3)
                confidence = max(confidence - volatility_penalty, 0.1)
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {str(e)}")
            return 0.5

    def store_predictions(self, predictions: Dict[str, Any], features_used: Dict[str, float]) -> bool:
        """
        Store predictions in DynamoDB
        
        Args:
            predictions: Prediction results
            features_used: Features used for prediction
            
        Returns:
            Success boolean
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Create prediction record
            prediction_item = {
                'id': f"prediction-{int(current_time.timestamp())}",
                'timestamp': int(current_time.timestamp()),
                'prediction_type': 'bitcoin_price',
                'predictions': predictions['predictions'],
                'changes': predictions['changes'],
                'confidence_score': predictions['confidence_score'],
                'current_price': predictions['current_price'],
                'model_version': predictions['model_version'],
                'method': predictions['method'],
                'created_at': current_time.isoformat(),
                'feature_count': len(features_used),
                'ttl': int((current_time.timestamp() + 90 * 24 * 3600))  # 90 days TTL
            }
            
            # Add warning if present
            if 'warning' in predictions:
                prediction_item['warning'] = predictions['warning']
            
            # Store summary of features used (top 10 most important)
            important_features = {}
            for feature, value in list(features_used.items())[:10]:
                important_features[feature] = value
            prediction_item['key_features'] = important_features
            
            predictions_table.put_item(Item=prediction_item)
            
            logger.info("Successfully stored predictions in DynamoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {str(e)}")
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
    """Main Lambda handler for Bitcoin prediction"""
    predictor = BitcoinPredictor()
    
    try:
        logger.info("Starting Bitcoin price prediction")
        start_time = time.time()
        
        # Load model
        model_loaded = predictor.load_model_from_s3()
        if not model_loaded:
            logger.error("Failed to load any model")
            return create_error_response(
                "Model loading failed",
                "Could not load prediction model"
            )
        
        # Fetch latest features
        latest_features = predictor.fetch_latest_features()
        if not latest_features:
            logger.error("No features available for prediction")
            return create_error_response(
                "No features available",
                "Cannot make predictions without feature data"
            )
        
        # Get current price from features
        current_price = latest_features.get('current_price', 50000.0)
        
        # Prepare features for model
        prepared_features = predictor.prepare_features(latest_features)
        if prepared_features is None:
            logger.error("Failed to prepare features")
            return create_error_response(
                "Feature preparation failed",
                "Could not prepare features for prediction"
            )
        
        # Make predictions
        predictions = predictor.make_predictions(prepared_features, current_price)
        
        # Store predictions
        storage_success = predictor.store_predictions(predictions, latest_features)
        
        # Send metrics
        duration = time.time() - start_time
        predictor.send_custom_metric('Prediction.ProcessingDuration', duration, 'Seconds')
        predictor.send_custom_metric('Prediction.ConfidenceScore', predictions['confidence_score'], 'None')
        predictor.send_custom_metric('Prediction.CurrentPrice', current_price, 'None')
        
        # Send prediction metrics for each horizon
        for horizon, price in predictions['predictions'].items():
            change_pct = predictions['changes'].get(f'{horizon}_pct', 0)
            predictor.send_custom_metric(f'Prediction.Change.{horizon}', change_pct, 'Percent')
        
        if storage_success:
            predictor.send_custom_metric('Prediction.StorageSuccess', 1, 'Count')
        else:
            predictor.send_custom_metric('Prediction.StorageFailure', 1, 'Count')
        
        logger.info(f"Bitcoin prediction completed in {duration:.2f} seconds")
        logger.info(f"Predictions: 1h: ${predictions['predictions']['1h']:.2f}, "
                   f"24h: ${predictions['predictions']['24h']:.2f}, "
                   f"7d: ${predictions['predictions']['7d']:.2f}")
        
        return create_success_response(
            "Bitcoin prediction completed",
            {
                "predictions": predictions['predictions'],
                "changes": predictions['changes'],
                "confidence_score": predictions['confidence_score'],
                "model_version": predictions['model_version'],
                "method": predictions['method'],
                "processing_duration": duration,
                "storage_success": storage_success
            }
        )
        
    except Exception as e:
        logger.error(f"Bitcoin prediction failed: {str(e)}")
        predictor.send_custom_metric('Prediction.ProcessingFailures', 1, 'Count')
        
        return create_error_response(
            "Bitcoin prediction failed",
            str(e)
        )