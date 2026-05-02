import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinModelTrainer:
    def __init__(self, dynamodb_table_name: str, s3_bucket: str):
        """
        Initialize the Bitcoin model trainer
        
        Args:
            dynamodb_table_name: Name of the DynamoDB table with training data
            s3_bucket: S3 bucket name for storing models
        """
        self.dynamodb = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        self.table = self.dynamodb.Table(dynamodb_table_name)
        self.s3_bucket = s3_bucket
        
        # Model configurations
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'ridge_regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        self.feature_columns = [
            # Price features
            'current_price', 'price_change_24h', 'price_change_pct_24h',
            'price_change_12h', 'price_change_pct_12h', 'price_change_6h', 'price_change_pct_6h',
            'price_mean_24h', 'price_std_24h', 'price_volatility',
            'price_min_24h', 'price_max_24h', 'price_range_24h',
            
            # Technical indicators
            'rsi', 'bb_position', 'bb_width', 'macd',
            'sma_12', 'sma_6', 'price_vs_sma_12', 'price_vs_sma_6',
            
            # Sentiment features
            'current_sentiment', 'avg_sentiment_24h', 'sentiment_volatility',
            'positive_sentiment_ratio', 'negative_sentiment_ratio',
            'weighted_sentiment', 'avg_confidence',
            
            # Volume features
            'current_volume', 'avg_volume_24h', 'volume_change_pct',
            'volume_vs_avg', 'current_trades', 'avg_trades_24h',
            
            # Temporal features
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours_utc',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            
            # Market structure
            'resistance_level', 'support_level', 'support_resistance_range'
        ]
        
        self.target_horizons = ['1h', '6h', '24h', '7d']

    def fetch_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch training data from DynamoDB
        
        Args:
            days_back: Number of days of historical data to fetch
            
        Returns:
            DataFrame with training data
        """
        try:
            cutoff_timestamp = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
            
            logger.info(f"Fetching training data from last {days_back} days")
            
            # Scan for feature vectors and predictions
            feature_response = self.table.scan(
                FilterExpression='#ts >= :cutoff AND data_type = :dt',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':cutoff': cutoff_timestamp,
                    ':dt': 'feature_vector'
                }
            )
            
            prediction_response = self.table.scan(
                FilterExpression='#ts >= :cutoff AND prediction_type = :pt',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':cutoff': cutoff_timestamp,
                    ':pt': 'bitcoin_price'
                }
            )
            
            # Convert to DataFrames
            features_df = pd.DataFrame(feature_response.get('Items', []))
            predictions_df = pd.DataFrame(prediction_response.get('Items', []))
            
            if features_df.empty:
                logger.warning("No feature data found for training")
                return pd.DataFrame()
            
            # Process features
            training_data = []
            
            for _, feature_row in features_df.iterrows():
                try:
                    feature_dict = feature_row.get('features', {})
                    timestamp = feature_row.get('timestamp', 0)
                    
                    # Find corresponding future prices for targets
                    future_prices = self._get_future_prices(timestamp, predictions_df)
                    
                    if future_prices:
                        row_data = {'timestamp': timestamp}
                        row_data.update(feature_dict)
                        row_data.update(future_prices)
                        training_data.append(row_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process feature row: {str(e)}")
                    continue
            
            if not training_data:
                logger.warning("No valid training data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(training_data)
            logger.info(f"Fetched {len(df)} training samples with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch training data: {str(e)}")
            return pd.DataFrame()

    def _get_future_prices(self, timestamp: int, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get future prices for target creation
        
        Args:
            timestamp: Current timestamp
            predictions_df: DataFrame with historical predictions
            
        Returns:
            Dictionary with target prices
        """
        targets = {}
        
        try:
            # Define time windows for each horizon (in seconds)
            horizons = {
                '1h': 3600,
                '6h': 6 * 3600,
                '24h': 24 * 3600,
                '7d': 7 * 24 * 3600
            }
            
            for horizon, seconds in horizons.items():
                target_timestamp = timestamp + seconds
                
                # Find closest prediction within a reasonable window
                time_diff = abs(predictions_df['timestamp'] - target_timestamp)
                closest_idx = time_diff.idxmin()
                
                if time_diff.iloc[closest_idx] < 3600:  # Within 1 hour
                    closest_prediction = predictions_df.iloc[closest_idx]
                    actual_price = closest_prediction.get('current_price', 0)
                    
                    if actual_price > 0:
                        targets[f'target_{horizon}'] = actual_price
            
            return targets
            
        except Exception as e:
            logger.warning(f"Failed to get future prices: {str(e)}")
            return {}

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and targets for training
        
        Args:
            df: Raw training DataFrame
            
        Returns:
            Tuple of (features, targets_dict)
        """
        try:
            # Extract available features
            available_features = []
            for col in self.feature_columns:
                if col in df.columns:
                    available_features.append(col)
            
            if not available_features:
                logger.error("No features available in training data")
                return np.array([]), {}
            
            logger.info(f"Using {len(available_features)} features: {available_features}")
            
            # Prepare feature matrix
            X = df[available_features].fillna(0).values
            
            # Prepare targets for each horizon
            targets = {}
            for horizon in self.target_horizons:
                target_col = f'target_{horizon}'
                if target_col in df.columns:
                    # Calculate price changes instead of absolute prices
                    current_prices = df['current_price'].values
                    target_prices = df[target_col].fillna(current_prices).values
                    
                    # Target is the price change
                    price_changes = target_prices - current_prices
                    targets[horizon] = price_changes
            
            if not targets:
                logger.error("No target variables found")
                return X, {}
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features, {len(targets)} targets")
            
            # Store feature column names for later use
            self.used_feature_columns = available_features
            
            return X, targets
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return np.array([]), {}

    def train_models(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Train models for all horizons
        
        Args:
            X: Feature matrix
            targets: Target dictionary
            
        Returns:
            Dictionary with trained models and metadata
        """
        try:
            if X.shape[0] == 0:
                logger.error("No training data available")
                return {}
            
            # Split data for training and validation
            X_train, X_test, indices_train, indices_test = train_test_split(
                X, np.arange(len(X)), test_size=0.2, random_state=42
            )
            
            # Scale features
            feature_scaler = RobustScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_test_scaled = feature_scaler.transform(X_test)
            
            trained_models = {}
            
            # Train models for each horizon
            for horizon in targets.keys():
                logger.info(f"Training models for {horizon} horizon")
                
                y_train = targets[horizon][indices_train]
                y_test = targets[horizon][indices_test]
                
                # Scale targets
                target_scaler = RobustScaler()
                y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                
                horizon_models = {}
                
                # Train each model type
                for model_name, model in self.models.items():
                    try:
                        logger.info(f"Training {model_name} for {horizon}")
                        
                        # Train model
                        model.fit(X_train_scaled, y_train_scaled)
                        
                        # Evaluate model
                        train_pred = model.predict(X_train_scaled)
                        test_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics on scaled data
                        train_mse = mean_squared_error(y_train_scaled, train_pred)
                        test_mse = mean_squared_error(y_test_scaled, test_pred)
                        train_r2 = r2_score(y_train_scaled, train_pred)
                        test_r2 = r2_score(y_test_scaled, test_pred)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, 
                                                  cv=5, scoring='neg_mean_squared_error')
                        
                        horizon_models[model_name] = {
                            'model': model,
                            'train_mse': train_mse,
                            'test_mse': test_mse,
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'cv_score_mean': -cv_scores.mean(),
                            'cv_score_std': cv_scores.std(),
                            'feature_scaler': feature_scaler,
                            'target_scaler': target_scaler
                        }
                        
                        logger.info(f"{model_name} for {horizon}: Test R2={test_r2:.3f}, Test MSE={test_mse:.3f}")
                        
                    except Exception as e:
                        logger.error(f"Failed to train {model_name} for {horizon}: {str(e)}")
                        continue
                
                # Select best model for this horizon
                if horizon_models:
                    best_model_name = min(horizon_models.keys(), 
                                        key=lambda x: horizon_models[x]['test_mse'])
                    
                    logger.info(f"Best model for {horizon}: {best_model_name}")
                    trained_models[horizon] = horizon_models[best_model_name]
                    trained_models[horizon]['best_model_name'] = best_model_name
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Failed to train models: {str(e)}")
            return {}

    def create_ensemble_model(self, trained_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ensemble model from individual horizon models
        
        Args:
            trained_models: Dictionary of trained models
            
        Returns:
            Ensemble model configuration
        """
        try:
            # For simplicity, use the 24h model as the primary model
            # In production, you might want a more sophisticated ensemble
            
            if '24h' in trained_models:
                primary_model = trained_models['24h']
                
                ensemble_model = {
                    'type': 'ensemble',
                    'primary_model': primary_model['model'],
                    'feature_scaler': primary_model['feature_scaler'],
                    'target_scaler': primary_model['target_scaler'],
                    'feature_columns': self.used_feature_columns,
                    'version': f"ensemble-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                    'training_metadata': {
                        'trained_at': datetime.now(timezone.utc).isoformat(),
                        'horizons_trained': list(trained_models.keys()),
                        'feature_count': len(self.used_feature_columns),
                        'model_performance': {
                            horizon: {
                                'test_r2': models['test_r2'],
                                'test_mse': models['test_mse'],
                                'best_model': models['best_model_name']
                            }
                            for horizon, models in trained_models.items()
                        }
                    }
                }
                
                logger.info("Created ensemble model successfully")
                return ensemble_model
                
            else:
                logger.error("No suitable models found for ensemble creation")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {str(e)}")
            return {}

    def save_model_to_s3(self, model: Dict[str, Any]) -> bool:
        """
        Save trained model to S3
        
        Args:
            model: Trained model dictionary
            
        Returns:
            Success boolean
        """
        try:
            # Serialize model
            model_data = pickle.dumps(model)
            
            # Upload to S3
            key = "models/bitcoin_predictor_model.pkl"
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=model_data,
                ContentType='application/octet-stream'
            )
            
            # Also save metadata separately
            metadata_key = "models/bitcoin_predictor_metadata.json"
            metadata = {
                'version': model['version'],
                'training_metadata': model['training_metadata'],
                'feature_columns': model['feature_columns']
            }
            
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Successfully saved model {model['version']} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model to S3: {str(e)}")
            return False

    def evaluate_model_performance(self, trained_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate overall model performance
        
        Args:
            trained_models: Dictionary of trained models
            
        Returns:
            Performance evaluation results
        """
        try:
            performance_summary = {
                'overall_quality': 'good',
                'horizon_performance': {},
                'recommendations': []
            }
            
            total_r2 = 0
            horizon_count = 0
            
            for horizon, model_data in trained_models.items():
                test_r2 = model_data['test_r2']
                test_mse = model_data['test_mse']
                cv_score = model_data['cv_score_mean']
                
                # Evaluate performance quality
                if test_r2 > 0.7:
                    quality = 'excellent'
                elif test_r2 > 0.5:
                    quality = 'good'
                elif test_r2 > 0.3:
                    quality = 'fair'
                else:
                    quality = 'poor'
                
                performance_summary['horizon_performance'][horizon] = {
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'cv_score': cv_score,
                    'quality': quality,
                    'best_model': model_data['best_model_name']
                }
                
                total_r2 += test_r2
                horizon_count += 1
            
            # Overall quality assessment
            avg_r2 = total_r2 / horizon_count if horizon_count > 0 else 0
            
            if avg_r2 > 0.6:
                performance_summary['overall_quality'] = 'excellent'
            elif avg_r2 > 0.4:
                performance_summary['overall_quality'] = 'good'
            elif avg_r2 > 0.2:
                performance_summary['overall_quality'] = 'fair'
            else:
                performance_summary['overall_quality'] = 'poor'
                performance_summary['recommendations'].append(
                    "Model performance is poor. Consider collecting more training data or feature engineering."
                )
            
            # Add recommendations
            if avg_r2 < 0.5:
                performance_summary['recommendations'].append(
                    "Consider adding more features or using more sophisticated models."
                )
            
            if horizon_count < 3:
                performance_summary['recommendations'].append(
                    "Train models for more time horizons to improve prediction coverage."
                )
            
            performance_summary['average_r2'] = avg_r2
            performance_summary['trained_horizons'] = horizon_count
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {str(e)}")
            return {'overall_quality': 'unknown', 'error': str(e)}

def main():
    """Main training function"""
    try:
        # Configuration (in production, these would come from environment variables or parameters)
        DYNAMODB_TABLE = "bitcoin-predictions"  # Replace with actual table name
        S3_BUCKET = "bitcoin-prediction-data-654654488711"  # Replace with actual bucket name
        
        logger.info("Starting Bitcoin model training")
        
        # Initialize trainer
        trainer = BitcoinModelTrainer(DYNAMODB_TABLE, S3_BUCKET)
        
        # Fetch training data
        training_df = trainer.fetch_training_data(days_back=30)
        
        if training_df.empty:
            logger.error("No training data available")
            return False
        
        # Prepare training data
        X, targets = trainer.prepare_training_data(training_df)
        
        if X.shape[0] == 0:
            logger.error("Failed to prepare training data")
            return False
        
        # Train models
        trained_models = trainer.train_models(X, targets)
        
        if not trained_models:
            logger.error("No models were successfully trained")
            return False
        
        # Create ensemble model
        ensemble_model = trainer.create_ensemble_model(trained_models)
        
        if not ensemble_model:
            logger.error("Failed to create ensemble model")
            return False
        
        # Evaluate performance
        performance = trainer.evaluate_model_performance(trained_models)
        logger.info(f"Model performance: {performance['overall_quality']}")
        logger.info(f"Average R²: {performance.get('average_r2', 0):.3f}")
        
        # Save model to S3
        save_success = trainer.save_model_to_s3(ensemble_model)
        
        if save_success:
            logger.info("Model training completed successfully")
            return True
        else:
            logger.error("Failed to save model")
            return False
            
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)