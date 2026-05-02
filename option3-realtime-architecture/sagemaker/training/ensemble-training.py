#!/usr/bin/env python3
"""
Advanced Ensemble Model Training for Real-time Bitcoin Predictions
Combines XGBoost, Random Forest, and Gradient Boosting with automated hyperparameter tuning
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import argparse
import boto3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import optuna

# SageMaker
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

class BitcoinEnsembleTrainer:
    def __init__(self, 
                 s3_bucket: str,
                 s3_prefix: str = 'models/ensemble',
                 region: str = 'eu-west-2'):
        """Initialize the ensemble trainer"""
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        
        # Initialize AWS clients
        self.s3 = boto3.client('s3', region_name=region)
        self.sagemaker_session = sagemaker.Session()
        
        # Model parameters
        self.feature_columns = []
        self.target_columns = ['price_1h', 'price_6h', 'price_24h', 'price_7d']
        self.scalers = {}
        self.feature_selectors = {}
        self.models = {}
        self.ensemble_weights = {}
        
        # Training configuration
        self.cv_folds = 5
        self.test_size = 0.2
        self.random_state = 42
        
    def load_data_from_s3(self, data_path: str) -> pd.DataFrame:
        """Load training data from S3"""
        try:
            print(f"Loading data from s3://{self.s3_bucket}/{data_path}")
            
            # Download data from S3
            local_path = '/tmp/training_data.parquet'
            self.s3.download_file(self.s3_bucket, data_path, local_path)
            
            # Load data
            df = pd.read_parquet(local_path)
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for training"""
        print("Preparing features...")
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Price-based features
        if 'price' in df.columns:
            # Price changes
            df['price_change_1h'] = df['price'].pct_change(1)
            df['price_change_6h'] = df['price'].pct_change(6)
            df['price_change_24h'] = df['price'].pct_change(24)
            
            # Moving averages
            df['sma_12'] = df['price'].rolling(12).mean()
            df['sma_24'] = df['price'].rolling(24).mean()
            df['sma_168'] = df['price'].rolling(168).mean()  # Weekly
            
            # Exponential moving averages
            df['ema_12'] = df['price'].ewm(span=12).mean()
            df['ema_24'] = df['price'].ewm(span=24).mean()
            
            # Bollinger Bands
            rolling_mean = df['price'].rolling(20).mean()
            rolling_std = df['price'].rolling(20).std()
            df['bb_upper'] = rolling_mean + (2 * rolling_std)
            df['bb_lower'] = rolling_mean - (2 * rolling_std)
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI (Relative Strength Index)
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility
            df['volatility_24h'] = df['price'].rolling(24).std()
            df['volatility_168h'] = df['price'].rolling(168).std()
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_change_24h'] = df['volume'].pct_change(24)
            df['volume_sma_24'] = df['volume'].rolling(24).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_24']
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Sentiment features
        if 'sentiment_score' in df.columns:
            df['sentiment_sma_6'] = df['sentiment_score'].rolling(6).mean()
            df['sentiment_sma_24'] = df['sentiment_score'].rolling(24).mean()
            df['sentiment_trend'] = df['sentiment_score'] - df['sentiment_sma_24']
        
        if 'news_count' in df.columns:
            df['news_count_24h'] = df['news_count'].rolling(24).sum()
            df['news_intensity'] = df['news_count'] / df['news_count_24h'].rolling(24).mean()
        
        # Lag features
        for lag in [1, 3, 6, 12, 24]:
            if 'price' in df.columns:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
            if 'sentiment_score' in df.columns:
                df[f'sentiment_lag_{lag}'] = df['sentiment_score'].shift(lag)
        
        # Target variables (future prices)
        if 'price' in df.columns:
            df['price_1h'] = df['price'].shift(-1)
            df['price_6h'] = df['price'].shift(-6)
            df['price_24h'] = df['price'].shift(-24)
            df['price_7d'] = df['price'].shift(-168)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select feature columns (exclude timestamp and target columns)
        exclude_cols = ['timestamp'] + self.target_columns + ['price']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Created {len(self.feature_columns)} features")
        return df
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict:
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state
                }
                model = RandomForestRegressor(**params)
                
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state
                }
                model = GradientBoostingRegressor(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best {model_type} parameters: {study.best_params}")
        return study.best_params
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray, target_col: str) -> Dict:
        """Train base models for ensemble"""
        print(f"Training base models for {target_col}...")
        
        models = {}
        
        # XGBoost
        print("Training XGBoost...")
        xgb_params = self.optimize_hyperparameters(X_train, y_train, 'xgboost')
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_params = self.optimize_hyperparameters(X_train, y_train, 'random_forest')
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb_params = self.optimize_hyperparameters(X_train, y_train, 'gradient_boosting')
        gb_model = GradientBoostingRegressor(**gb_params)
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        # Calculate ensemble weights based on validation performance
        weights = self.calculate_ensemble_weights(models, X_val, y_val)
        
        return {'models': models, 'weights': weights}
    
    def calculate_ensemble_weights(self, models: Dict, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Calculate optimal ensemble weights based on validation performance"""
        print("Calculating ensemble weights...")
        
        # Get predictions from each model
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            pred = model.predict(X_val)
            predictions[name] = pred
            scores[name] = r2_score(y_val, pred)
        
        # Calculate weights based on R² scores
        total_score = sum(max(0, score) for score in scores.values())
        if total_score == 0:
            # Equal weights if all scores are negative
            weights = {name: 1.0/len(models) for name in models}
        else:
            weights = {name: max(0, score)/total_score for name, score in scores.items()}
        
        print(f"Model scores: {scores}")
        print(f"Ensemble weights: {weights}")
        
        return weights
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict:
        """Train complete ensemble model"""
        print("Starting ensemble training...")
        
        # Prepare data
        X = df[self.feature_columns].values
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection
        feature_selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_columns)))
        X_selected = feature_selector.fit_transform(X_scaled, df['price_24h'].values)
        
        # Store preprocessors
        self.scalers['main'] = scaler
        self.feature_selectors['main'] = feature_selector
        
        # Split data (time series split)
        split_idx = int(len(df) * (1 - self.test_size))
        X_train, X_val = X_selected[:split_idx], X_selected[split_idx:]
        
        # Train models for each target
        ensemble_results = {}
        
        for target_col in self.target_columns:
            print(f"\nTraining ensemble for {target_col}")
            
            y = df[target_col].values
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train base models
            result = self.train_base_models(X_train, y_train, X_val, y_val, target_col)
            ensemble_results[target_col] = result
            
            # Evaluate ensemble
            ensemble_pred = self.predict_ensemble(result['models'], result['weights'], X_val)
            
            mse = mean_squared_error(y_val, ensemble_pred)
            mae = mean_absolute_error(y_val, ensemble_pred)
            r2 = r2_score(y_val, ensemble_pred)
            
            print(f"{target_col} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models = ensemble_results
        
        # Calculate feature importance
        self.feature_importance = self.calculate_feature_importance()
        
        return ensemble_results
    
    def predict_ensemble(self, models: Dict, weights: Dict, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        weighted_predictions = np.zeros(X.shape[0])
        
        for name, model in models.items():
            pred = model.predict(X)
            weighted_predictions += weights[name] * pred
        
        return weighted_predictions
    
    def calculate_feature_importance(self) -> Dict:
        """Calculate feature importance across all models"""
        importance_scores = {}
        
        for target_col, result in self.models.items():
            target_importance = {}
            
            for name, model in result['models'].items():
                if hasattr(model, 'feature_importances_'):
                    # Get selected feature names
                    selected_features = self.feature_selectors['main'].get_support()
                    selected_feature_names = [self.feature_columns[i] for i in range(len(selected_features)) if selected_features[i]]
                    
                    # Map importance to feature names
                    for i, importance in enumerate(model.feature_importances_):
                        feature_name = selected_feature_names[i]
                        if feature_name not in target_importance:
                            target_importance[feature_name] = 0
                        target_importance[feature_name] += importance * result['weights'][name]
            
            importance_scores[target_col] = target_importance
        
        return importance_scores
    
    def save_model_artifacts(self) -> str:
        """Save model artifacts to S3"""
        print("Saving model artifacts...")
        
        # Create model directory
        model_dir = '/tmp/model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'feature_importance': self.feature_importance,
            'training_timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': '1.0'
        }
        
        # Save as pickle
        model_path = os.path.join(model_dir, 'ensemble_model.pkl')
        joblib.dump(model_data, model_path)
        
        # Save model metadata
        metadata = {
            'model_type': 'ensemble',
            'framework': 'sklearn_xgboost',
            'feature_count': len(self.feature_columns),
            'target_count': len(self.target_columns),
            'training_timestamp': model_data['training_timestamp'],
            'model_version': model_data['model_version']
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create inference script
        inference_script = '''
import json
import joblib
import numpy as np
from typing import Dict, List

def model_fn(model_dir):
    """Load the model"""
    model_path = f"{model_dir}/ensemble_model.pkl"
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    # Extract components
    models = model['models']
    scalers = model['scalers']
    feature_selectors = model['feature_selectors']
    feature_columns = model['feature_columns']
    target_columns = model['target_columns']
    
    predictions = {}
    
    # Scale features
    X_scaled = scalers['main'].transform(input_data)
    
    # Select features
    X_selected = feature_selectors['main'].transform(X_scaled)
    
    # Make predictions for each target
    for target_col in target_columns:
        target_models = models[target_col]['models']
        target_weights = models[target_col]['weights']
        
        # Ensemble prediction
        weighted_pred = np.zeros(X_selected.shape[0])
        for name, model in target_models.items():
            pred = model.predict(X_selected)
            weighted_pred += target_weights[name] * pred
        
        predictions[target_col] = weighted_pred.tolist()
    
    return predictions

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
        
        inference_path = os.path.join(model_dir, 'inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_script)
        
        # Create requirements.txt
        requirements = '''
numpy==1.21.0
pandas==1.3.0
scikit-learn==1.0.2
xgboost==1.6.0
joblib==1.0.1
'''
        
        requirements_path = os.path.join(model_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        
        # Create tar.gz archive
        import tarfile
        
        model_tar_path = '/tmp/model.tar.gz'
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            tar.add(model_dir, arcname='.')
        
        # Upload to S3
        s3_key = f"{self.s3_prefix}/model.tar.gz"
        self.s3.upload_file(model_tar_path, self.s3_bucket, s3_key)
        
        s3_model_uri = f"s3://{self.s3_bucket}/{s3_key}"
        print(f"Model saved to: {s3_model_uri}")
        
        return s3_model_uri

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Bitcoin Ensemble Model')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--s3-data-path', required=True, help='S3 path to training data')
    parser.add_argument('--s3-prefix', default='models/ensemble', help='S3 prefix for model artifacts')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BitcoinEnsembleTrainer(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region=args.region
    )
    
    try:
        # Load data
        df = trainer.load_data_from_s3(args.s3_data_path)
        
        # Prepare features
        df = trainer.prepare_features(df)
        
        # Train ensemble
        ensemble_results = trainer.train_ensemble(df)
        
        # Save model
        model_uri = trainer.save_model_artifacts()
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_uri}")
        
        # Print feature importance
        print("\nTop 10 Most Important Features:")
        for target_col in trainer.target_columns:
            print(f"\n{target_col}:")
            importance = trainer.feature_importance.get(target_col, {})
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in sorted_features:
                print(f"  {feature}: {score:.4f}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()