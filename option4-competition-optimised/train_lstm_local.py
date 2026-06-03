#!/usr/bin/env python3
"""
Deep Learning Model Training for Real-time Bitcoin Predictions
Advanced LSTM/GRU model with attention mechanism and multi-horizon predictions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import boto3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

class BitcoinLSTMTrainer:
    def __init__(self, 
                 s3_bucket: str,
                 s3_prefix: str = 'models/deeplearning',
                 region: str = 'eu-west-2'):
        """Initialize the LSTM trainer"""
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        
        # Initialize AWS clients
        self.s3 = boto3.client('s3', region_name=region)
        
        # Model parameters
        self.sequence_length = 168  # 1 week of hourly data
        self.feature_columns = []
        self.target_columns = ['price_1h', 'price_6h', 'price_24h', 'price_7d']
        self.scalers = {}
        self.model = None
        
        # Training configuration
        self.test_size = 0.2
        self.validation_size = 0.2
        self.batch_size = 32
        self.epochs = 100
        self.early_stopping_patience = 10
        self.random_state = 42
        
        # Set random seeds
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
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
        """Prepare and engineer features for LSTM training"""
        print("Preparing features for LSTM...")
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Price-based features
        if 'price' in df.columns:
            # Returns and log returns
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            
            # Moving averages
            for window in [6, 12, 24, 48, 168]:
                df[f'sma_{window}'] = df['price'].rolling(window).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
                df[f'price_ratio_sma_{window}'] = df['price'] / df[f'sma_{window}']
            
            # Volatility features
            for window in [6, 12, 24, 48]:
                df[f'volatility_{window}'] = df['returns'].rolling(window).std()
                df[f'volatility_ema_{window}'] = df['returns'].ewm(span=window).std()
            
            # Technical indicators
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            rolling_mean = df['price'].rolling(20).mean()
            rolling_std = df['price'].rolling(20).std()
            df['bb_upper'] = rolling_mean + (2 * rolling_std)
            df['bb_lower'] = rolling_mean - (2 * rolling_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / rolling_mean
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum
            for period in [1, 3, 6, 12, 24]:
                df[f'momentum_{period}'] = df['price'] / df['price'].shift(period)
        
        # Volume-based features
        if 'volume' in df.columns:
            # Volume indicators
            for window in [6, 12, 24, 48]:
                df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            
            # Volume-price trend
            df['vpt'] = (df['volume'] * df['returns']).cumsum()
            df['vpt_sma_24'] = df['vpt'].rolling(24).mean()
            
            # On-Balance Volume
            df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        
        # Sentiment and news features
        if 'sentiment_score' in df.columns:
            # Sentiment moving averages
            for window in [3, 6, 12, 24]:
                df[f'sentiment_sma_{window}'] = df['sentiment_score'].rolling(window).mean()
                df[f'sentiment_ema_{window}'] = df['sentiment_score'].ewm(span=window).mean()
            
            # Sentiment momentum
            df['sentiment_momentum'] = df['sentiment_score'] - df['sentiment_sma_24']
            df['sentiment_volatility'] = df['sentiment_score'].rolling(24).std()
        
        if 'news_count' in df.columns:
            # News intensity features
            for window in [6, 12, 24]:
                df[f'news_count_sum_{window}'] = df['news_count'].rolling(window).sum()
                df[f'news_intensity_{window}'] = df['news_count'] / df[f'news_count_sum_{window}'].rolling(window).mean()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Target variables (future prices)
        if 'price' in df.columns:
            df['price_1h'] = df['price'].shift(-1)
            df['price_6h'] = df['price'].shift(-6)
            df['price_24h'] = df['price'].shift(-24)
            df['price_7d'] = df['price'].shift(-168)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select feature columns
        exclude_cols = ['timestamp'] + self.target_columns + ['price']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Created {len(self.feature_columns)} features for LSTM")
        return df
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple, output_dim: int, trial: Optional[optuna.Trial] = None) -> Model:
        """Build LSTM model with attention mechanism"""
        
        if trial:
            # Hyperparameter optimization
            lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            attention_units = trial.suggest_int('attention_units', 16, 64)
            dense_units = trial.suggest_int('dense_units', 32, 128)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        else:
            # Default hyperparameters
            lstm_units_1 = 128
            lstm_units_2 = 64
            dropout_rate = 0.2
            attention_units = 32
            dense_units = 64
            learning_rate = 0.001
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # First LSTM layer
        lstm_1 = layers.LSTM(lstm_units_1, return_sequences=True, return_state=True)
        lstm_out_1, state_h_1, state_c_1 = lstm_1(inputs)
        
        # Second LSTM layer
        lstm_2 = layers.LSTM(lstm_units_2, return_sequences=True, return_state=True)
        lstm_out_2, state_h_2, state_c_2 = lstm_2(lstm_out_1)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=attention_units,
            dropout=dropout_rate
        )
        attention_out = attention(lstm_out_2, lstm_out_2)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(attention_out)
        
        # Dense layers
        dense_1 = layers.Dense(dense_units, activation='relu')(pooled)
        dropout_1 = layers.Dropout(dropout_rate)(dense_1)
        
        dense_2 = layers.Dense(dense_units // 2, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(dropout_rate / 2)(dense_2)
        
        # Output layers (one for each prediction horizon)
        outputs = []
        for i in range(output_dim):
            output = layers.Dense(1, activation='linear', name=f'output_{i}')(dropout_2)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model — Keras 3 requires per-output metrics for multi-output models
        optimizer = Adam(learning_rate=learning_rate)
        per_output_metrics = {
            f'output_{i}': [MeanAbsoluteError(), RootMeanSquaredError()]
            for i in range(output_dim)
        }
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=per_output_metrics
        )
        
        return model
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna"""
        print("Optimizing LSTM hyperparameters...")
        
        def objective(trial):
            # Build model with trial parameters
            model = self.build_lstm_model(
                input_shape=(self.sequence_length, len(self.feature_columns)),
                output_dim=len(self.target_columns),
                trial=trial
            )
            
            # Train model
            history = model.fit(
                X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                validation_data=(X_val, [y_val[:, i] for i in range(y_val.shape[1])]),
                epochs=20,  # Reduced epochs for hyperparameter optimization
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[
                    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Return validation loss
            val_loss = min(history.history['val_loss'])
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        print(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def train_lstm_model(self, df: pd.DataFrame) -> Model:
        """Train the complete LSTM model"""
        print("Starting LSTM training...")
        
        # Prepare data
        feature_data = df[self.feature_columns].values
        target_data = df[self.target_columns].values
        
        # Scale features
        feature_scaler = MinMaxScaler()
        feature_data_scaled = feature_scaler.fit_transform(feature_data)
        
        # Scale targets
        target_scaler = MinMaxScaler()
        target_data_scaled = target_scaler.fit_transform(target_data)
        
        # Store scalers
        self.scalers['features'] = feature_scaler
        self.scalers['targets'] = target_scaler
        
        # Create sequences
        X, y = self.create_sequences(feature_data_scaled, target_data_scaled)
        
        print(f"Created {len(X)} sequences of length {self.sequence_length}")
        
        # Split data (time series split)
        total_size = len(X)
        test_size = int(total_size * self.test_size)
        val_size = int(total_size * self.validation_size)
        train_size = total_size - test_size - val_size
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Build final model with best hyperparameters
        model = self.build_lstm_model(
            input_shape=(self.sequence_length, len(self.feature_columns)),
            output_dim=len(self.target_columns),
            trial=None  # Use default params, but could integrate best_params here
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            callbacks.ModelCheckpoint(
                '/tmp/best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        print("Training final LSTM model...")
        history = model.fit(
            X_train, [y_train[:, i] for i in range(y_train.shape[1])],
            validation_data=(X_val, [y_val[:, i] for i in range(y_val.shape[1])]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Load best weights
        model.load_weights('/tmp/best_model.h5')
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_predictions = model.predict(X_test)
        
        # Convert predictions back to original scale
        for i, target_col in enumerate(self.target_columns):
            pred_scaled = test_predictions[i].flatten()
            true_scaled = y_test[:, i]
            
            # Create dummy array for inverse transform
            dummy_pred = np.zeros((len(pred_scaled), len(self.target_columns)))
            dummy_true = np.zeros((len(true_scaled), len(self.target_columns)))
            dummy_pred[:, i] = pred_scaled
            dummy_true[:, i] = true_scaled
            
            pred_original = target_scaler.inverse_transform(dummy_pred)[:, i]
            true_original = target_scaler.inverse_transform(dummy_true)[:, i]
            
            mse = mean_squared_error(true_original, pred_original)
            mae = mean_absolute_error(true_original, pred_original)
            r2 = r2_score(true_original, pred_original)
            
            print(f"{target_col} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.model = model
        return model
    
    def save_model_artifacts(self) -> str:
        """Save model artifacts to S3"""
        print("Saving LSTM model artifacts...")
        
        # Create model directory
        model_dir = '/tmp/model'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the Keras model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model.save(model_path)
        
        # Save scalers and metadata
        import pickle
        
        scalers_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save model configuration
        config = {
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'training_timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': '1.0',
            'model_type': 'lstm_attention',
            'framework': 'tensorflow_keras'
        }
        
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create inference script
        inference_script = '''
import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

def model_fn(model_dir):
    """Load the model"""
    # Load Keras model
    model = keras.models.load_model(f"{model_dir}/lstm_model.h5")
    
    # Load scalers
    with open(f"{model_dir}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    
    # Load config
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)
    
    return {
        "model": model,
        "scalers": scalers,
        "config": config
    }

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions"""
    model = model_artifacts['model']
    scalers = model_artifacts['scalers']
    config = model_artifacts['config']
    
    sequence_length = config['sequence_length']
    feature_columns = config['feature_columns']
    target_columns = config['target_columns']
    
    # Scale input features
    feature_scaler = scalers['features']
    target_scaler = scalers['targets']
    
    # Scale features
    input_scaled = feature_scaler.transform(input_data)
    
    # Create sequences
    if len(input_scaled) < sequence_length:
        # Pad with zeros if insufficient data
        padding = np.zeros((sequence_length - len(input_scaled), input_scaled.shape[1]))
        input_scaled = np.vstack([padding, input_scaled])
    
    # Take the last sequence_length points
    sequence = input_scaled[-sequence_length:].reshape(1, sequence_length, -1)
    
    # Make prediction
    predictions = model.predict(sequence)
    
    # Convert predictions back to original scale
    results = {}
    for i, target_col in enumerate(target_columns):
        pred_value = predictions[i][0][0]
        
        # Create dummy array for inverse transform
        dummy = np.zeros((1, len(target_columns)))
        dummy[0, i] = pred_value
        pred_original = target_scaler.inverse_transform(dummy)[0, i]
        
        results[target_col] = float(pred_original)
    
    return results

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
tensorflow==2.12.0
numpy==1.21.0
scikit-learn==1.0.2
pandas==1.3.0
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
        print(f"LSTM model saved to: {s3_model_uri}")
        
        return s3_model_uri

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Bitcoin LSTM Model')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--s3-data-path', required=True, help='S3 path to training data')
    parser.add_argument('--s3-prefix', default='models/deeplearning', help='S3 prefix for model artifacts')
    parser.add_argument('--region', default='eu-west-2', help='AWS region')
    parser.add_argument('--sequence-length', type=int, default=168, help='LSTM sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BitcoinLSTMTrainer(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region=args.region
    )
    
    # Update configuration
    trainer.sequence_length = args.sequence_length
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    
    try:
        # Load data
        df = trainer.load_data_from_s3(args.s3_data_path)
        
        # Prepare features
        df = trainer.prepare_features(df)
        
        # Train model
        model = trainer.train_lstm_model(df)
        
        # Save model
        model_uri = trainer.save_model_artifacts()
        
        print(f"\nLSTM training completed successfully!")
        print(f"Model saved to: {model_uri}")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()