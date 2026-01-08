# ============================================================================
# FILE: modeling.py
# ============================================================================
"""
Model training and baseline implementations.
Includes time-series splitting and XGBoost training.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def time_series_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for time-series modeling.
    
    CRITICAL: Never shuffle time-series data!
    
    Args:
        df: Feature DataFrame sorted by time
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        
    Returns:
        (df_train, df_val, df_test) tuple
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\nTime-series split:")
    print(f"  Train: {len(df_train)} rows ({df_train.index.min()} to {df_train.index.max()})")
    print(f"  Val:   {len(df_val)} rows ({df_val.index.min()} to {df_val.index.max()})")
    print(f"  Test:  {len(df_test)} rows ({df_test.index.min()} to {df_test.index.max()})")
    
    return df_train, df_val, df_test


class AlwaysUpModel:
    """
    Baseline model that always predicts the market will go up.
    Useful for comparing against market's natural upward bias.
    """
    
    def __init__(self):
        self.name = "AlwaysUp"
    
    def fit(self, X, y):
        """No training needed for this baseline."""
        return self
    
    def predict(self, X):
        """Always predict 1 (up)."""
        return np.ones(len(X), dtype=int)
    
    def predict_proba(self, X):
        """Return probability of [down, up] = [0, 1]."""
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


class NaiveLastDirectionModel:
    """
    Baseline model that predicts the same direction as the last bar.
    Tests if simple momentum has predictive power.
    """
    
    def __init__(self):
        self.name = "NaiveLastDirection"
    
    def fit(self, X, y):
        """No training needed for this baseline."""
        return self
    
    def predict(self, X):
        """
        Predict same direction as last bar.
        Assumes X contains 'ret_1' feature.
        """
        if isinstance(X, pd.DataFrame):
            last_ret = X['ret_1'].values
        else:
            # If numpy array, assume first column is ret_1
            last_ret = X[:, 0]
        
        return (last_ret > 0).astype(int)
    
    def predict_proba(self, X):
        """Return probability based on last direction."""
        predictions = self.predict(X)
        n = len(predictions)
        proba = np.zeros((n, 2))
        proba[predictions == 0, 0] = 1.0
        proba[predictions == 1, 1] = 1.0
        return proba


def train_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         params: dict = None) -> xgb.XGBClassifier:
    """
    Train XGBoost binary classifier with early stopping.
    
    XGBoost is ideal for financial data:
    - Handles non-linear relationships
    - Built-in regularization
    - Feature importance
    - Fast training
    - No need for feature scaling
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional hyperparameter dict
        
    Returns:
        Fitted XGBoost model
    """
    if params is None:
        params = {
            'max_depth': 4,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist'  # Faster training
        }
    
    print("\nTraining XGBoost classifier...")
    
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_prec = precision_score(y_val, val_pred, zero_division=0)
    val_rec = recall_score(y_val, val_pred, zero_division=0)
    
    print(f"Validation metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics


