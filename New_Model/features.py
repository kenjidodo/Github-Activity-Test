

# ============================================================================
# FILE: features.py
# ============================================================================
"""
Feature engineering for intraday trading.
Implements technical indicators and creates target labels.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct engineered intraday features and target label.
    
    This function creates a comprehensive set of technical features:
    - Returns (current and lagged)
    - Volatility measures
    - Moving averages
    - Volume indicators
    - Time-of-day features
    - Binary target label (next bar direction)
    
    Args:
        df: DataFrame with OHLCV data and timestamp index
        
    Returns:
        DataFrame with feature columns and target 'y'
        Rows with NaN are dropped (from rolling calculations)
    """
    df = df.copy()
    df = df.set_index('timestamp').sort_index()
    
    # ========== RETURNS ==========
    # Log returns (more stable than simple returns)
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    
    # Lagged returns (momentum features)
    for lag in range(1, 11):
        df[f'ret_lag_{lag}'] = df['ret_1'].shift(lag)
    
    # Cumulative returns over different windows
    df['ret_sum_3'] = df['ret_1'].rolling(3).sum()
    df['ret_sum_6'] = df['ret_1'].rolling(6).sum()
    df['ret_sum_12'] = df['ret_1'].rolling(12).sum()
    
    # ========== VOLATILITY ==========
    # Rolling standard deviation of returns
    df['vol_6'] = df['ret_1'].rolling(6).std()
    df['vol_12'] = df['ret_1'].rolling(12).std()
    df['vol_24'] = df['ret_1'].rolling(24).std()
    
    # Average True Range (ATR) - measures volatility using high-low range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    
    # Normalized ATR (as % of price)
    df['atr_pct'] = df['atr_14'] / df['close']
    
    # ========== MOVING AVERAGES ==========
    df['sma_6'] = df['close'].rolling(6).mean()
    df['sma_12'] = df['close'].rolling(12).mean()
    df['sma_24'] = df['close'].rolling(24).mean()
    
    # Distance from moving averages (mean reversion signal)
    df['dist_from_sma_6'] = (df['close'] - df['sma_6']) / df['sma_6']
    df['dist_from_sma_24'] = (df['close'] - df['sma_24']) / df['sma_24']
    
    # Moving average crossovers
    df['sma_6_12_cross'] = (df['sma_6'] - df['sma_12']) / df['sma_12']
    
    # ========== VOLUME ==========
    # Rolling mean volume
    df['vol_mean_12'] = df['volume'].rolling(12).mean()
    df['vol_mean_24'] = df['volume'].rolling(24).mean()
    
    # Volume surprise (relative volume)
    df['vol_surprise'] = (df['volume'] - df['vol_mean_24']) / (df['vol_mean_24'] + 1)
    
    # Volume trend
    df['vol_trend'] = df['volume'].rolling(12).mean() / df['volume'].rolling(24).mean()
    
    # ========== TIME OF DAY ==========
    # Extract time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    # Cyclical encoding (sin/cos for periodicity)
    # Market is open 390 minutes (9:30 - 16:00)
    df['time_sin'] = np.sin(2 * np.pi * (df['minute_of_day'] - 570) / 390)
    df['time_cos'] = np.cos(2 * np.pi * (df['minute_of_day'] - 570) / 390)
    
    # Is it first/last hour? (typically higher volatility)
    df['is_first_hour'] = (df['minute_of_day'] < 630).astype(int)
    df['is_last_hour'] = (df['minute_of_day'] > 900).astype(int)
    
    # ========== TARGET LABEL ==========
    # Next bar return (what we're trying to predict)
    df['ret_next'] = df['ret_1'].shift(-1)
    
    # Binary classification: will next bar be positive?
    df['y'] = (df['ret_next'] > 0).astype(int)
    
    # ========== CLEANUP ==========
    # Drop intermediate calculation columns
    df = df.drop(['high_low', 'high_close', 'low_close', 'true_range', 'hour', 'minute'], axis=1)
    
    # Drop rows with NaN (from rolling calculations and shift)
    df = df.dropna()
    
    print(f"Features built: {len(df)} rows, {len(df.columns)} columns")
    print(f"Target distribution: {df['y'].value_counts(normalize=True).to_dict()}")
    
    return df




