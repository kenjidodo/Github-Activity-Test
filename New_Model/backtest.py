# ============================================================================
# FILE: backtest.py
# ============================================================================
"""
Backtesting engine for intraday trading strategies.
Realistic P&L calculation with transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Dict


def backtest_strategy(df_features: pd.DataFrame,
                     model,
                     threshold: float = 0.55,
                     trading_cost_bps: float = 1.0) -> Dict:
    """
    Backtest a long/flat intraday strategy.
    
    Strategy Logic:
    - At each bar, predict probability of next bar going up
    - If prob > threshold: go long (position = 1)
    - Otherwise: stay flat (position = 0)
    - Apply transaction costs when position changes
    
    Args:
        df_features: DataFrame with features and 'ret_next' column
        model: Trained model with predict_proba method
        threshold: Probability threshold for going long (0.5-1.0)
        trading_cost_bps: Transaction cost in basis points (1 bp = 0.01%)
        
    Returns:
        Dictionary with equity curves, metrics, and trade log
    """
    df = df_features.copy()
    
    # Get feature columns (exclude target, returns, and OHLCV)
    exclude_cols = ['y', 'ret_next', 'ret_1', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # Get predictions
    print("\nGenerating predictions for backtest...")
    proba = model.predict_proba(X)
    df['prob_up'] = proba[:, 1]  # Probability of up move
    
    # Generate trading signals
    df['signal'] = (df['prob_up'] > threshold).astype(int)
    
    # Calculate position (lag signal by 1 to avoid lookahead)
    df['position'] = df['signal'].shift(1).fillna(0)
    
    # Calculate transaction costs
    df['position_change'] = df['position'].diff().abs()
    df['transaction_cost'] = df['position_change'] * (trading_cost_bps / 10000)
    
    # Strategy returns (position * next return - transaction costs)
    df['strat_ret'] = df['position'] * df['ret_next'] - df['transaction_cost']
    
    # Buy and hold returns (always long)
    df['bh_ret'] = df['ret_next']
    
    # Calculate equity curves (cumulative returns)
    df['strat_equity'] = (1 + df['strat_ret']).cumprod()
    df['bh_equity'] = (1 + df['bh_ret']).cumprod()
    
    # Calculate metrics
    metrics = calculate_performance_metrics(df)
    
    # Trade log
    trades = df[df['position_change'] > 0].copy()
    
    print(f"\nBacktest complete:")
    print(f"  Total bars: {len(df)}")
    print(f"  Number of trades: {int(df['position_change'].sum())}")
    print(f"  Time in market: {df['position'].mean():.1%}")
    
    return {
        'df': df,
        'metrics': metrics,
        'trades': trades,
        'equity_curves': df[['strat_equity', 'bh_equity']].copy()
    }


def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        df: DataFrame with strat_ret, bh_ret, position columns
        
    Returns:
        Dictionary of performance metrics
    """
    strat_ret = df['strat_ret'].dropna()
    bh_ret = df['bh_ret'].dropna()
    
    # Bars per day (assuming 5-min bars, ~78 per day)
    bars_per_day = 78
    trading_days = len(strat_ret) / bars_per_day
    
    # Total returns
    total_return_strat = df['strat_equity'].iloc[-1] - 1
    total_return_bh = df['bh_equity'].iloc[-1] - 1
    
    # Annualized returns (252 trading days)
    annual_return_strat = (1 + total_return_strat) ** (252 / trading_days) - 1
    annual_return_bh = (1 + total_return_bh) ** (252 / trading_days) - 1
    
    # Sharpe ratio (annualized)
    # sqrt(252 * bars_per_day) to annualize
    sharpe_factor = np.sqrt(252 * bars_per_day)
    sharpe_strat = strat_ret.mean() / strat_ret.std() * sharpe_factor if strat_ret.std() > 0 else 0
    sharpe_bh = bh_ret.mean() / bh_ret.std() * sharpe_factor if bh_ret.std() > 0 else 0
    
    # Maximum drawdown
    def max_drawdown(equity_curve):
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    max_dd_strat = max_drawdown(df['strat_equity'])
    max_dd_bh = max_drawdown(df['bh_equity'])
    
    # Win rate (for trades where position > 0)
    profitable_bars = df[df['position'] > 0]['strat_ret'] > 0
    hit_rate = profitable_bars.mean() if len(profitable_bars) > 0 else 0
    
    # Number of trades (position changes)
    num_trades = int(df['position'].diff().abs().sum())
    
    # Prediction accuracy
    y_true = df['y'].values
    y_pred = df['signal'].values
    accuracy = (y_true == y_pred).mean()
    
    metrics = {
        'total_return_strat': total_return_strat,
        'total_return_bh': total_return_bh,
        'annual_return_strat': annual_return_strat,
        'annual_return_bh': annual_return_bh,
        'sharpe_strat': sharpe_strat,
        'sharpe_bh': sharpe_bh,
        'max_drawdown_strat': max_dd_strat,
        'max_drawdown_bh': max_dd_bh,
        'hit_rate': hit_rate,
        'num_trades': num_trades,
        'prediction_accuracy': accuracy,
        'time_in_market': df['position'].mean()
    }
    
    return metrics
