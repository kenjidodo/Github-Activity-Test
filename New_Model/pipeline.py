# ============================================================================
# FILE: pipeline.py
# ============================================================================
"""
Complete ML pipeline orchestration.
Coordinates data loading, feature engineering, training, and backtesting.
"""

import pandas as pd
from typing import Dict, Tuple
import numpy as np


def run_full_experiment(df_raw: pd.DataFrame,
                       threshold: float = 0.55,
                       cost_bps: float = 1.0,
                       model_type: str = 'xgboost') -> Dict:
    """
    Run complete ML trading experiment.
    
    Pipeline steps:
    1. Feature engineering
    2. Time-series split
    3. Train models (baselines + XGBoost)
    4. Evaluate on test set
    5. Backtest strategy
    6. Extract feature importances
    
    Args:
        df_raw: Raw OHLCV DataFrame
        threshold: Probability threshold for trading
        cost_bps: Transaction cost in basis points
        model_type: 'xgboost', 'always_up', or 'naive_last_direction'
        
    Returns:
        Dictionary with all results
    """
    print("="*60)
    print("STARTING ML TRADING PIPELINE")
    print("="*60)
    
    # Step 1: Feature Engineering
    print("\n[1/6] Building features...")
    df_features = build_features(df_raw)
    
    # Step 2: Time-series split
    print("\n[2/6] Splitting data chronologically...")
    df_train, df_val, df_test = time_series_split(df_features, train_frac=0.6, val_frac=0.2)
    
    # Prepare feature matrices
    exclude_cols = ['y', 'ret_next', 'ret_1', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X_train = df_train[feature_cols]
    y_train = df_train['y']
    X_val = df_val[feature_cols]
    y_val = df_val['y']
    X_test = df_test[feature_cols]
    y_test = df_test['y']
    
    # Step 3: Train model
    print(f"\n[3/6] Training {model_type} model...")
    
    if model_type == 'xgboost':
        model = train_xgb_classifier(X_train, y_train, X_val, y_val)
    elif model_type == 'always_up':
        model = AlwaysUpModel()
        model.fit(X_train, y_train)
    elif model_type == 'naive_last_direction':
        model = NaiveLastDirectionModel()
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Step 4: Evaluate baselines
    print("\n[4/6] Evaluating baseline models...")
    baseline_results = {}
    
    for baseline_name, baseline_model in [('AlwaysUp', AlwaysUpModel()), 
                                           ('NaiveLastDirection', NaiveLastDirectionModel())]:
        baseline_model.fit(X_train, y_train)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
        baseline_results[baseline_name] = baseline_metrics
        print(f"  {baseline_name}: Accuracy={baseline_metrics['accuracy']:.4f}")
    
    # Step 5: Evaluate main model on test set
    print(f"\n[5/6] Evaluating {model_type} on test set...")
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    
    # Step 6: Backtest strategy
    print(f"\n[6/6] Backtesting strategy (threshold={threshold}, cost={cost_bps}bps)...")
    backtest_results = backtest_strategy(df_test, model, threshold, cost_bps)
    
    # Extract feature importances (if XGBoost)
    feature_importance = None
    if model_type == 'xgboost' and hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance = importance_df
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print("="*60)
    print(f"\nStrategy Performance:")
    print(f"  Total Return:    {backtest_results['metrics']['total_return_strat']:.2%}")
    print(f"  Sharpe Ratio:    {backtest_results['metrics']['sharpe_strat']:.2f}")
    print(f"  Max Drawdown:    {backtest_results['metrics']['max_drawdown_strat']:.2%}")
    print(f"  Hit Rate:        {backtest_results['metrics']['hit_rate']:.2%}")
    print(f"  Number of Trades: {backtest_results['metrics']['num_trades']}")
    
    print(f"\nBuy & Hold Comparison:")
    print(f"  Total Return:    {backtest_results['metrics']['total_return_bh']:.2%}")
    print(f"  Sharpe Ratio:    {backtest_results['metrics']['sharpe_bh']:.2f}")
    
    return {
        'model': model,
        'model_type': model_type,
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'feature_cols': feature_cols,
        'baseline_results': baseline_results,
        'test_metrics': test_metrics,
        'backtest_results': backtest_results,
        'feature_importance': feature_importance
    }
