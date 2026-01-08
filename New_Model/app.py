"""
Streamlit app for intraday ML trading system.
Interactive interface for running experiments and visualizing results.

To run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys

# Import our modules
# NOTE: In production, these should be proper imports
# For this example, assume all functions are available
try:
    from data_download import download_intraday_data, load_local_csv
    from features import build_features
    from modeling import time_series_split, train_xgb_classifier, AlwaysUpModel, NaiveLastDirectionModel, evaluate_model
    from backtest import backtest_strategy
    from pipeline import run_full_experiment
except ImportError:
    st.error("Cannot import modules. Make sure all .py files are in the same directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Intraday ML Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: black;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: black;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Intraday ML Trading System</h1>', unsafe_allow_html=True)
st.markdown("**5-Minute Bar Prediction | XGBoost Classification | Realistic Backtesting**")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Polygon.io (Real Data)", "Synthetic Data", "Local CSV"],
    help="Choose your data source: Real market data, synthetic, or CSV file"
)

# Ticker input
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Stock symbol for trading (All US stocks supported)"
).upper()

# Show ticker details if available
if ticker:
    with st.sidebar.expander("ℹ️ Ticker Info"):
        try:
            from data_download import get_ticker_details
            details = get_ticker_details(ticker)
            st.write(f"**{details.get('name', 'N/A')}**")
            st.write(f"Exchange: {details.get('primary_exchange', 'N/A')}")
            st.write(f"Type: {details.get('type', 'N/A')}")
        except:
            st.write("Ticker information not available")

# Date range
st.sidebar.subheader("📅 Data Configuration")

# Preset date ranges for convenience
date_preset = st.sidebar.selectbox(
    "Quick Date Range",
    ["Custom", "Last Week", "Last Month", "Last 3 Months", "Last 6 Months", "Last Year", "Last 2 Years"],
    help="Stocks Starter supports up to 5 years of history"
)

if date_preset == "Last Week":
    start_date_default = datetime.now() - timedelta(days=7)
    end_date_default = datetime.now()
elif date_preset == "Last Month":
    start_date_default = datetime.now() - timedelta(days=30)
    end_date_default = datetime.now()
elif date_preset == "Last 3 Months":
    start_date_default = datetime.now() - timedelta(days=90)
    end_date_default = datetime.now()
elif date_preset == "Last 6 Months":
    start_date_default = datetime.now() - timedelta(days=180)
    end_date_default = datetime.now()
elif date_preset == "Last Year":
    start_date_default = datetime.now() - timedelta(days=365)
    end_date_default = datetime.now()
elif date_preset == "Last 2 Years":
    start_date_default = datetime.now() - timedelta(days=730)
    end_date_default = datetime.now()
else:  # Custom
    start_date_default = datetime.now() - timedelta(days=60)
    end_date_default = datetime.now()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=start_date_default,
        help="Start date for historical data"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=end_date_default,
        help="End date for historical data"
    )

# Bar size selection
timespan = st.sidebar.selectbox(
    "Bar Size",
    ["1 min", "5 min", "15 min", "30 min", "60 min"],
    index=1,
    help="Smaller bars = more data points, but longer training time"
)
timespan_value = timespan.split()[0]  # Extract number

# Model selection
st.sidebar.subheader("🤖 Model Configuration")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["xgboost", "always_up", "naive_last_direction"],
    help="Choose model: XGBoost (ML), AlwaysUp (baseline), or NaiveLastDirection (momentum)"
)

# Trading parameters
st.sidebar.subheader("💰 Trading Parameters")
threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.50,
    max_value=0.70,
    value=0.55,
    step=0.01,
    help="Minimum probability to enter long position"
)

cost_bps = st.sidebar.slider(
    "Trading Cost (bps)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Transaction cost in basis points (1 bp = 0.01%)"
)

# CSV file upload
csv_file = None
if data_source == "Local CSV":
    csv_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=['csv'],
        help="CSV must have columns: timestamp, open, high, low, close, volume"
    )

# Run button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if not run_analysis:
    # Welcome screen
    st.info("👈 Configure your experiment in the sidebar and click **Run Analysis** to begin.")
    
    st.header("📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        Predict the direction of the next 5-minute bar using machine learning.
        
        - **Input:** 5-min OHLCV data
        - **Output:** Binary prediction (up/down)
        - **Strategy:** Long when prob > threshold
        """)
    
    with col2:
        st.subheader("🔧 Features")
        st.write("""
        Professional technical indicators:
        
        - Returns & momentum (10 lags)
        - Volatility (rolling std, ATR)
        - Moving averages (6, 12, 24)
        - Volume analysis
        - Time-of-day effects
        """)
    
    with col3:
        st.subheader("✅ Best Practices")
        st.write("""
        Realistic quant workflow:
        
        - ✓ No lookahead bias
        - ✓ Time-series split
        - ✓ Transaction costs
        - ✓ Baseline comparisons
        - ✓ Walk-forward logic
        """)
    
    # Methodology section
    st.header("📚 Methodology")
    
    with st.expander("🔍 Data Processing"):
        st.write("""
        **5-Minute Intraday Bars**
        - Market hours: 9:30 AM - 4:00 PM EST
        - 78 bars per trading day
        - OHLCV format (Open, High, Low, Close, Volume)
        
        **Quality Checks**
        - Remove NaN values
        - Sort chronologically
        - Verify data integrity
        """)
    
    with st.expander("⚙️ Feature Engineering"):
        st.write("""
        **Categories:**
        
        1. **Returns** (11 features)
           - Current log return
           - 10 lagged returns
           - 3/6/12-bar cumulative returns
        
        2. **Volatility** (5 features)
           - Rolling std (6, 12, 24 bars)
           - ATR (14-bar)
           - Normalized ATR %
        
        3. **Moving Averages** (6 features)
           - SMA 6, 12, 24
           - Distance from SMAs
           - Crossover signals
        
        4. **Volume** (4 features)
           - Rolling mean volume
           - Volume surprise
           - Volume trend
        
        5. **Time** (4 features)
           - Sin/cos encoding
           - First/last hour flags
        
        **Total: ~30 features**
        """)
    
    with st.expander("🤖 Models"):
        st.write("""
        **XGBoost Classifier** (Primary)
        - Gradient boosted trees
        - Early stopping on validation set
        - Hyperparameters:
          - max_depth: 4
          - learning_rate: 0.05
          - n_estimators: 200
          - subsample: 0.8
        
        **Baselines** (For comparison)
        - AlwaysUp: Always predicts market up
        - NaiveLastDirection: Follows last bar
        
        **Why XGBoost?**
        - Handles non-linear patterns
        - No feature scaling needed
        - Built-in regularization
        - Fast training
        - Feature importance
        """)
    
    with st.expander("📊 Backtesting"):
        st.write("""
        **Strategy Logic**
        1. At each 5-min bar, predict next bar direction
        2. If P(up) > threshold: Go long (position = 1)
        3. Otherwise: Stay flat (position = 0)
        4. Apply transaction costs on position changes
        
        **Metrics Calculated**
        - Total return (strategy vs buy-and-hold)
        - Sharpe ratio (annualized)
        - Maximum drawdown
        - Hit rate (% profitable trades)
        - Number of trades
        - Time in market
        
        **Realistic Assumptions**
        - Transaction costs: 1 basis point per trade
        - No slippage modeling (can be added)
        - Intraday only (no overnight positions)
        - Signal generated at bar close, executed immediately
        """)
    
    with st.expander("⚠️ Important Notes"):
        st.write("""
        **No Lookahead Bias**
        - Features use only past data
        - Target is shifted to prevent leakage
        - Predictions use t-1 features for t+1 returns
        
        **Time-Series Split**
        - Train: 60% (oldest data)
        - Validation: 20% (middle)
        - Test: 20% (most recent)
        - Never shuffle time-series data!
        
        **Limitations**
        - Past performance ≠ future results
        - Markets change over time
        - Real trading has additional complexities
        - This is for educational purposes only
        """)

else:
    # Run the analysis
    try:
        # Load data
        with st.spinner(f"📥 Loading {timespan} data for {ticker}..."):
            if data_source == "Polygon.io (Real Data)":
                df_raw = download_intraday_data(ticker, str(start_date), str(end_date), use_real_data=True, timespan=timespan_value)
            elif data_source == "Synthetic Data":
                df_raw = download_intraday_data(ticker, str(start_date), str(end_date), use_real_data=False)
            else:  # Local CSV
                if csv_file is None:
                    st.error("Please upload a CSV file.")
                    st.stop()
                df_raw = pd.read_csv(csv_file)
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        
        st.success(f"✅ Loaded {len(df_raw)} bars from {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        
        # Run pipeline
        with st.spinner("🔄 Running ML pipeline..."):
            results = run_full_experiment(
                df_raw,
                threshold=threshold,
                cost_bps=cost_bps,
                model_type=model_type
            )
        
        # ===== SECTION 1: DATA OVERVIEW =====
        st.header("📊 1. Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bars", len(df_raw))
        with col2:
            trading_days = len(df_raw) / 78
            st.metric("Trading Days", f"{trading_days:.1f}")
        with col3:
            st.metric("Start Date", df_raw['timestamp'].min().strftime("%Y-%m-%d"))
        with col4:
            st.metric("End Date", df_raw['timestamp'].max().strftime("%Y-%m-%d"))
        
        # Price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_raw['timestamp'],
            y=df_raw['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=1)
        ))
        fig_price.update_layout(
            title=f"{ticker} Intraday Price (5-min bars)",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # ===== SECTION 2: MODEL PERFORMANCE =====
        st.header("🤖 2. Model Performance")
        
        test_metrics = results['test_metrics']
        baseline_results = results['baseline_results']
        
        # Metrics comparison
        st.subheader("Classification Metrics (Test Set)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{test_metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{test_metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{test_metrics['f1']:.2%}")
        
        # Baseline comparison
        st.subheader("Baseline Comparison")
        comparison_df = pd.DataFrame({
            'Model': [model_type.upper()] + list(baseline_results.keys()),
            'Accuracy': [test_metrics['accuracy']] + [baseline_results[k]['accuracy'] for k in baseline_results],
            'Precision': [test_metrics['precision']] + [baseline_results[k]['precision'] for k in baseline_results],
            'Recall': [test_metrics['recall']] + [baseline_results[k]['recall'] for k in baseline_results]
        })
        
        fig_baseline = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall']:
            fig_baseline.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=[f"{v:.1%}" for v in comparison_df[metric]],
                textposition='auto'
            ))
        
        fig_baseline.update_layout(
            title="Model vs Baselines",
            barmode='group',
            yaxis_title="Score",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_baseline, use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        df_test = results['df_test']
        X_test = df_test[results['feature_cols']]
        y_test = df_test['y']
        y_pred = results['model'].predict(X_test)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Down', 'Predicted Up'],
            y=['Actual Down', 'Actual Up'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ===== SECTION 3: BACKTEST RESULTS =====
        st.header("💰 3. Backtest Results")
        
        backtest_results = results['backtest_results']
        metrics = backtest_results['metrics']
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Strategy Return",
                f"{metrics['total_return_strat']:.2%}",
                delta=f"{(metrics['total_return_strat'] - metrics['total_return_bh']):.2%} vs B&H"
            )
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_strat']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown_strat']:.2%}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")
        with col2:
            st.metric("Number of Trades", f"{metrics['num_trades']}")
        with col3:
            st.metric("Time in Market", f"{metrics['time_in_market']:.1%}")
        
        # Equity curve
        st.subheader("Equity Curves")
        df_backtest = backtest_results['df']
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df_backtest.index,
            y=df_backtest['strat_equity'],
            mode='lines',
            name='Strategy',
            line=dict(color='#2ca02c', width=2)
        ))
        fig_equity.add_trace(go.Scatter(
            x=df_backtest.index,
            y=df_backtest['bh_equity'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#1f77b4', width=2, dash='dash')
        ))
        
        fig_equity.update_layout(
            title=f"Strategy vs Buy & Hold (Threshold={threshold}, Cost={cost_bps}bps)",
            xaxis_title="Time",
            yaxis_title="Equity (Starting = 1.0)",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        running_max_strat = df_backtest['strat_equity'].expanding().max()
        drawdown_strat = (df_backtest['strat_equity'] - running_max_strat) / running_max_strat
        
        running_max_bh = df_backtest['bh_equity'].expanding().max()
        drawdown_bh = (df_backtest['bh_equity'] - running_max_bh) / running_max_bh
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_backtest.index,
            y=drawdown_strat,
            mode='lines',
            name='Strategy DD',
            fill='tozeroy',
            line=dict(color='#d62728', width=1)
        ))
        fig_dd.add_trace(go.Scatter(
            x=df_backtest.index,
            y=drawdown_bh,
            mode='lines',
            name='B&H DD',
            line=dict(color='#ff7f0e', width=1, dash='dash')
        ))
        
        fig_dd.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Performance interpretation
        if metrics['sharpe_strat'] > 1.5:
            st.markdown('<div class="success-box">✅ <strong>Strong Performance:</strong> Sharpe ratio > 1.5 indicates good risk-adjusted returns.</div>', unsafe_allow_html=True)
        elif metrics['sharpe_strat'] > 0.5:
            st.markdown('<div class="info-box">ℹ️ <strong>Moderate Performance:</strong> Sharpe ratio between 0.5-1.5 shows potential but room for improvement.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <strong>Weak Performance:</strong> Sharpe ratio < 0.5 suggests limited edge over buy-and-hold.</div>', unsafe_allow_html=True)
        
        # ===== SECTION 4: FEATURE IMPORTANCE =====
        if results['feature_importance'] is not None:
            st.header("📊 4. Feature Importance")
            
            fi_df = results['feature_importance'].head(20)
            
            fig_fi = go.Figure(go.Bar(
                x=fi_df['importance'],
                y=fi_df['feature'],
                orientation='h',
                marker_color='#1f77b4'
            ))
            
            fig_fi.update_layout(
                title="Top 20 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600,
                template='plotly_white',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.write("**Interpretation:** Higher importance = greater impact on predictions")
        
        # ===== SECTION 5: TRADE LOG =====
        with st.expander("📋 View Trade Log (First 100)"):
            trades_df = backtest_results['trades'].head(100)
            if len(trades_df) > 0:
                display_cols = ['prob_up', 'signal', 'position', 'strat_ret']
                st.dataframe(trades_df[display_cols], use_container_width=True)
            else:
                st.info("No trades executed in backtest period.")
        
        # ===== SECTION 6: DOWNLOAD RESULTS =====
        st.header("💾 5. Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download backtest results
            csv_backtest = df_backtest.to_csv()
            st.download_button(
                label="📥 Download Backtest Results",
                data=csv_backtest,
                file_name=f"{ticker}_backtest_{model_type}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download feature importance
            if results['feature_importance'] is not None:
                csv_fi = results['feature_importance'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Feature Importance",
                    data=csv_fi,
                    file_name=f"{ticker}_feature_importance.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer:** This system is for educational and research purposes only. 
Past performance does not guarantee future results. Intraday trading involves substantial risk. 
Always conduct thorough research and consider consulting with financial professionals before trading.
""")

st.markdown("**Built with ❤️ using Streamlit, XGBoost, and Python**")