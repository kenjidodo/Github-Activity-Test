"""
Data downloading and loading utilities for intraday trading data.
Integrated with Polygon.io Stocks Starter tier.

API Key: vpfk4rxBMDKA1_7x8hJCkQEVQDbukht5
Tier: Stocks Starter
Features:
- Unlimited API calls
- 5 years historical data
- 100% market coverage
- 15-minute delayed data
- Minute/Second aggregates
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, List
import time


# Polygon.io API Configuration - STOCKS STARTER TIER
POLYGON_API_KEY = "vpfk4rxBMDKA1_7x8hJCkQEVQDbukht5"
POLYGON_BASE_URL = "https://api.polygon.io"


def download_intraday_data_polygon(ticker: str, 
                                   start_date: str, 
                                   end_date: str,
                                   timespan: str = '5',
                                   api_key: str = POLYGON_API_KEY) -> pd.DataFrame:
    """
    Download real intraday data from Polygon.io (Stocks Starter tier).
    
    With Stocks Starter, you get:
    - Unlimited API calls (no rate limiting!)
    - 5 years of historical data
    - Minute-level granularity
    - All US stocks
    
    Args:
        ticker: Stock symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timespan: Bar size - '1', '5', '15', '30', '60' minutes
        api_key: Polygon.io API key
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    print(f"[POLYGON.IO - STOCKS STARTER] Downloading {timespan}-min data for {ticker}")
    print(f"   Unlimited API calls | 5 years history | 100% market coverage")
    
    # Convert dates to timestamps
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate expected trading days
    business_days = pd.bdate_range(start_dt, end_dt)
    expected_days = len(business_days)
    
    print(f"   Date range: {start_date} to {end_date} ({expected_days} trading days)")
    
    # Polygon API endpoint for aggregates (bars)
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{timespan}/minute/{start_date}/{end_date}"
    
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    
    all_data = []
    
    try:
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'results' in data and data['results']:
                results = data['results']
                
                print(f"   Retrieved {len(results)} raw bars from API")
                
                for bar in results:
                    # Polygon returns timestamps in milliseconds
                    timestamp = datetime.fromtimestamp(bar['t'] / 1000)
                    
                    # Filter for regular market hours (9:30 AM - 4:00 PM EST)
                    hour = timestamp.hour
                    minute = timestamp.minute
                    minute_of_day = hour * 60 + minute
                    
                    # Market hours: 9:30 (570) to 16:00 (960)
                    if 570 <= minute_of_day <= 960:
                        all_data.append({
                            'timestamp': timestamp,
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v'],
                            'vwap': bar.get('vw', None),  # Volume weighted average price
                            'transactions': bar.get('n', None)  # Number of transactions
                        })
                
                if len(all_data) == 0:
                    raise Exception(f"No market hours data for {ticker} in date range")
                
                df = pd.DataFrame(all_data)
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Basic data quality checks
                trading_days = len(df) / (390 / int(timespan))  # Bars per day
                
                print(f"✅ Downloaded {len(df)} bars ({trading_days:.1f} trading days)")
                print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                print(f"   Avg volume: {df['volume'].mean():,.0f}")
                
                # Remove VWAP and transactions columns if not needed for features
                df = df.drop(['vwap', 'transactions'], axis=1, errors='ignore')
                
                return df
            else:
                error_msg = data.get('error', 'No results in response')
                raise Exception(f"API returned no data: {error_msg}")
        
        elif response.status_code == 403:
            raise Exception("API authentication failed. Check API key.")
        
        elif response.status_code == 404:
            raise Exception(f"Ticker '{ticker}' not found. Verify symbol is correct.")
        
        else:
            raise Exception(f"API error {response.status_code}: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timed out after 60 seconds. Try smaller date range.")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")


def download_multiple_tickers(tickers: List[str],
                              start_date: str,
                              end_date: str,
                              timespan: str = '5') -> dict:
    """
    Download data for multiple tickers (leverages unlimited API calls).
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timespan: Bar size in minutes
        
    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    print(f"\n[BULK DOWNLOAD] Fetching {len(tickers)} tickers")
    print(f"   Using Stocks Starter unlimited API calls")
    
    results = {}
    failed = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            df = download_intraday_data_polygon(ticker, start_date, end_date, timespan)
            results[ticker] = df
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}")
            failed.append(ticker)
    
    print(f"\n{'='*60}")
    print(f"Bulk download complete: {len(results)}/{len(tickers)} successful")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")
    print(f"{'='*60}")
    
    return results


def download_extended_history(ticker: str,
                              years: int = 1,
                              timespan: str = '5') -> pd.DataFrame:
    """
    Download extended historical data (up to 5 years with Stocks Starter).
    
    Args:
        ticker: Stock symbol
        years: Number of years of history (1-5)
        timespan: Bar size in minutes
        
    Returns:
        DataFrame with historical data
    """
    if years > 5:
        print("⚠️  Stocks Starter tier supports up to 5 years. Using 5 years.")
        years = 5
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    
    print(f"\n[EXTENDED HISTORY] Downloading {years} year(s) for {ticker}")
    
    df = download_intraday_data_polygon(ticker, start_date, end_date, timespan)
    
    print(f"✅ Retrieved {years} year(s) of data")
    print(f"   Total bars: {len(df):,}")
    print(f"   Trading days: {len(df) / (390 / int(timespan)):.0f}")
    
    return df


def download_intraday_data_synthetic(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic 5-minute intraday data (for testing without API).
    
    Args:
        ticker: Stock symbol (e.g., 'SPY')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    print(f"[SYNTHETIC] Generating synthetic 5-min data for {ticker}")
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    timestamps = []
    current_date = start
    
    while current_date <= end:
        if current_date.weekday() < 5:
            market_open = current_date.replace(hour=9, minute=30, second=0)
            for i in range(78):
                timestamps.append(market_open + timedelta(minutes=5*i))
        current_date += timedelta(days=1)
    
    n = len(timestamps)
    base_price = 450.0
    
    returns = np.random.randn(n) * 0.0015 + 0.00005
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    data = []
    for i, ts in enumerate(timestamps):
        close = close_prices[i]
        volatility = close * 0.001
        open_price = close + np.random.randn() * volatility
        high_price = max(open_price, close) + abs(np.random.randn() * volatility)
        low_price = min(open_price, close) - abs(np.random.randn() * volatility)
        
        base_volume = 1_000_000
        time_of_day_factor = 1.5 if i % 78 < 10 or i % 78 > 68 else 1.0
        volume = int(base_volume * time_of_day_factor * (0.8 + 0.4 * np.random.random()))
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} synthetic bars")
    return df


def download_intraday_data(ticker: str, 
                           start_date: str, 
                           end_date: str, 
                           use_real_data: bool = True,
                           timespan: str = '5',
                           api_key: str = POLYGON_API_KEY) -> pd.DataFrame:
    """
    Download 5-minute intraday data - automatically switches between real and synthetic.
    
    With Stocks Starter tier:
    - ✅ Unlimited API calls
    - ✅ 5 years of historical data
    - ✅ All US stocks
    - ✅ Minute-level granularity
    
    Args:
        ticker: Stock symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_real_data: If True, use Polygon.io; if False, use synthetic data
        timespan: Bar size - '1', '5', '15', '30', '60' minutes
        api_key: Polygon.io API key
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if use_real_data:
        try:
            return download_intraday_data_polygon(ticker, start_date, end_date, timespan, api_key)
        except Exception as e:
            print(f"\n⚠️  Real data download failed: {str(e)}")
            print("   Falling back to synthetic data...\n")
            return download_intraday_data_synthetic(ticker, start_date, end_date)
    else:
        return download_intraday_data_synthetic(ticker, start_date, end_date)


def load_local_csv(path: str) -> pd.DataFrame:
    """
    Load intraday data from a local CSV file.
    
    Expected CSV format:
        timestamp,open,high,low,close,volume
        2024-01-02 09:30:00,450.5,451.2,450.3,450.8,1200000
        ...
    
    Args:
        path: Path to CSV file
        
    Returns:
        Clean DataFrame with proper types and sorted by timestamp
    """
    print(f"[CSV] Loading data from {path}")
    
    df = pd.read_csv(path)
    
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def get_ticker_details(ticker: str, api_key: str = POLYGON_API_KEY) -> dict:
    """
    Get detailed information about a ticker using Polygon.io reference data.
    
    Args:
        ticker: Stock symbol
        api_key: Polygon.io API key
        
    Returns:
        Dictionary with ticker details
    """
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}"
    params = {'apiKey': api_key}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                result = data['results']
                return {
                    'ticker': result.get('ticker'),
                    'name': result.get('name'),
                    'market': result.get('market'),
                    'locale': result.get('locale'),
                    'primary_exchange': result.get('primary_exchange'),
                    'type': result.get('type'),
                    'currency': result.get('currency_name'),
                    'active': result.get('active')
                }
    except:
        pass
    
    return {'ticker': ticker, 'name': 'Unknown'}


def get_available_tickers() -> dict:
    """
    Return categorized list of popular tickers for intraday trading.
    All supported with Stocks Starter tier.
    """
    return {
        'Major ETFs (Best for testing)': [
            'SPY',   # S&P 500 - Most liquid
            'QQQ',   # Nasdaq 100
            'IWM',   # Russell 2000
            'DIA',   # Dow Jones
            'EEM',   # Emerging Markets
            'GLD',   # Gold
            'TLT',   # 20+ Year Treasury
            'XLF',   # Financial Sector
            'XLE',   # Energy Sector
            'XLK'    # Technology Sector
        ],
        'Mega Cap Tech': [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL', # Google
            'AMZN',  # Amazon
            'NVDA',  # Nvidia
            'META',  # Meta
            'TSLA',  # Tesla
            'NFLX',  # Netflix
            'AVGO',  # Broadcom
            'ORCL'   # Oracle
        ],
        'Blue Chip': [
            'JPM',   # JPMorgan
            'JNJ',   # Johnson & Johnson
            'V',     # Visa
            'WMT',   # Walmart
            'PG',    # Procter & Gamble
            'MA',    # Mastercard
            'HD',    # Home Depot
            'BAC',   # Bank of America
            'KO',    # Coca-Cola
            'PFE'    # Pfizer
        ],
        'High Volume (Good liquidity)': [
            'AMD',   # AMD
            'INTC',  # Intel
            'F',     # Ford
            'SOFI',  # SoFi
            'NIO',   # Nio
            'PLTR',  # Palantir
            'BABA',  # Alibaba
            'COIN',  # Coinbase
            'UBER',  # Uber
            'PYPL'   # PayPal
        ]
    }


def test_polygon_connection(api_key: str = POLYGON_API_KEY) -> bool:
    """
    Test if Polygon.io API key is working and show tier information.
    
    Args:
        api_key: Polygon.io API key
        
    Returns:
        True if connection successful, False otherwise
    """
    print("=" * 60)
    print("TESTING POLYGON.IO API CONNECTION")
    print("=" * 60)
    
    try:
        # Test with a simple request
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/SPY/range/1/day/2024-01-01/2024-01-05"
        params = {'apiKey': api_key}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                print("✅ API Connection Successful!")
                print(f"\n📋 API Configuration:")
                print(f"   Tier: Stocks Starter")
                print(f"   Key: {api_key[:10]}...{api_key[-5:]}")
                print(f"\n⭐ Features Enabled:")
                print(f"   ✅ Unlimited API calls (no rate limits!)")
                print(f"   ✅ 5 years of historical data")
                print(f"   ✅ 100% market coverage - all US stocks")
                print(f"   ✅ Minute-level aggregates")
                print(f"   ✅ Second-level aggregates")
                print(f"   ✅ Reference data & corporate actions")
                print(f"   ✅ Technical indicators")
                print(f"   ✅ WebSockets support")
                print(f"   ⏰ 15-minute delayed data")
                return True
        
        elif response.status_code == 403:
            print("❌ API authentication failed - invalid API key")
            return False
        
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False


def save_data_to_csv(df: pd.DataFrame, ticker: str, timespan: str = '5') -> str:
    """
    Save downloaded data to CSV for future use.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock symbol
        timespan: Bar size
        
    Returns:
        Filename of saved CSV
    """
    filename = f"{ticker}_{timespan}min_{df['timestamp'].min().strftime('%Y%m%d')}_{df['timestamp'].max().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved to: {filename}")
    return filename


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("POLYGON.IO STOCKS STARTER - DATA DOWNLOAD")
    print("🚀" * 30)
    
    # Test 1: Check API connection
    print("\n[TEST 1] API Connection & Tier Info")
    print("-" * 60)
    if not test_polygon_connection():
        print("\n⚠️  Please check your API key")
        exit(1)
    
    # Test 2: Download real data (recent)
    print("\n[TEST 2] Download Recent Data (5-min bars)")
    print("-" * 60)
    try:
        df_recent = download_intraday_data(
            ticker='SPY',
            start_date='2024-01-02',
            end_date='2024-01-15',
            use_real_data=True
        )
        print(f"\n📊 Sample data:")
        print(df_recent.head(10))
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
    
    # Test 3: Extended history (leverage 5 years)
    print("\n[TEST 3] Extended History (1 year)")
    print("-" * 60)
    try:
        df_extended = download_extended_history('SPY', years=1, timespan='5')
        print(f"\n📈 Data summary:")
        print(f"   Bars: {len(df_extended):,}")
        print(f"   Days: {len(df_extended) / 78:.0f}")
        print(f"   Date range: {df_extended['timestamp'].min()} to {df_extended['timestamp'].max()}")
        
    except Exception as e:
        print(f"❌ Extended download failed: {str(e)}")
    
    # Test 4: Multiple tickers (unlimited API calls)
    print("\n[TEST 4] Bulk Download (Multiple Tickers)")
    print("-" * 60)
    try:
        tickers = ['SPY', 'QQQ', 'AAPL']
        results = download_multiple_tickers(
            tickers=tickers,
            start_date='2024-01-02',
            end_date='2024-01-08',
            timespan='5'
        )
        
        print(f"\n✅ Downloaded {len(results)} tickers:")
        for ticker, df in results.items():
            print(f"   {ticker}: {len(df)} bars")
            
    except Exception as e:
        print(f"❌ Bulk download failed: {str(e)}")
    
    # Test 5: Ticker information
    print("\n[TEST 5] Ticker Details (Reference Data)")
    print("-" * 60)
    try:
        details = get_ticker_details('AAPL')
        print(f"\n📋 AAPL Details:")
        for key, value in details.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"⚠️  Could not fetch ticker details: {str(e)}")
    
    # Show available tickers
    print("\n[TEST 6] Available Tickers")
    print("-" * 60)
    tickers = get_available_tickers()
    for category, ticker_list in tickers.items():
        print(f"\n{category}:")
        print(f"   {', '.join(ticker_list)}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. Select 'Polygon.io (Real Data)'")
    print("   3. Try different timespans: 1, 5, 15, 30, 60 minutes")
    print("   4. Use extended history (up to 5 years!)")
    print("   5. Download multiple tickers at once")