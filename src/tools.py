from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
import requests
import time
import os

# API Keys setup
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
FMP_API_KEY = os.environ.get("FMP_API_KEY")

# Validate API keys
if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not found in environment variables")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY not found in environment variables")

# Optional: Add rate limiting setup
RATE_LIMIT_PAUSE = 0.12  # 120ms pause between API calls for rate limiting


def get_market_cap_peers(ticker: str, min_pct: float = 0.2, max_pct: float = 5.0) -> dict:
    """
    Get peers based on market cap range from related companies.
    Returns similarly sized companies.
    """
    # Get related companies first
    related = get_related_tickers(ticker)
    if not related or 'results' not in related:
        return {'error': 'No related companies found'}
    
    # Filter by market cap
    related_tickers = [company['ticker'] for company in related['results']]
    size_filtered = filter_by_market_cap_range(related_tickers, ticker, min_pct, max_pct)
    
    return {
        'base_company': get_ticker_details(ticker),
        'peer_count': len(size_filtered),
        'peers': size_filtered,
        'market_cap_range': {
            'min_pct': min_pct,
            'max_pct': max_pct
        }
    }

def get_prices(ticker, start_date, end_date):
    """Fetch price data from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
    return prices

def prices_to_df(prices):
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

# Update the get_price_data function to use the new functions
def get_price_data(ticker, start_date, end_date):
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

def get_financial_metrics(ticker, report_period, period='ttm', limit=1):
    """Fetch financial metrics from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError("No financial metrics returned")
    return financial_metrics

def get_insider_trades(ticker, start_date, end_date):
    """
    Fetch insider trades for a given ticker and date range.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_gte={start_date}"
        f"&filing_date_lte={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    insider_trades = data.get("insider_trades")
    if not insider_trades:
        raise ValueError("No insider trades returned")
    return insider_trades

def calculate_confidence_level(signals):
    """Calculate confidence level based on the difference between SMAs."""
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df, window=20):
    """Calculate Bollinger Bands"""
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_obv(prices_df):
    """Calculate On-Balance Volume (OBV)"""
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']



# --------------    new additions  ------
# We are looking to make some new tools / agents that solve these pormpts: 

# 3 prompts: 1) Publicly trading stocks that their market cap is trading below their revenue

# 2) Of those stocks, what sector are they in and what is that multiple

# 3) Based on similar stocks, the 75 day moving avg. and 200 day moving avg., are these stocks trading above or below their actual price target?

# 4) What do the top ratings indicate their price point to me? Hold, buy, sell

# 5) Are there any events or news that contributed to trading volatility?

#general terms we also want to have information on currently

# "multiple" is how much it trades vs. revenue. Saas Company sector = 10x forward multiple.

# P/E for records


def calculate_moving_averages(ticker: str, end_date: str = None) -> dict:
    """
    Calculates key moving averages and their relationships to current price.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        end_date (str, optional): End date for calculation in 'YYYY-MM-DD' format
    
    Returns:
        dict: Moving average data containing:
            - 'ma_75': 75-day moving average
            - 'ma_200': 200-day moving average
            - 'price_to_75ma': Ratio of current price to 75MA
            - 'price_to_200ma': Ratio of current price to 200MA
            - 'trend_signals': Dict of trend indicators
            - 'support_resistance': Dict of support/resistance levels
    """
    # Calculate start date to get enough data for 200MA
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=300)).strftime('%Y-%m-%d')
    
    # Get price data
    df = get_price_data(ticker, start_date, end_date)
    
    # Calculate moving averages
    df['MA75'] = df['close'].rolling(window=75).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
    # Get the most recent values
    current_price = df['close'].iloc[-1]
    ma_75 = df['MA75'].iloc[-1]
    ma_200 = df['MA200'].iloc[-1]
    
    # Calculate price to MA ratios
    price_to_75ma = current_price / ma_75 if ma_75 else None
    price_to_200ma = current_price / ma_200 if ma_200 else None
    
    # Determine trend signals
    trend_signals = {
        'above_75ma': current_price > ma_75 if ma_75 else None,
        'above_200ma': current_price > ma_200 if ma_200 else None,
        'golden_cross': df['MA75'].iloc[-1] > df['MA200'].iloc[-1] if ma_75 and ma_200 else None,
        'death_cross': df['MA75'].iloc[-1] < df['MA200'].iloc[-1] if ma_75 and ma_200 else None,
        'trend_strength': 'strong_uptrend' if price_to_75ma > 1.05 and price_to_200ma > 1.05 else
                         'strong_downtrend' if price_to_75ma < 0.95 and price_to_200ma < 0.95 else
                         'neutral'
    }
    
    # Calculate support and resistance levels using recent price action
    recent_df = df.tail(20)  # Last 20 days
    support_level = recent_df['low'].min()
    resistance_level = recent_df['high'].max()
    
    support_resistance = {
        'support': support_level,
        'resistance': resistance_level,
        'distance_to_support': ((current_price - support_level) / current_price) * 100,
        'distance_to_resistance': ((resistance_level - current_price) / current_price) * 100
    }
    
    return {
        'ma_75': ma_75,
        'ma_200': ma_200,
        'current_price': current_price,
        'price_to_75ma': price_to_75ma,
        'price_to_200ma': price_to_200ma,
        'trend_signals': trend_signals,
        'support_resistance': support_resistance,
        'last_updated': end_date
    }


def analyze_volatility_events(ticker: str, start_date: str = None, end_date: str = None) -> dict:
    """
    Analyzes periods of high volatility and their characteristics.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
    
    Returns:
        dict: Volatility analysis containing:
            - 'volatility_periods': List of high volatility periods
            - 'summary_metrics': Dict of volatility metrics
            - 'trading_volume_analysis': Volume analysis during volatile periods
    """
    # Set up dates if not provided
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = start_date or (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Get price data
    df = get_price_data(ticker, start_date, end_date)
    
    # Calculate daily returns and volatility
    df['daily_returns'] = df['close'].pct_change()
    df['volatility'] = df['daily_returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate volume metrics
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Identify high volatility periods (>2 standard deviations from mean)
    volatility_mean = df['volatility'].mean()
    volatility_std = df['volatility'].std()
    high_volatility_threshold = volatility_mean + 2 * volatility_std
    
    # Find periods of high volatility
    high_vol_days = df[df['volatility'] > high_volatility_threshold]
    
    volatility_periods = []
    if not high_vol_days.empty:
        # Group consecutive days of high volatility
        high_vol_days['date_diff'] = high_vol_days.index.to_series().diff().dt.days
        period_start = high_vol_days.index[0]
        
        for i in range(1, len(high_vol_days)):
            if high_vol_days['date_diff'].iloc[i] > 1:
                # End of a period
                period_end = high_vol_days.index[i-1]
                period_data = df[period_start:period_end]
                
                volatility_periods.append({
                    'start_date': period_start.strftime('%Y-%m-%d'),
                    'end_date': period_end.strftime('%Y-%m-%d'),
                    'max_volatility': period_data['volatility'].max(),
                    'avg_volume_ratio': period_data['volume_ratio'].mean(),
                    'price_change': ((period_data['close'].iloc[-1] / period_data['close'].iloc[0]) - 1) * 100
                })
                period_start = high_vol_days.index[i]
        
        # Add the last period
        period_end = high_vol_days.index[-1]
        period_data = df[period_start:period_end]
        volatility_periods.append({
            'start_date': period_start.strftime('%Y-%m-%d'),
            'end_date': period_end.strftime('%Y-%m-%d'),
            'max_volatility': period_data['volatility'].max(),
            'avg_volume_ratio': period_data['volume_ratio'].mean(),
            'price_change': ((period_data['close'].iloc[-1] / period_data['close'].iloc[0]) - 1) * 100
        })
    
    # Calculate summary metrics
    summary_metrics = {
        'avg_daily_volatility': df['volatility'].mean(),
        'max_volatility': df['volatility'].max(),
        'current_volatility': df['volatility'].iloc[-1],
        'volatility_trend': 'increasing' if df['volatility'].iloc[-1] > df['volatility'].iloc[-20:].mean()
                           else 'decreasing',
        'high_volatility_days': len(high_vol_days)
    }
    
    # Analyze trading volume during volatile periods
    volume_analysis = {
        'avg_daily_volume': df['volume'].mean(),
        'avg_volume_volatile_periods': high_vol_days['volume'].mean() if not high_vol_days.empty else 0,
        'volume_trend': 'increasing' if df['volume'].iloc[-1] > df['volume'].iloc[-20:].mean()
                       else 'decreasing',
        'highest_volume_date': df['volume'].idxmax().strftime('%Y-%m-%d'),
        'highest_volume': df['volume'].max()
    }
    
    return {
        'volatility_periods': volatility_periods,
        'summary_metrics': summary_metrics,
        'trading_volume_analysis': volume_analysis
    }


def get_revenue_data(ticker: str, period: str = 'annual') -> dict:
    """
    Retrieves revenue data for a given stock ticker using Polygon.io API.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        period (str): Time period for revenue data. Options:
                     'annual': Annual revenue
                     'quarterly': Quarterly revenue
                     'ttm': Trailing twelve months
    
    Returns:
        dict: Revenue data containing:
            - 'revenue': Revenue value in USD
            - 'period_end': End date of the period
            - 'growth': Year-over-year growth rate
            - 'revenue_per_share': Revenue per share
            - 'historical_data': List of historical revenue data points
    """
    headers = {"Authorization": f"Bearer {os.environ.get('POLYGON_API_KEY')}"}
    
    # First get company details for shares outstanding
    details_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={os.environ.get('POLYGON_API_KEY')}"
    details_response = requests.get(details_url)
    
    if details_response.status_code != 200:
        raise Exception(f"Error fetching company details: {details_response.status_code} - {details_response.text}")
    
    company_details = details_response.json().get('results', {})
    shares_outstanding = company_details.get('weighted_shares_outstanding', 0)
    
    # Get financial statements using the correct endpoint
    timespan = 'quarter' if period == 'quarterly' else 'annual'
    financials_url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe={timespan}&limit=8&apiKey={os.environ.get('POLYGON_API_KEY')}"
    financials_response = requests.get(financials_url)
    
    if financials_response.status_code != 200:
        raise Exception(f"Error fetching financials: {financials_response.status_code} - {financials_response.text}")
    
    financials_data = financials_response.json()
    results = financials_data.get('results', [])
    
    if not results:
        raise ValueError(f"No financial data found for {ticker}")
    
    # Sort results by date to get most recent first
    results.sort(key=lambda x: x.get('end_date', ''), reverse=True)
    
    # Get the most recent data point
    current = results[0]
    
    # Find previous year's data for growth calculation
    previous_year = None
    current_end_date = datetime.strptime(current.get('end_date', ''), '%Y-%m-%d')
    
    for result in results[1:]:
        result_date = datetime.strptime(result.get('end_date', ''), '%Y-%m-%d')
        date_diff = (current_end_date - result_date).days
        
        # For quarterly, look back 4 quarters; for annual, look back 1 year
        if (period == 'quarterly' and 350 <= date_diff <= 380) or \
           (period == 'annual' and 350 <= date_diff <= 380):
            previous_year = result
            break
    
    # Calculate growth rate if previous year data exists
    growth_rate = None
    if previous_year:
        current_revenue = current.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value', 0)
        prev_revenue = previous_year.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value', 0)
        if prev_revenue and prev_revenue != 0:
            growth_rate = (current_revenue - prev_revenue) / prev_revenue
    
    # Get current revenue
    current_revenue = current.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value')
    
    # Calculate revenue per share
    revenue_per_share = current_revenue / shares_outstanding if current_revenue and shares_outstanding else None
    
    # Prepare historical data
    historical = []
    for result in results:
        revenue = result.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value')
        if revenue:
            historical.append({
                'period_end': result.get('end_date'),
                'revenue': revenue,
                'fiscal_period': result.get('fiscal_period'),
                'fiscal_year': result.get('fiscal_year')
            })
    
    return {
        'revenue': current_revenue,
        'period_end': current.get('end_date'),
        'growth': growth_rate,
        'revenue_per_share': revenue_per_share,
        'historical_data': historical,
        'currency': current.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('unit', 'USD'),
        'fiscal_period': current.get('fiscal_period'),
    }

def get_analyst_ratings(ticker: str) -> Dict:
    """
    Get current analyst ratings and price targets using FMP API.
    
    Returns:
        Dict containing:
        - consensus_rating
        - price_targets (high, low, mean)
        - rating_breakdown
        - analyst_count
        - recent_changes
    """
    # Get analyst recommendations
    recommendations = make_fmp_request(f"/v3/analyst-stock-recommendations/{ticker}")
    if not recommendations or not isinstance(recommendations, list):
        return {"error": "Failed to get analyst recommendations"}
        
    # Get price targets - using a different endpoint that might work better
    price_targets = make_fmp_request(f"/v3/analyst-price-target/{ticker}")  # Changed endpoint
    if not price_targets or isinstance(price_targets, dict) and 'error' in price_targets:
        return {"error": "Failed to get price targets"}
        
    # Get consensus data - using a different endpoint
    consensus = make_fmp_request(f"/v3/analyst-estimates/{ticker}")  # Changed endpoint
    if not consensus or isinstance(consensus, dict) and 'error' in consensus:
        return {"error": "Failed to get consensus data"}
    
    try:
        # Calculate rating breakdown from most recent month
        latest = recommendations[0]
        rating_breakdown = {
            'buy': latest.get('analystRatingsbuy', 0) + latest.get('analystRatingsStrongBuy', 0),
            'hold': latest.get('analystRatingsHold', 0),
            'sell': latest.get('analystRatingsSell', 0) + latest.get('analystRatingsStrongSell', 0)
        }
        total_ratings = sum(rating_breakdown.values())
        
        # Calculate consensus based on ratings
        buy_ratio = rating_breakdown['buy'] / total_ratings if total_ratings > 0 else 0
        if buy_ratio >= 0.7:
            consensus_rating = "Strong Buy"
        elif buy_ratio >= 0.5:
            consensus_rating = "Buy"
        elif buy_ratio >= 0.3:
            consensus_rating = "Hold"
        else:
            consensus_rating = "Sell"
        
        # Get recent rating changes
        upgrades_downgrades = make_fmp_request(f"/v3/upgrades-downgrades/{ticker}")
        recent_changes = []
        if upgrades_downgrades and isinstance(upgrades_downgrades, list):
            recent_changes = [{
                'date': change.get('date'),
                'company': change.get('company'),
                'change': change.get('action'),
                'to_grade': change.get('toGrade'),
                'from_grade': change.get('fromGrade')
            } for change in upgrades_downgrades[:5]]
        
        return {
            'consensus_rating': consensus_rating,
            'price_targets': {
                'high': max([r.get('targetPrice', 0) for r in price_targets]) if isinstance(price_targets, list) else None,
                'low': min([r.get('targetPrice', 0) for r in price_targets]) if isinstance(price_targets, list) else None,
                'mean': sum([r.get('targetPrice', 0) for r in price_targets]) / len(price_targets) if isinstance(price_targets, list) and price_targets else None,
                'median': sorted([r.get('targetPrice', 0) for r in price_targets])[len(price_targets)//2] if isinstance(price_targets, list) and price_targets else None
            },
            'rating_breakdown': rating_breakdown,
            'analyst_count': total_ratings,
            'recent_changes': recent_changes
        }
        
    except Exception as e:
        return {
            "error": f"Error processing analyst data: {str(e)}",
            "details": {
                "recommendations_available": bool(recommendations),
                "price_targets_available": bool(price_targets),
                "consensus_available": bool(consensus)
            }
        }

def calculate_revenue_multiple(ticker: str) -> dict:
    """
    Calculates the revenue multiple (Market Cap / Revenue) for a given stock.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Revenue multiple data containing:
            - 'multiple': Current revenue multiple
            - 'ttm_multiple': Trailing twelve months multiple
            - 'forward_multiple': Forward multiple (if available)
            - 'sector_average': Average multiple for the sector
            - 'peer_comparison': List of peer multiples
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass


def filter_by_market_cap(companies: list, target_cap: float, range_percent: float = 0.5) -> list:
    """
    Filter companies by market cap within a specified range of a target.
    
    Args:
        companies (list): List of company dictionaries with market_cap field
        target_cap (float): Target market cap to compare against
        range_percent (float): Percentage range above and below target (0.5 = ±50%)
    
    Returns:
        list: Filtered list of companies within market cap range
    """
    min_cap = target_cap * (1 - range_percent)
    max_cap = target_cap * (1 + range_percent)
    
    return [
        company for company in companies
        if company.get('market_cap') and min_cap <= company['market_cap'] <= max_cap
    ]

def filter_by_sic_code(tickers: list, base_ticker: str, tolerance: float = 0.0) -> list:
    """
    Filter companies by matching SIC code with optional tolerance for related industries.
    
    Args:
        tickers (list): List of ticker symbols to filter
        base_ticker (str): Base company ticker to compare against
        tolerance (float): How closely SIC codes must match (0.0 = exact match, 0.1 = allow similar industry)
    
    Returns:
        list: Filtered list of companies with matching industry classification
    """
    # Get base company details
    base_company = get_ticker_details(base_ticker)
    if not base_company.get('sic_code'):
        return []
    
    base_sic = base_company['sic_code']
    filtered_companies = []
    
    for ticker in tickers:
        try:
            company = get_ticker_details(ticker)
            if company and company.get('sic_code'):
                # Exact match
                if tolerance == 0.0 and company['sic_code'] == base_sic:
                    filtered_companies.append(company)
                # Similar industry (first 2-3 digits match)
                elif tolerance > 0.0 and company['sic_code'][:3] == base_sic[:3]:
                    filtered_companies.append(company)
            time.sleep(0.12)  # Rate limiting
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    return filtered_companies

def filter_by_market_cap_range(companies: list, base_ticker: str, 
                             lower_pct: float = 0.2, upper_pct: float = 5.0) -> list:
    """
    Filter companies by market cap within a reasonable range of the base company.
    
    Args:
        companies (list): List of company dictionaries with market_cap field
        base_ticker (str): Ticker symbol of the base company
        lower_pct (float): Lower bound as percentage of base market cap (0.2 = 20%)
        upper_pct (float): Upper bound as percentage of base market cap (5.0 = 500%)
    
    Returns:
        list: Filtered list of companies within market cap range
    """
    base_company = get_ticker_details(base_ticker)
    if not base_company.get('market_cap'):
        return []
    
    base_cap = float(base_company['market_cap'])
    min_cap = base_cap * lower_pct
    max_cap = base_cap * upper_pct
    
    return [
        company for company in companies
        if company.get('market_cap') and min_cap <= float(company['market_cap']) <= max_cap
    ]

def get_comparable_peers(ticker: str, sic_tolerance: float = 0.0, 
                        min_market_cap_pct: float = 0.2, 
                        max_market_cap_pct: float = 5.0) -> dict:
    """
    Get comparable peer companies using multiple filters.
    
    Args:
        ticker (str): Base company ticker symbol
        sic_tolerance (float): Tolerance for SIC code matching
        min_market_cap_pct (float): Minimum market cap as % of base company
        max_market_cap_pct (float): Maximum market cap as % of base company
    
    Returns:
        dict: Filtered peer companies with analysis
    """
    # Get related companies from Polygon API
    related = get_related_tickers(ticker)
    if not related or 'results' not in related:
        return {'error': 'No related companies found'}
    
    # Extract tickers from results
    related_tickers = [company['ticker'] for company in related['results']]
    
    # Apply filters
    sic_filtered = filter_by_sic_code(related_tickers, ticker, sic_tolerance)
    final_peers = filter_by_market_cap_range(
        sic_filtered, 
        ticker, 
        min_market_cap_pct, 
        max_market_cap_pct
    )
    
    # Sort by market cap
    final_peers.sort(key=lambda x: float(x.get('market_cap', 0)), reverse=True)
    
    # Get base company details
    base_company = get_ticker_details(ticker)
    
    return {
        'base_company': base_company,
        'peer_count': len(final_peers),
        'peers': final_peers,
        'filters_applied': {
            'sic_tolerance': sic_tolerance,
            'market_cap_range': {
                'min_pct': min_market_cap_pct,
                'max_pct': max_market_cap_pct
            }
        }
    }

def analyze_sector_metrics(ticker: str) -> dict:
    """Analyzes sector-specific metrics and comparisons using FMP API."""
    try:
        # Get company profile
        print(f"\nDEBUG - Getting profile for {ticker}")
        profile = make_fmp_request(f"/v3/profile/{ticker}")
        if not profile or not isinstance(profile, list):
            return {"error": "Failed to get company profile"}
        
        company_details = profile[0]
        sector = company_details.get('sector')
        industry = company_details.get('industry')
        print(f"DEBUG - Sector: {sector}, Industry: {industry}")
        
        # Get company metrics for P/S ratio
        print("\nDEBUG - Getting company metrics")
        company_metrics = make_fmp_request(f"/v3/key-metrics-ttm/{ticker}")
        if not company_metrics or not isinstance(company_metrics, list):
            return {"error": "Failed to get company metrics"}
            
        # Get quote data for current market data
        print("\nDEBUG - Getting quote data")
        quote = make_fmp_request(f"/v3/quote/{ticker}")
        if not quote or not isinstance(quote, list):
            return {"error": "Failed to get quote data"}
            
        # Get company data from quote
        company_market_cap = quote[0].get('marketCap', 0)
        company_shares = quote[0].get('sharesOutstanding', 0)
        company_multiple = company_metrics[0].get('priceToSalesRatioTTM')
        
        print(f"\nDEBUG - Company metrics:")
        print(f"Market Cap: ${company_market_cap:,.0f}")
        print(f"Shares Outstanding: {company_shares:,.0f}")
        print(f"Price to Sales Ratio: {company_multiple:.2f}x" if company_multiple else "P/S Ratio: None")
        print(f"P/E Ratio: {quote[0].get('pe', 'N/A')}")
        
        # Get peer companies
        peers_data = make_fmp_request(f"/v4/stock_peers?symbol={ticker}")
        if not peers_data or not isinstance(peers_data, list):
            return {"error": "Failed to get peer companies"}
            
        peer_list = peers_data[0].get('peersList', []) if peers_data else []
        print(f"\nDEBUG - Found {len(peer_list)} peers: {peer_list}")
        
        if not peer_list:
            return {"error": "No peers found for company"}
        
        # Get company metrics
        print("\nDEBUG - Getting company metrics")
        company_metrics = make_fmp_request(f"/v3/key-metrics-ttm/{ticker}")
        print(f"DEBUG - Company metrics response: {company_metrics[:1]}")  # Show first item
        
        if not company_metrics or not isinstance(company_metrics, list):
            return {"error": "Failed to get company metrics"}
        
        # Get quote data
        print("\nDEBUG - Getting quote data")
        quote = make_fmp_request(f"/v3/quote/{ticker}")
        print(f"DEBUG - Quote response: {quote}")
        
        if not quote or not isinstance(quote, list):
            return {"error": "Failed to get quote data"}
            
        # Calculate company multiple
        company_market_cap = quote[0].get('marketCap', 0)
        shares_outstanding = company_metrics[0].get('sharesOutstanding', 0)
        revenue_per_share = company_metrics[0].get('revenuePerShare', 0)
        company_revenue = revenue_per_share * shares_outstanding
        
        print(f"\nDEBUG - Company calculations:")
        print(f"Market Cap: ${company_market_cap:,.0f}")
        print(f"Shares Outstanding: {shares_outstanding:,.0f}")
        print(f"Revenue Per Share: ${revenue_per_share:.2f}")
        print(f"Calculated Revenue: ${company_revenue:,.0f}")
        
        company_multiple = company_market_cap / company_revenue if company_revenue else None
        print(f"Company Multiple: {company_multiple:.2f}x" if company_multiple else "Company Multiple: None")
        
        # Get peer metrics
        peer_multiples = []
        peer_metrics = []
        
        print("\nDEBUG - Processing peers:")
        for peer_symbol in peer_list:
            try:
                print(f"\nProcessing peer: {peer_symbol}")
                peer_metrics_data = make_fmp_request(f"/v3/key-metrics-ttm/{peer_symbol}")
                peer_quote = make_fmp_request(f"/v3/quote/{peer_symbol}")
                
                if not peer_metrics_data or not peer_quote:
                    print(f"Missing data for {peer_symbol}")
                    continue
                
                peer_multiple = peer_metrics_data[0].get('priceToSalesRatioTTM')
                peer_market_cap = peer_quote[0].get('marketCap', 0)
                
                if peer_multiple:
                    print(f"Market Cap: ${peer_market_cap:,.0f}")
                    print(f"Price to Sales Ratio: {peer_multiple:.2f}x")
                    peer_multiples.append(peer_multiple)
                    peer_metrics.append({
                        'symbol': peer_symbol,
                        'multiple': peer_multiple,
                        'market_cap': peer_market_cap,
                        'net_margin': peer_metrics_data[0].get('netIncomePerShareTTM'),
                        'roe': peer_metrics_data[0].get('roeTTM')
                    })
                    
            except Exception as e:
                print(f"Error processing peer {peer_symbol}: {str(e)}")
                continue
                
            time.sleep(0.12)
        
        print(f"\nDEBUG - Collected {len(peer_multiples)} valid peer multiples")
        
        # Calculate sector averages
        if peer_multiples:
            avg_multiple = sum(peer_multiples) / len(peer_multiples)
            median_multiple = sorted(peer_multiples)[len(peer_multiples)//2]
            print(f"Average Multiple: {avg_multiple:.2f}x")
            print(f"Median Multiple: {median_multiple:.2f}x")
        else:
            avg_multiple = None
            median_multiple = None
            print("No valid peer multiples calculated")
        
        return {
            'sector_name': sector,
            'industry': industry,
            'sector_metrics': {
                'avg_revenue_multiple': avg_multiple,
                'median_revenue_multiple': median_multiple,
                'company_multiple': company_multiple,
                'percentile': ((sum(1 for x in peer_multiples if x < company_multiple) / len(peer_multiples))
                             if peer_multiples and company_multiple else None)
            },
            'peer_metrics': peer_metrics,
            'total_peers_analyzed': len(peer_metrics),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"DEBUG - Sector analysis error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "error": f"Sector analysis failed: {str(e)}",
            "sector_name": sector if 'sector' in locals() else 'Unknown',
            "industry": industry if 'industry' in locals() else 'Unknown'
        }

def analyze_price_targets(ticker: str) -> Dict:
    """
    Get price targets for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        Dict containing price target data
    """
    # Use the correct v4 endpoint
    price_targets = make_fmp_request(f"/v4/price-target-consensus?symbol={ticker}")
    print(f"DEBUG - Price targets response type: {type(price_targets)}")
    print(f"DEBUG - Price targets content: {price_targets}")
    
    # Validate the response
    if not price_targets or not isinstance(price_targets, list) or len(price_targets) == 0:
        print(f"DEBUG - API Key being used: {os.environ.get('FMP_API_KEY', 'Not found')[:5]}...")
        print(f"DEBUG - Full URL: /v4/price-target-consensus?symbol={ticker}")
        return {"error": "No price target data available"}
    
    # Return the raw price target data
    return price_targets[0]

def get_news_events(ticker: str, start_date: str = None, end_date: str = None) -> List[Dict]:
    """Get news events and basic market impact data"""
    news = make_fmp_request(f"/v3/stock-news?tickers={ticker}&limit=50")
    sentiment_data = make_fmp_request(f"/v4/stock-news-sentiments?symbol={ticker}")
    
    analyzed_events = []
    for article in news:
        publish_date = article.get('publishedDate', '').split(' ')[0]
        
        # Get simple price data around the event
        price_data = make_fmp_request(f"/v3/historical-price-full/{ticker}?from={publish_date}&to={publish_date}")
        if price_data and 'historical' in price_data:
            price_change = price_data['historical'][0].get('changePercent')
            volume = price_data['historical'][0].get('volume')
        else:
            price_change = None
            volume = None
            
        analyzed_events.append({
            'date': publish_date,
            'headline': article.get('title'),
            'source': article.get('site'),
            'sentiment_score': sentiment_data.get('sentiment'),
            'price_change': price_change,
            'volume': volume,
            'url': article.get('url')
        })
    
    return analyzed_events

def get_sec_filings(ticker: str, filing_type: str = None, limit: int = 5) -> list:
    """
    Retrieves SEC filings for a given company.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        filing_type (str, optional): Specific filing type (e.g., '10-K', '10-Q', '8-K')
                                   If None, returns all types
        limit (int): Maximum number of filings to return (default: 5)
    
    Returns:
        list: List of filings
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financials/filings"
        f"?ticker={ticker}"
        f"&limit={limit}"
    )
    
    if filing_type:
        url += f"&type={filing_type}"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching SEC filings: {response.status_code} - {response.text}")
    
    return response.json().get('filings', [])

def get_financial_statements(ticker: str, 
                           statement_type: str = 'all', 
                           period: str = 'quarterly', 
                           limit: int = 4) -> dict:
    """
    Retrieves financial statements from SEC filings.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        statement_type (str): Type of statement to retrieve:
                            'all' - All statements
                            'income' - Income Statement
                            'balance' - Balance Sheet
                            'cash_flow' - Cash Flow Statement
        period (str): 'quarterly', 'annual', or 'ttm'
        limit (int): Number of periods to retrieve (default: 4)
    
    Returns:
        dict: Financial statements containing:
            - 'income_statements': List of income statements if requested
            - 'balance_sheets': List of balance sheets if requested
            - 'cash_flow_statements': List of cash flow statements if requested
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    result = {}
    
    # Helper function to fetch specific statement
    def fetch_statement(endpoint):
        url = (
            f"https://api.financialdatasets.ai/financials/{endpoint}"
            f"?ticker={ticker}"
            f"&period={period}"
            f"&limit={limit}"
        )
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching {endpoint}: {response.status_code} - {response.text}")
        return response.json()
    
    try:
        # Fetch requested statements
        if statement_type in ['all', 'income']:
            income_data = fetch_statement('income-statements')
            result['income_statements'] = income_data.get('income_statements', [])
        
        if statement_type in ['all', 'balance']:
            balance_data = fetch_statement('balance-sheets')
            result['balance_sheets'] = balance_data.get('balance_sheets', [])
        
        if statement_type in ['all', 'cash_flow']:
            cash_flow_data = fetch_statement('cash-flow-statements')
            result['cash_flow_statements'] = cash_flow_data.get('cash_flow_statements', [])
        
        return result
    
    except Exception as e:
        print(f"Error fetching financial statements: {str(e)}")
        return result

def analyze_financial_metrics(ticker: str, period: str = 'quarterly') -> dict:
    """
    Analyzes key financial metrics and ratios from statements.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        period (str): 'quarterly', 'annual', or 'ttm'
    
    Returns:
        dict: Financial analysis containing:
            - 'profitability': Profitability ratios
            - 'liquidity': Liquidity ratios
            - 'solvency': Solvency ratios
            - 'growth': Growth metrics
    """
    # Get all financial statements
    statements = get_financial_statements(ticker, 'all', period, 2)  # Get last 2 periods
    
    # Get the most recent statements
    income_stmt = statements.get('income_statements', [{}])[0]
    balance_sheet = statements.get('balance_sheets', [{}])[0]
    cash_flow = statements.get('cash_flow_statements', [{}])[0]
    
    # Calculate profitability ratios
    revenue = income_stmt.get('revenue', 0)
    net_income = income_stmt.get('net_income', 0)
    total_assets = balance_sheet.get('total_assets', 0)
    total_equity = balance_sheet.get('total_equity', 0)
    
    profitability = {
        'gross_margin': income_stmt.get('gross_profit', 0) / revenue if revenue else None,
        'operating_margin': income_stmt.get('operating_income', 0) / revenue if revenue else None,
        'net_margin': net_income / revenue if revenue else None,
        'roe': net_income / total_equity if total_equity else None,
        'roa': net_income / total_assets if total_assets else None
    }
    
    # Calculate liquidity ratios
    current_assets = balance_sheet.get('current_assets', 0)
    current_liabilities = balance_sheet.get('current_liabilities', 0)
    inventory = balance_sheet.get('inventory', 0)
    
    liquidity = {
        'current_ratio': current_assets / current_liabilities if current_liabilities else None,
        'quick_ratio': (current_assets - inventory) / current_liabilities if current_liabilities else None,
        'cash_ratio': balance_sheet.get('cash_and_equivalents', 0) / current_liabilities if current_liabilities else None
    }
    
    # Calculate solvency ratios
    total_liabilities = balance_sheet.get('total_liabilities', 0)
    
    solvency = {
        'debt_to_equity': total_liabilities / total_equity if total_equity else None,
        'debt_to_assets': total_liabilities / total_assets if total_assets else None,
        'interest_coverage': (income_stmt.get('operating_income', 0) / 
                            income_stmt.get('interest_expense', 1) if income_stmt.get('interest_expense') else None)
    }
    
    return {
        'profitability': profitability,
        'liquidity': liquidity,
        'solvency': solvency,
        'period_end': income_stmt.get('report_period'),
        'currency': income_stmt.get('currency')
    }

def get_ticker_details(ticker: str) -> dict:
    """
    Get detailed information about a ticker using Polygon.io API.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Ticker details containing company information
    """
    headers = {"Authorization": f"Bearer {os.environ.get('POLYGON_API_KEY')}"}
    
    # Get ticker details
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={os.environ.get('POLYGON_API_KEY')}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching ticker details: {response.status_code} - {response.text}")
    
    data = response.json()
    results = data.get('results', {})
    
    # Extract SIC code and description
    sic_code = results.get('sic_code')
    sic_description = results.get('sic_description', '')
    
    # Determine sector and industry from SIC code
    # This is a simple mapping - you might want to expand this
    sector_mapping = {
        '10': 'Metal Mining',
        '20': 'Food Products',
        '35': 'Technology',
        '36': 'Electronics',
        '37': 'Transportation Equipment',
        '48': 'Communications',
        '60': 'Financial',
        '73': 'Business Services',
        '80': 'Healthcare',
        '99': 'Other'
    }
    
    # Get first two digits of SIC code for sector
    sector_code = sic_code[:2] if sic_code else None
    sector = sector_mapping.get(sector_code, sic_description.split()[0] if sic_description else None)
    
    # Get market cap from ticker snapshot
    snapshot_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={os.environ.get('POLYGON_API_KEY')}"
    snapshot_response = requests.get(snapshot_url)
    market_cap = None
    
    if snapshot_response.status_code == 200:
        snapshot_data = snapshot_response.json()
        if 'ticker' in snapshot_data:
            market_cap = snapshot_data['ticker'].get('market_cap')
        else:
            # Try getting market cap from daily values
            daily = snapshot_data.get('day', {})
            if daily:
                last_price = daily.get('c', 0)  # Closing price
                shares = results.get('weighted_shares_outstanding', 0)
                if last_price and shares:
                    market_cap = last_price * shares
    
    # If still no market cap, calculate from shares and price
    if market_cap is None and results.get('weighted_shares_outstanding'):
        # Get current price from another endpoint
        aggs_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?apiKey={os.environ.get('POLYGON_API_KEY')}"
        aggs_response = requests.get(aggs_url)
        if aggs_response.status_code == 200:
            aggs_data = aggs_response.json()
            if aggs_data.get('results'):
                last_price = aggs_data['results'][0].get('c', 0)  # Closing price
                shares = results.get('weighted_shares_outstanding', 0)
                market_cap = last_price * shares
    
    return {
        'name': results.get('name'),
        'ticker': results.get('ticker'),
        'market_cap': market_cap,
        'weighted_shares_outstanding': results.get('weighted_shares_outstanding'),
        'sic_code': sic_code,
        'sic_description': sic_description,
        'sector': sector,
        'industry': sic_description,  # Using full SIC description as industry
        'description': results.get('description'),
        'homepage_url': results.get('homepage_url'),
        'list_date': results.get('list_date'),
        'locale': results.get('locale'),
        'market': results.get('market'),
        'currency_name': results.get('currency_name', 'USD'),
        'primary_exchange': results.get('primary_exchange'),
        'type': results.get('type')
    }

def get_related_tickers(ticker: str) -> list:
    """
    Get related tickers using Polygon.io's related companies endpoint.
    
    Args:
        ticker (str): The stock ticker symbol
    
    Returns:
        list: List of related ticker symbols
    """
    headers = {"Authorization": f"Bearer {os.environ.get('POLYGON_API_KEY')}"}
    url = f"https://api.polygon.io/v1/related-companies/{ticker}"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting related tickers: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error in get_related_tickers: {str(e)}")
        return []

def search_tickers(query: str, market: str = 'stocks', limit: int = 100) -> list:
    """
    Search for tickers using Polygon.io's search endpoint.
    
    Args:
        query (str): Search query (company name, ticker, etc.)
        market (str): Market type (stocks, crypto, fx)
        limit (int): Maximum number of results
        
    Returns:
        list: List of matching tickers
    """
    headers = {"Authorization": f"Bearer {os.environ.get('POLYGON_API_KEY')}"}
    url = (
        f"https://api.polygon.io/v3/reference/tickers?"
        f"search={query}"
        f"&market={market}&active=true&limit={limit}"
    )
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"Error searching tickers: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error in search_tickers: {str(e)}")
        return []

#####################################
# Revenue Analysis Tools
#####################################

def calculate_sector_average_multiple(sector: str) -> float:
    """
    Calculate the average revenue multiple for a given sector.
    
    Considerations:
    - Filter outliers
    - Weight by company size
    - Consider subsector differences
    - Account for growth rates
    """
    pass

def get_historical_revenue_growth(ticker: str) -> float:
    """
    Calculate historical revenue growth rates and patterns.
    
    Returns:
    - YoY growth rate
    - 3-year CAGR
    - Growth stability metrics
    - Seasonal patterns
    """
    pass

def analyze_revenue_quality(ticker: str) -> Dict:
    """
    Analyze the quality and sustainability of revenue.
    
    Metrics:
    - Recurring vs one-time revenue
    - Customer concentration
    - Contract length/stability
    - Gross margin trends
    """
    pass

#####################################
# Enhanced Sector Analysis
#####################################

def get_peer_group_metrics(ticker: str) -> Dict:
    """
    Get detailed peer group comparison metrics.
    
    Includes:
    - Size-based peers
    - Growth-based peers
    - Margin-based peers
    - Business model similarity
    """
    pass

def calculate_sector_benchmarks(sector: str) -> Dict:
    """
    Calculate key sector-specific benchmarks and metrics.
    
    Metrics:
    - Typical multiples range
    - Growth expectations
    - Margin standards
    - R&D investments
    """
    pass

def analyze_competitive_position(ticker: str) -> Dict:
    """
    Analyze company's competitive position in sector.
    
    Factors:
    - Market share trends
    - Product differentiation
    - Cost position
    - Entry barriers
    """
    pass

#####################################
# Market Sentiment Enhancement
#####################################

def analyze_news_source_credibility(source: str) -> float:
    """
    Calculate credibility score for news sources.
    
    Factors:
    - Historical accuracy
    - Market impact
    - Timeliness
    - Industry focus
    """
    pass

def calculate_analyst_historical_accuracy(analyst_id: str) -> float:
    """
    Track and score analyst prediction accuracy.
    
    Metrics:
    - Price target accuracy
    - Rating change timing
    - Industry specialization
    - Consistency
    """
    pass

def analyze_institutional_holdings_changes(ticker: str) -> Dict:
    """
    Analyze changes in institutional ownership.
    
    Tracks:
    - Ownership concentration
    - Smart money movements
    - Insider transactions
    - Institution type distribution
    """
    pass

#####################################
# Risk Management Tools
#####################################

def calculate_position_size_recommendation(confidence: float, portfolio: Dict) -> float:
    """
    Calculate recommended position size based on confidence and risk.
    
    Factors:
    - Signal confidence
    - Portfolio correlation
    - Volatility regime
    - Current sector exposure
    """
    pass

def analyze_portfolio_correlation(ticker: str, portfolio: Dict) -> float:
    """
    Analyze correlation between stock and existing portfolio.
    
    Includes:
    - Return correlation
    - Sector correlation
    - Risk factor exposure
    - Diversification impact
    """
    pass

def calculate_sector_exposure_limits(sector: str) -> Dict:
    """
    Calculate maximum sector exposure limits.
    
    Considers:
    - Sector volatility
    - Market conditions
    - Portfolio size
    - Risk tolerance
    """
    pass

#####################################
# Data Quality Tools
#####################################

def validate_financial_data(data: Dict) -> bool:
    """
    Validate financial data quality and consistency.
    
    Checks:
    - Data completeness
    - Logical consistency
    - Outlier detection
    - Format validation
    """
    pass

def check_data_freshness(data: Dict) -> bool:
    """
    Check if financial data is current enough for analysis.
    
    Validates:
    - Last update timestamp
    - Trading day relevance
    - Corporate action impacts
    - Source reliability
    """
    pass

def handle_missing_data(data: Dict) -> Dict:
    """
    Handle missing or incomplete financial data.
    
    Strategies:
    - Interpolation
    - Industry averages
    - Historical patterns
    - Conservative estimates
    """
    pass

def get_industry_classification(ticker: str) -> Dict:
    """
    Get the industry classification for a given ticker using FMP API.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Dict containing:
        - sic_code
        - industry_title
        - industry_group
        - division
    """
    url = f"https://financialmodelingprep.com/api/v4/standard_industrial_classification?symbol={ticker}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

def get_industry_stocks(industry_title: str = None, sic_code: str = None) -> List[Dict]:
    """
    Get all stocks in a specific industry using FMP's stock screener.
    
    Args:
        industry_title (str, optional): Industry title to filter by
        sic_code (str, optional): SIC code to filter by
        
    Returns:
        List[Dict]: List of stocks in the industry with their details
        
    Example:
        >>> get_industry_stocks(industry_title="Computer Programming, Data Processing")
        >>> get_industry_stocks(sic_code="7370")
    """
    base_url = "https://financialmodelingprep.com/api/v3/stock-screener"
    
    # Build query parameters
    params = {
        "apikey": FMP_API_KEY,
        "limit": 1000,  # Adjust as needed
    }
    
    if industry_title:
        params["industry"] = industry_title
    if sic_code:
        params["sicCode"] = sic_code
        
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return []
        
    stocks = response.json()
    
    # Enrich with additional metrics
    for stock in stocks:
        stock['industry_metrics'] = {
            'market_cap': stock.get('marketCap'),
            'price': stock.get('price'),
            'beta': stock.get('beta'),
            'volume': stock.get('volume'),
            'last_annual_revenue': stock.get('revenue'),
            'sector': stock.get('sector')
        }
    
    return stocks

def analyze_industry_metrics(stocks: List[Dict]) -> Dict:
    """
    Calculate industry-wide metrics from a list of stocks.
    
    Args:
        stocks (List[Dict]): List of stocks with their metrics
        
    Returns:
        Dict containing:
        - median_market_cap
        - median_revenue_multiple
        - total_market_cap
        - company_count
        - largest_companies
        - revenue_multiple_range
    """
    if not stocks:
        return {}
        
    market_caps = [s['industry_metrics']['market_cap'] for s in stocks if s['industry_metrics']['market_cap']]
    revenues = [s['industry_metrics']['last_annual_revenue'] for s in stocks if s['industry_metrics']['last_annual_revenue']]
    
    # Calculate revenue multiples
    revenue_multiples = []
    for stock in stocks:
        mc = stock['industry_metrics']['market_cap']
        rev = stock['industry_metrics']['last_annual_revenue']
        if mc and rev and rev > 0:
            revenue_multiples.append(mc/rev)
    
    return {
        'median_market_cap': np.median(market_caps) if market_caps else None,
        'median_revenue_multiple': np.median(revenue_multiples) if revenue_multiples else None,
        'total_market_cap': sum(market_caps) if market_caps else None,
        'company_count': len(stocks),
        'largest_companies': sorted(stocks, key=lambda x: x['industry_metrics']['market_cap'], reverse=True)[:5],
        'revenue_multiple_range': {
            'min': min(revenue_multiples) if revenue_multiples else None,
            'max': max(revenue_multiples) if revenue_multiples else None,
            'p25': np.percentile(revenue_multiples, 25) if revenue_multiples else None,
            'p75': np.percentile(revenue_multiples, 75) if revenue_multiples else None
        }
    }

#####################################
# FMP API Infrastructure
#####################################

def make_fmp_request(endpoint: str, params: Dict = None) -> Dict:
    """
    Make a request to the FMP API with proper error handling and rate limiting.
    
    Args:
        endpoint (str): API endpoint (without base URL)
        params (Dict): Query parameters
        
    Returns:
        Dict: JSON response from API
    """
    base_url = "https://financialmodelingprep.com/api"
    
    # Ensure params dictionary exists
    params = params or {}
    
    # Always include API key
    params["apikey"] = FMP_API_KEY
    
    try:
        # Implement rate limiting
        time.sleep(RATE_LIMIT_PAUSE)
        
        # Make request
        response = requests.get(f"{base_url}{endpoint}", params=params)
        
        # Handle HTTP errors
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Basic validation of response
        if not data:
            return {"error": "Empty response received"}
            
        return data
        
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except ValueError as e:
        return {"error": f"JSON parsing failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def validate_fmp_response(data: Dict) -> bool:
    """
    Validate FMP API response data.
    
    Args:
        data (Dict): Response data from FMP API
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if not data:
        return False
        
    if isinstance(data, dict) and "error" in data:
        return False
        
    if isinstance(data, list) and len(data) == 0:
        return False
        
    return True

#####################################
# Enhanced Sector/Industry Analysis
#####################################

def get_sector_profile(ticker: str) -> Dict:
    """
    Get comprehensive sector and industry profile for a company.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        Dict containing:
        - sector_name
        - industry_name
        - sic_code
        - peer_companies
        - sector_metrics
        - industry_position
    """
    # Get basic profile
    profile_data = make_fmp_request(f"/v3/profile/{ticker}")
    if not validate_fmp_response(profile_data):
        return {"error": "Failed to get company profile"}
    
    # Get sector performance
    sector_name = profile_data[0].get('sector')
    sector_perf = make_fmp_request(f"/v3/sector-performance")
    
    # Get industry classification
    industry_data = make_fmp_request(f"/v4/standard_industrial_classification?symbol={ticker}")
    
    # Compile sector profile
    return {
        "sector_name": sector_name,
        "industry_name": profile_data[0].get('industry'),
        "sic_code": industry_data[0].get('sicCode') if industry_data else None,
        "sector_performance": next(
            (item for item in sector_perf if item.get('sector') == sector_name), 
            None
        ) if validate_fmp_response(sector_perf) else None,
        "company_profile": {
            "market_cap": profile_data[0].get('mktCap'),
            "beta": profile_data[0].get('beta'),
            "volume": profile_data[0].get('volAvg'),
            "exchange": profile_data[0].get('exchange'),
        }
    }

def get_sector_peers(ticker: str, limit: int = 10) -> Dict:
    """
    Get peer companies in the same sector with similar characteristics.
    
    Args:
        ticker (str): Company ticker symbol
        limit (int): Maximum number of peers to return
        
    Returns:
        Dict containing:
        - peer_list
        - comparison_metrics
        - sector_averages
    """
    # Get company profile first
    profile = get_sector_profile(ticker)
    if "error" in profile:
        return profile
    
    # Use stock screener to find peers
    screener_params = {
        "sector": profile["sector_name"],
        "industry": profile["industry_name"],
        "limit": limit * 2  # Get extra for filtering
    }
    
    peers_data = make_fmp_request("/v3/stock-screener", screener_params)
    if not validate_fmp_response(peers_data):
        return {"error": "Failed to get peer data"}
    
    # Filter and enrich peer data
    peers = []
    for peer in peers_data:
        if peer['symbol'] != ticker:  # Exclude the input company
            peers.append({
                "symbol": peer['symbol'],
                "name": peer['companyName'],
                "market_cap": peer['marketCap'],
                "price": peer['price'],
                "beta": peer.get('beta'),
                "sector": peer['sector']
            })
    
    # Sort by market cap similarity and take top 'limit' peers
    company_mkt_cap = profile["company_profile"]["market_cap"]
    peers.sort(key=lambda x: abs(x['market_cap'] - company_mkt_cap))
    peers = peers[:limit]
    
    return {
        "base_company": ticker,
        "sector": profile["sector_name"],
        "industry": profile["industry_name"],
        "peer_companies": peers,
        "sector_metrics": {
            "avg_market_cap": sum(p['market_cap'] for p in peers) / len(peers),
            "avg_beta": sum(p['beta'] for p in peers if p['beta']) / len([p for p in peers if p['beta']]),
            "total_peers_found": len(peers)
        }
    }

#####################################
# Enhanced Revenue Analysis
#####################################

def get_revenue_metrics(ticker: str) -> Dict:
    """
    Get comprehensive revenue metrics for a company.
    
    Returns:
        Dict containing:
        - annual_revenue
        - revenue_growth
        - revenue_multiple
        - sector_comparison
    """
    # Get income statement for revenue data
    income_stmt = make_fmp_request(f"/v3/income-statement/{ticker}?limit=4")
    if not validate_fmp_response(income_stmt):
        return {"error": "Failed to get income statement data"}
    
    # Get key metrics for multiples
    key_metrics = make_fmp_request(f"/v3/key-metrics/{ticker}?limit=1")
    if not validate_fmp_response(key_metrics):
        return {"error": "Failed to get key metrics"}
    
    # Calculate metrics
    latest_revenue = income_stmt[0].get('revenue', 0)
    market_cap = key_metrics[0].get('marketCap', 0)
    revenue_multiple = market_cap / latest_revenue if latest_revenue > 0 else None
    
    # Calculate year-over-year growth
    if len(income_stmt) >= 2:
        prev_revenue = income_stmt[1].get('revenue', 0)
        yoy_growth = ((latest_revenue - prev_revenue) / prev_revenue) if prev_revenue > 0 else None
    else:
        yoy_growth = None
    
    return {
        "annual_revenue": latest_revenue,
        "market_cap": market_cap,
        "revenue_multiple": revenue_multiple,
        "yoy_growth": yoy_growth,
        "period": income_stmt[0].get('date'),
        "currency": income_stmt[0].get('reportedCurrency', 'USD')
    }

def find_stocks_below_revenue(sector: str = None, max_multiple: float = 1.0) -> List[Dict]:
    """
    Find stocks trading below or near their annual revenue.
    
    Args:
        sector (str, optional): Filter by specific sector
        max_multiple (float): Maximum revenue multiple to consider
        
    Returns:
        List of stocks with their metrics
    """
    # Get screener data with basic metrics
    screener_params = {
        "limit": 1000,
    }
    if sector:
        screener_params["sector"] = sector
    
    stocks = make_fmp_request("/v3/stock-screener", screener_params)
    if not validate_fmp_response(stocks):
        return []
    
    # Filter and analyze stocks
    below_revenue = []
    for stock in stocks:
        ticker = stock.get('symbol')
        metrics = get_revenue_metrics(ticker)
        
        if "error" in metrics or not metrics.get('revenue_multiple'):
            continue
            
        if metrics['revenue_multiple'] <= max_multiple:
            below_revenue.append({
                "symbol": ticker,
                "name": stock.get('companyName'),
                "sector": stock.get('sector'),
                "industry": stock.get('industry'),
                "market_cap": metrics['market_cap'],
                "annual_revenue": metrics['annual_revenue'],
                "revenue_multiple": metrics['revenue_multiple'],
                "yoy_growth": metrics['yoy_growth']
            })
    
    # Sort by revenue multiple
    below_revenue.sort(key=lambda x: x['revenue_multiple'])
    
    return below_revenue

def analyze_revenue_vs_sector(ticker: str) -> Dict:
    """
    Compare company's revenue metrics against sector peers.
    
    Returns:
        Dict containing comparison metrics and rankings
    """
    # Get company metrics
    company_metrics = get_revenue_metrics(ticker)
    if "error" in company_metrics:
        return company_metrics
    
    # Get sector peers
    profile = get_sector_profile(ticker)
    if "error" in profile:
        return profile
    
    # Get peer metrics
    peers = get_sector_peers(ticker, limit=20)
    if "error" in peers:
        return peers
    
    # Calculate peer metrics
    peer_multiples = []
    peer_growth_rates = []
    
    for peer in peers['peer_companies']:
        peer_metrics = get_revenue_metrics(peer['symbol'])
        if "error" not in peer_metrics and peer_metrics.get('revenue_multiple'):
            peer_multiples.append(peer_metrics['revenue_multiple'])
            if peer_metrics.get('yoy_growth'):
                peer_growth_rates.append(peer_metrics['yoy_growth'])
    
    return {
        "company_metrics": company_metrics,
        "sector_comparison": {
            "median_multiple": np.median(peer_multiples) if peer_multiples else None,
            "multiple_percentile": percentile_rank(company_metrics['revenue_multiple'], peer_multiples) if peer_multiples else None,
            "median_growth": np.median(peer_growth_rates) if peer_growth_rates else None,
            "growth_percentile": percentile_rank(company_metrics['yoy_growth'], peer_growth_rates) if peer_growth_rates and company_metrics.get('yoy_growth') else None
        },
        "peer_stats": {
            "multiple_range": {
                "min": min(peer_multiples) if peer_multiples else None,
                "max": max(peer_multiples) if peer_multiples else None,
                "p25": np.percentile(peer_multiples, 25) if peer_multiples else None,
                "p75": np.percentile(peer_multiples, 75) if peer_multiples else None
            },
            "peer_count": len(peer_multiples)
        }
    }

def percentile_rank(value: float, distribution: List[float]) -> float:
    """Helper function to calculate percentile rank of a value in a distribution."""
    if not distribution:
        return None
    return sum(1 for x in distribution if x < value) / len(distribution) * 100


def analyze_margin_trends(ticker: str, years: int = 5) -> Dict:
    """
    Analyze comprehensive margin trends and cost structures.
    
    Args:
        ticker: Company ticker symbol
        years: Number of years of historical data
        
    Returns:
        Dict containing margin analysis, trends, and peer comparison
    """
    # Get historical income statements
    income_stmt = make_fmp_request(f"/v3/income-statement/{ticker}?limit={years}")
    if not validate_fmp_response(income_stmt):
        return {"error": "Failed to get income statement data"}
    
    # Calculate margin trends
    margin_trends = []
    for stmt in income_stmt:
        revenue = stmt.get('revenue', 0)
        if revenue > 0:
            margin_trends.append({
                'date': stmt['date'],
                'gross_margin': (stmt.get('grossProfit', 0) / revenue) * 100,
                'operating_margin': (stmt.get('operatingIncome', 0) / revenue) * 100,
                'net_margin': (stmt.get('netIncome', 0) / revenue) * 100,
                'cost_structure': {
                    'cogs_pct': (stmt.get('costOfRevenue', 0) / revenue) * 100,
                    'operating_expenses_pct': (stmt.get('operatingExpenses', 0) / revenue) * 100,
                    'rd_pct': (stmt.get('researchAndDevelopmentExpenses', 0) / revenue) * 100,
                    'sga_pct': (stmt.get('sellingGeneralAndAdministrativeExpenses', 0) / revenue) * 100
                }
            })
    
    # Calculate margin momentum (year-over-year changes)
    margin_momentum = []
    for i in range(len(margin_trends) - 1):
        current = margin_trends[i]
        previous = margin_trends[i + 1]
        momentum = {
            'period': f"{previous['date']} to {current['date']}",
            'gross_margin_change': current['gross_margin'] - previous['gross_margin'],
            'operating_margin_change': current['operating_margin'] - previous['operating_margin'],
            'net_margin_change': current['net_margin'] - previous['net_margin']
        }
        margin_momentum.append(momentum)
    
    # Get peer comparison
    peers = get_sector_peers(ticker, limit=10)
    peer_margins = []
    
    if "error" not in peers:
        for peer in peers['peer_companies']:
            peer_income = make_fmp_request(f"/v3/income-statement/{peer['symbol']}?limit=1")
            if validate_fmp_response(peer_income) and peer_income[0].get('revenue', 0) > 0:
                revenue = peer_income[0]['revenue']
                peer_margins.append({
                    'symbol': peer['symbol'],
                    'gross_margin': (peer_income[0].get('grossProfit', 0) / revenue) * 100,
                    'operating_margin': (peer_income[0].get('operatingIncome', 0) / revenue) *100,
                    'net_margin': (peer_income[0].get('netIncome', 0) / revenue) *100
                })
    
    # Calculate latest margins for percentile ranking
    latest_margins = margin_trends[0] if margin_trends else None
    
    return {
        'company_name': ticker,
        'analysis_period': f"{margin_trends[-1]['date']} to {margin_trends[0]['date']}",
        'current_margins': latest_margins,
        'margin_trends': margin_trends,
        'margin_momentum': margin_momentum,
        'peer_comparison': {
            'gross_margin_percentile': percentile_rank(
                latest_margins['gross_margin'] if latest_margins else None,
                [p['gross_margin'] for p in peer_margins]
            ),
            'operating_margin_percentile': percentile_rank(
                latest_margins['operating_margin'] if latest_margins else None,
                [p['operating_margin'] for p in peer_margins]
            ),
            'net_margin_percentile': percentile_rank(
                latest_margins['net_margin'] if latest_margins else None,
                [p['net_margin'] for p in peer_margins]
            ),
            'peer_stats': {
                'median_gross_margin': np.median([p['gross_margin'] for p in peer_margins]) if peer_margins else None,
                'median_operating_margin': np.median([p['operating_margin'] for p in peer_margins]) if peer_margins else None,
                'median_net_margin': np.median([p['net_margin'] for p in peer_margins]) if peer_margins else None,
                'peer_count': len(peer_margins)
            }
        }
    }

def get_sector_multiples(ticker: str) -> Dict:
    """
    Get comprehensive sector-specific multiples for analysis.
    
    Returns:
        Dict containing:
        - Company multiples
        - Sector average multiples
        - Percentile rankings
        - Growth-adjusted ratios
    """
    # Get company metrics
    key_metrics = make_fmp_request(f"/v3/key-metrics-ttm/{ticker}")
    ratios = make_fmp_request(f"/v3/ratios-ttm/{ticker}")
    
    if not validate_fmp_response(key_metrics) or not validate_fmp_response(ratios):
        return {"error": "Failed to get company metrics"}
    
    # Get sector peers for comparison
    peers = get_sector_peers(ticker, limit=20)
    if "error" in peers:
        return peers
    
    # Initialize peer metrics lists
    peer_metrics = {
        'ev_ebitda': [],
        'pe_ratio': [],
        'price_book': [],
        'price_sales': [],
        'gross_margin': [],
        'operating_margin': [],
        'net_margin': []
    }
    
    # Collect peer metrics
    for peer in peers['peer_companies']:
        peer_key_metrics = make_fmp_request(f"/v3/key-metrics-ttm/{peer['symbol']}")
        peer_ratios = make_fmp_request(f"/v3/ratios-ttm/{peer['symbol']}")
        
        if validate_fmp_response(peer_key_metrics) and validate_fmp_response(peer_ratios):
            metrics = peer_key_metrics[0]
            ratios_data = peer_ratios[0]
            
            # Collect valid metrics
            if metrics.get('enterpriseValueOverEBITDA'):
                peer_metrics['ev_ebitda'].append(metrics['enterpriseValueOverEBITDA'])
            if metrics.get('peRatio'):
                peer_metrics['pe_ratio'].append(metrics['peRatio'])
            if metrics.get('priceToBookRatio'):
                peer_metrics['price_book'].append(metrics['priceToBookRatio'])
            if ratios_data.get('priceToSalesRatio'):
                peer_metrics['price_sales'].append(ratios_data['priceToSalesRatio'])
            if ratios_data.get('grossProfitMargin'):
                peer_metrics['gross_margin'].append(ratios_data['grossProfitMargin'])
            if ratios_data.get('operatingProfitMargin'):
                peer_metrics['operating_margin'].append(ratios_data['operatingProfitMargin'])
            if ratios_data.get('netProfitMargin'):
                peer_metrics['net_margin'].append(ratios_data['netProfitMargin'])
    
    # Calculate company's metrics
    company_metrics = {
        'ev_ebitda': key_metrics[0].get('enterpriseValueOverEBITDA'),
        'pe_ratio': key_metrics[0].get('peRatio'),
        'price_book': key_metrics[0].get('priceToBookRatio'),
        'price_sales': ratios[0].get('priceToSalesRatio'),
        'gross_margin': ratios[0].get('grossProfitMargin'),
        'operating_margin': ratios[0].get('operatingProfitMargin'),
        'net_margin': ratios[0].get('netProfitMargin')
    }
    
    # Calculate sector averages and percentiles
    sector_analysis = {}
    for metric, values in peer_metrics.items():
        if values:  # Only analyze if we have peer data
            sector_analysis[metric] = {
                'company_value': company_metrics[metric],
                'sector_median': np.median(values),
                'sector_avg': np.mean(values),
                'percentile': percentile_rank(company_metrics[metric], values) if company_metrics[metric] else None,
                'range': {
                    'min': min(values),
                    'max': max(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75)
                }
            }
    
    return {
        'company_name': peers['base_company'],
        'sector': peers['sector'],
        'industry': peers['industry'],
        'metrics_analysis': sector_analysis,
        'peer_count': len(peers['peer_companies']),
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }

#####################################
# Enhanced Growth Analysis
#####################################
def analyze_business_quality(ticker: str, years: int = 5) -> Dict:
    """
    Analyze business quality metrics including returns on capital,
    cash flow efficiency, and capital allocation.
    
    Args:
        ticker: Company ticker symbol
        years: Number of years for historical analysis
        
    Returns:
        Dict containing quality metrics, trends, and peer comparison
    """
    # Get required financial statements
    income_stmt = make_fmp_request(f"/v3/income-statement/{ticker}?limit={years}")
    balance_sheet = make_fmp_request(f"/v3/balance-sheet-statement/{ticker}?limit={years}")
    cash_flow = make_fmp_request(f"/v3/cash-flow-statement/{ticker}?limit={years}")
    
    if not all(validate_fmp_response(x) for x in [income_stmt, balance_sheet, cash_flow]):
        return {"error": "Failed to get financial statements"}
    
    # Calculate quality metrics over time
    quality_trends = []
    
    for i in range(len(income_stmt)):
        if i >= len(balance_sheet) or i >= len(cash_flow):
            break
            
        inc = income_stmt[i]
        bal = balance_sheet[i]
        cf = cash_flow[i]
        
        # Calculate invested capital
        total_assets = bal.get('totalAssets', 0)
        current_liabilities = bal.get('totalCurrentLiabilities', 0)
        invested_capital = total_assets - current_liabilities
        
        # Get operating metrics
        operating_income = inc.get('operatingIncome', 0)
        net_income = inc.get('netIncome', 0)
        total_equity = bal.get('totalStockholdersEquity', 0)
        
        # Get cash flow metrics
        operating_cash_flow = cf.get('operatingCashFlow', 0)
        capex = cf.get('capitalExpenditure', 0)
        free_cash_flow = operating_cash_flow + capex  # capex is negative
        
        # Calculate key ratios
        quality_metrics = {
            'date': inc['date'],
            'roic': (operating_income * (1 - 0.21)) / invested_capital * 100 if invested_capital else None,  # Assuming 21% tax rate
            'roe': (net_income / total_equity * 100) if total_equity else None,
            'fcf_conversion': (free_cash_flow / operating_income * 100) if operating_income else None,
            'asset_turnover': inc.get('revenue', 0) / total_assets if total_assets else None,
            'working_capital_efficiency': {
                'days_receivables': (bal.get('netReceivables', 0) / inc.get('revenue', 0) * 365) if inc.get('revenue', 0) else None,
                'days_inventory': (bal.get('inventory', 0) / inc.get('costOfRevenue', 0) * 365) if inc.get('costOfRevenue', 0) else None,
                'days_payables': (bal.get('accountPayables', 0) / inc.get('costOfRevenue', 0) * 365) if inc.get('costOfRevenue', 0) else None
            },
            'capital_allocation': {
                'capex_to_revenue': (abs(capex) / inc.get('revenue', 0) * 100) if inc.get('revenue', 0) else None,
                'fcf_to_revenue': (free_cash_flow / inc.get('revenue', 0) * 100) if inc.get('revenue', 0) else None,
                'dividend_payout': (cf.get('dividendsPaid', 0) / net_income * 100) if net_income else None
            }
        }
        quality_trends.append(quality_metrics)
    
    # Get peer comparison
    peers = get_sector_peers(ticker, limit=10)
    peer_quality = []
    
    if "error" not in peers:
        for peer in peers['peer_companies']:
            peer_inc = make_fmp_request(f"/v3/income-statement/{peer['symbol']}?limit=1")
            peer_bal = make_fmp_request(f"/v3/balance-sheet-statement/{peer['symbol']}?limit=1")
            
            if validate_fmp_response(peer_inc) and validate_fmp_response(peer_bal):
                inc = peer_inc[0]
                bal = peer_bal[0]
                
                invested_capital = bal.get('totalAssets', 0) - bal.get('totalCurrentLiabilities', 0)
                operating_income = inc.get('operatingIncome', 0)
                
                if invested_capital and operating_income:
                    roic = (operating_income * (1 - 0.21)) / invested_capital * 100
                    peer_quality.append({
                        'symbol': peer['symbol'],
                        'roic': roic,
                        'roe': (inc.get('netIncome', 0) / bal.get('totalStockholdersEquity', 0) * 100) 
                               if bal.get('totalStockholdersEquity', 0) else None
                    })
    
    # Get latest metrics for percentile ranking
    latest_metrics = quality_trends[0] if quality_trends else None
    
    return {
        'company_name': ticker,
        'analysis_period': f"{quality_trends[-1]['date']} to {quality_trends[0]['date']}",
        'current_metrics': latest_metrics,
        'quality_trends': quality_trends,
        'peer_comparison': {
            'roic_percentile': percentile_rank(
                latest_metrics['roic'] if latest_metrics else None,
                [p['roic'] for p in peer_quality if p['roic']]
            ),
            'roe_percentile': percentile_rank(
                latest_metrics['roe'] if latest_metrics else None,
                [p['roe'] for p in peer_quality if p['roe']]
            ),
            'peer_stats': {
                'median_roic': np.median([p['roic'] for p in peer_quality if p['roic']]) if peer_quality else None,
                'median_roe': np.median([p['roe'] for p in peer_quality if p['roe']]) if peer_quality else None,
                'peer_count': len(peer_quality)
            }
        }
    }


def analyze_growth_metrics(ticker: str, years: int = 5) -> Dict:
    """
    Analyze comprehensive growth metrics and trends.
    
    Args:
        ticker: Company ticker symbol
        years: Number of years of historical data to analyze
        
    Returns:
        Dict containing growth metrics, trends, and peer comparison
    """
    # Get historical financials
    income_stmt = make_fmp_request(f"/v3/income-statement/{ticker}?limit={years}")
    balance_sheet = make_fmp_request(f"/v3/balance-sheet-statement/{ticker}?limit={years}")
    
    if not validate_fmp_response(income_stmt) or not validate_fmp_response(balance_sheet):
        return {"error": "Failed to get financial statements"}
    
    # Calculate growth rates
    growth_rates = {
        'revenue': [],
        'operating_income': [],
        'net_income': [],
        'eps': []
    }
    
    # Calculate year-over-year growth rates
    for i in range(len(income_stmt) - 1):
        current = income_stmt[i]
        previous = income_stmt[i + 1]
        
        growth_rates['revenue'].append(
            ((current['revenue'] - previous['revenue']) / previous['revenue'])
            if previous['revenue'] else None
        )
        
        growth_rates['operating_income'].append(
            ((current['operatingIncome'] - previous['operatingIncome']) / previous['operatingIncome'])
            if previous['operatingIncome'] else None
        )
        
        growth_rates['net_income'].append(
            ((current['netIncome'] - previous['netIncome']) / previous['netIncome'])
            if previous['netIncome'] else None
        )

        growth_rates['eps'].append(
            ((current['eps'] - previous['eps']) / previous['eps'])
            if previous['eps'] else None
        )
        
    # Calculate compound annual growth rates (CAGR)
    def calculate_cagr(start_value: float, end_value: float, years: int) -> float:
        if not start_value or not end_value or years == 0:
            return None
        return (end_value / start_value) ** (1/years) - 1
    
    latest = income_stmt[0]
    oldest = income_stmt[-1]
    period_years = len(income_stmt) - 1
    
    cagr_metrics = {
        'revenue_cagr': calculate_cagr(oldest['revenue'], latest['revenue'], period_years),
        'operating_income_cagr': calculate_cagr(oldest['operatingIncome'], latest['operatingIncome'], period_years),
        'net_income_cagr': calculate_cagr(oldest['netIncome'], latest['netIncome'], period_years),
        'eps_cagr': calculate_cagr(oldest['eps'], latest['eps'], period_years)
    }
    
    # Get peer comparison
    peers = get_sector_peers(ticker, limit=10)
    peer_growth = []
    
    if "error" not in peers:
        for peer in peers['peer_companies']:
            peer_income = make_fmp_request(f"/v3/income-statement/{peer['symbol']}?limit={years}")
            if validate_fmp_response(peer_income) and len(peer_income) >= 2:
                current_rev = peer_income[0]['revenue']
                prev_rev = peer_income[1]['revenue']
                if prev_rev:
                    peer_growth.append({
                        'symbol': peer['symbol'],
                        'yoy_growth': (current_rev - prev_rev) / prev_rev
                    })
    
    return {
        'company_name': ticker,
        'analysis_period': f"{oldest['date']} to {latest['date']}",
        'growth_rates': {
            'recent_yoy': {
                'revenue': growth_rates['revenue'][0] if growth_rates['revenue'] else None,
                'operating_income': growth_rates['operating_income'][0] if growth_rates['operating_income'] else None,
                'net_income': growth_rates['net_income'][0] if growth_rates['net_income'] else None,
                'eps': growth_rates['eps'][0] if growth_rates['eps'] else None
            },
            'historical': {
                'revenue': growth_rates['revenue'],
                'operating_income': growth_rates['operating_income'],
                'net_income': growth_rates['net_income'],
                'eps': growth_rates['eps']
            },
            'cagr': cagr_metrics
        },
        'peer_comparison': {
            'revenue_growth_percentile': percentile_rank(
                growth_rates['revenue'][0] if growth_rates['revenue'] else None,
                [p['yoy_growth'] for p in peer_growth]
            ),
            'peer_median_growth': np.median([p['yoy_growth'] for p in peer_growth]) if peer_growth else None,
            'peer_data': peer_growth
        }
    }

def get_company_classification(ticker: str) -> Dict:
    """Get company's sector and industry classification."""
    profile = make_fmp_request(f"/v3/profile/{ticker}")
    if not profile or not isinstance(profile, list):
        return {"error": "Failed to get company profile"}
    
    company_details = profile[0]
    return {
        'sector': company_details.get('sector'),
        'industry': company_details.get('industry'),
        'company_name': company_details.get('companyName'),
        'exchange': company_details.get('exchange')
    }

def get_peer_companies(ticker: str) -> Dict:
    """Get list of peer companies and validate them."""
    peers_data = make_fmp_request(f"/v4/stock_peers?symbol={ticker}")
    if not peers_data or not isinstance(peers_data, list):
        return {"error": "Failed to get peer companies"}
        
    peer_list = peers_data[0].get('peersList', []) if peers_data else []
    if not peer_list:
        return {"error": "No peers found for company"}
    
    return {
        'peers': peer_list,
        'total_peers': len(peer_list)
    }

def calculate_revenue_multiple(ticker: str) -> Dict:
    """Calculate company's revenue multiple and related metrics."""
    metrics = make_fmp_request(f"/v3/key-metrics-ttm/{ticker}")
    quote = make_fmp_request(f"/v3/quote/{ticker}")
    
    if not metrics or not quote:
        return {"error": "Failed to get company metrics"}
    
    return {
        'price_to_sales': metrics[0].get('priceToSalesRatioTTM'),
        'market_cap': quote[0].get('marketCap'),
        'pe_ratio': quote[0].get('pe'),
        'shares_outstanding': quote[0].get('sharesOutstanding')
    }

def analyze_peer_metrics(peer_list: List[str]) -> Dict:
    """Analyze peer companies' metrics and calculate comparisons."""
    peer_metrics = []
    
    for peer in peer_list:
        metrics = make_fmp_request(f"/v3/key-metrics-ttm/{peer}")
        quote = make_fmp_request(f"/v3/quote/{peer}")
        
        if metrics and quote:
            peer_metrics.append({
                'symbol': peer,
                'price_to_sales': metrics[0].get('priceToSalesRatioTTM'),
                'market_cap': quote[0].get('marketCap'),
                'pe_ratio': quote[0].get('pe')
            })
        time.sleep(0.12)  # Rate limiting
    
    if not peer_metrics:
        return {"error": "No valid peer metrics found"}
    
    multiples = [p['price_to_sales'] for p in peer_metrics if p['price_to_sales']]
    
    return {
        'peer_metrics': peer_metrics,
        'sector_average': sum(multiples) / len(multiples) if multiples else None,
        'sector_median': sorted(multiples)[len(multiples)//2] if multiples else None,
        'total_peers_analyzed': len(peer_metrics)
    }

def analyze_sector_metrics(ticker: str) -> Dict:
    """Main function that coordinates sector analysis using helper functions."""
    try:
        # Get company classification
        classification = get_company_classification(ticker)
        if "error" in classification:
            return classification
            
        # Get peer companies
        peers = get_peer_companies(ticker)
        if "error" in peers:
            return peers
            
        # Get company metrics
        company_metrics = calculate_revenue_multiple(ticker)
        if "error" in company_metrics:
            return company_metrics
            
        # Analyze peer metrics
        peer_analysis = analyze_peer_metrics(peers['peers'])
        if "error" in peer_analysis:
            return peer_analysis
        
        # Combine all analyses
        return {
            'sector_name': classification['sector'],
            'industry': classification['industry'],
            'company_metrics': company_metrics,
            'peer_analysis': peer_analysis,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"DEBUG - Sector analysis error: {str(e)}")
        return {"error": f"Sector analysis failed: {str(e)}"}

