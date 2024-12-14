from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import time
import os

def get_sic_peers(ticker: str, tolerance: float = 0.0) -> dict:
    """
    Get peers based on SIC code matching from related companies.
    Returns companies in same industry classification.
    """
    # Get related companies first
    related = get_related_tickers(ticker)
    if not related or 'results' not in related:
        return {'error': 'No related companies found'}
    
    # Filter by SIC code
    related_tickers = [company['ticker'] for company in related['results']]
    sic_filtered = filter_by_sic_code(related_tickers, ticker, tolerance)
    
    return {
        'base_company': get_ticker_details(ticker),
        'peer_count': len(sic_filtered),
        'peers': sic_filtered,
        'classification': 'SIC Code'
    }

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
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df, period=14):
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df, window=20):
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_obv(prices_df):
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


def get_market_cap_data(ticker: str, date: str = None) -> float:
    """
    Retrieves the market capitalization for a given stock ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        date (str, optional): The date to get market cap for in 'YYYY-MM-DD' format.
                            If None, gets the most recent market cap.
    
    Returns:
        float: Market capitalization in USD
    """
    # Set up dates
    end_date = date or datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # Get price data
    df = get_price_data(ticker, start_date, end_date)
    
    # Get the closing price for the specified date
    if date:
        price = df.loc[df.index == date, 'close'].iloc[-1]
    else:
        price = df['close'].iloc[-1]
    
    # Get shares outstanding from financial metrics
    metrics = get_financial_metrics(ticker, 'quarterly', 'ttm', 1)
    shares_outstanding = metrics.get('shares_outstanding', 0)
    
    # Calculate market cap
    market_cap = price * shares_outstanding
    
    return market_cap


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
        'fiscal_year': current.get('fiscal_year')
    }

def get_analyst_ratings(ticker: str) -> dict:
    """
    Retrieves current analyst ratings and price targets for a given stock ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Analyst ratings data containing:
            - 'consensus': Overall rating (e.g., 'Buy', 'Hold', 'Sell')
            - 'price_target': Dictionary containing:
                - 'low': Lowest price target
                - 'high': Highest price target
                - 'mean': Mean price target
                - 'median': Median price target
            - 'ratings_breakdown': Dictionary of rating counts:
                - 'buy': Number of buy ratings
                - 'hold': Number of hold ratings
                - 'sell': Number of sell ratings
            - 'total_analysts': Total number of analysts covering the stock
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass

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
    """
    Analyzes sector-specific metrics and comparisons using peer analysis.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Sector analysis containing sector metrics and peer comparisons
    """
    try:
        # Get base company details and metrics
        company_details = get_ticker_details(ticker)
        if not company_details.get('market_cap'):
            raise ValueError(f"No market cap data available for {ticker}")
            
        company_metrics = analyze_financial_metrics(ticker)
        
        # Get peer analysis
        peer_analysis = get_comparable_peers(ticker)
        peers = peer_analysis.get('size_filtered_peers', [])
        
        if not peers:
            return {
                'sector_name': company_details.get('sector', 'Unknown'),
                'industry': company_details.get('industry', 'Unknown'),
                'sector_metrics': {
                    'revenue_multiple': {
                        'company': None,
                        'average': None,
                        'median': None,
                        'percentile': None
                    }
                },
                'peer_rankings': {
                    'net_margin_rank': None,
                    'roe_rank': None,
                    'total_peers': 0
                },
                'total_peers_analyzed': 0,
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
        
        # Calculate revenue multiples for company and peers
        company_revenue = get_revenue_data(ticker)
        company_multiple = (company_details['market_cap'] / company_revenue['revenue'] 
                          if company_revenue.get('revenue') and company_revenue['revenue'] != 0 
                          else None)
        
        peer_multiples = []
        peer_metrics = []
        
        for peer in peers:
            try:
                if not peer.get('market_cap'):
                    continue
                    
                peer_revenue = get_revenue_data(peer['ticker'])
                if peer_revenue.get('revenue') and peer_revenue['revenue'] != 0:
                    peer_multiple = peer['market_cap'] / peer_revenue['revenue']
                    peer_multiples.append(peer_multiple)
                
                # Get peer financial metrics
                peer_metric = analyze_financial_metrics(peer['ticker'])
                if peer_metric:
                    peer_metrics.append(peer_metric)
            except Exception as e:
                print(f"Error processing peer {peer.get('ticker', 'Unknown')}: {str(e)}")
                continue
        
        # Calculate sector averages
        sector_metrics = {
            'revenue_multiple': {
                'average': sum(peer_multiples) / len(peer_multiples) if peer_multiples else None,
                'median': sorted(peer_multiples)[len(peer_multiples)//2] if peer_multiples else None,
                'company': company_multiple,
                'percentile': (sum(1 for x in peer_multiples if x < company_multiple) / len(peer_multiples) 
                             if peer_multiples and company_multiple is not None else None)
            }
        }
        
        # Only calculate these metrics if we have valid peer data
        if peer_metrics:
            valid_margins = [p['profitability']['net_margin'] for p in peer_metrics 
                           if p.get('profitability', {}).get('net_margin') is not None]
            valid_roes = [p['profitability']['roe'] for p in peer_metrics 
                         if p.get('profitability', {}).get('roe') is not None]
            valid_debt_equity = [p['solvency']['debt_to_equity'] for p in peer_metrics 
                               if p.get('solvency', {}).get('debt_to_equity') is not None]
            
            sector_metrics.update({
                'profitability': {
                    'average_margin': sum(valid_margins) / len(valid_margins) if valid_margins else None,
                    'average_roe': sum(valid_roes) / len(valid_roes) if valid_roes else None
                },
                'solvency': {
                    'average_debt_to_equity': sum(valid_debt_equity) / len(valid_debt_equity) if valid_debt_equity else None
                }
            })
        
        # Calculate company's rankings
        company_margin = company_metrics.get('profitability', {}).get('net_margin')
        company_roe = company_metrics.get('profitability', {}).get('roe')
        
        rankings = {
            'net_margin_rank': (sum(1 for p in peer_metrics 
                                  if p.get('profitability', {}).get('net_margin') and 
                                  p['profitability']['net_margin'] > company_margin) + 1
                              if company_margin is not None else None),
            'roe_rank': (sum(1 for p in peer_metrics 
                            if p.get('profitability', {}).get('roe') and 
                            p['profitability']['roe'] > company_roe) + 1
                        if company_roe is not None else None),
            'total_peers': len(peer_metrics)
        }
        
        return {
            'sector_name': company_details.get('sector', 'Unknown'),
            'industry': company_details.get('industry', 'Unknown'),
            'sector_metrics': sector_metrics,
            'peer_rankings': rankings,
            'total_peers_analyzed': len(peers),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"Error in sector analysis: {str(e)}")
        return {
            'sector_name': company_details.get('sector', 'Unknown') if 'company_details' in locals() else 'Unknown',
            'industry': company_details.get('industry', 'Unknown') if 'company_details' in locals() else 'Unknown',
            'error': str(e),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }

def analyze_price_targets(ticker: str) -> dict:
    """
    Analyzes price targets in relation to moving averages and current price.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Price target analysis containing:
            - 'current_price': Current stock price
            - 'target_consensus': Consensus price target
            - 'ma_analysis': Moving average analysis
            - 'upside_potential': Percentage to consensus target
            - 'target_range': Dict of price target range
            - 'probability_analysis': Dict of probability metrics
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass

def get_news_events(ticker: str, start_date: str = None, end_date: str = None) -> list:
    """
    Retrieves significant news events and their potential market impact.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
    
    Returns:
        list: List of news events, each containing:
            - 'date': Date of the news event
            - 'headline': News headline
            - 'source': News source
            - 'summary': Brief summary of the news
            - 'sentiment_score': Sentiment analysis score (-1 to 1)
            - 'impact_metrics': Dict containing:
                - 'volume_change': Trading volume change after news
                - 'price_change': Price change after news
                - 'volatility_change': Volatility change after news
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass

def calculate_event_impact(ticker: str, event_date: str, window_days: int = 5) -> dict:
    """
    Calculates the market impact of a specific event over a given time window.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        event_date (str): Date of the event in 'YYYY-MM-DD' format
        window_days (int, optional): Number of days to analyze after event (default: 5)
    
    Returns:
        dict: Event impact analysis containing:
            - 'price_impact': Dict of price changes
                - 'immediate': Next day price change
                - 'window': Price change over full window
                - 'abnormal_return': Excess return over market
            - 'volume_impact': Dict of volume changes
                - 'avg_volume_change': Average volume change
                - 'peak_volume': Highest volume in window
            - 'volatility_impact': Dict of volatility measures
                - 'pre_event': Volatility before event
                - 'post_event': Volatility after event
                - 'change': Percentage change in volatility
            - 'market_reaction': Overall market reaction assessment
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
        ValueError: If event_date is invalid or not in correct format
    """
    pass

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

