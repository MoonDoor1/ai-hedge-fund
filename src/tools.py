from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import os


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
    Retrieves revenue data for a given stock ticker.
    
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
            - 'growth': Year-over-year growth rate (if available)
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
        ValueError: If period is not one of 'annual', 'quarterly', or 'ttm'
    """
    pass

def get_sector_classification(ticker: str) -> dict:
    """
    Retrieves sector and industry classification for a given stock ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Classification data containing:
            - 'sector': Main sector (e.g., 'Technology')
            - 'industry': Specific industry (e.g., 'Consumer Electronics')
            - 'sub_industry': Sub-industry classification if available
            - 'peer_group': List of peer company tickers
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass

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

def analyze_sector_metrics(ticker: str) -> dict:
    """
    Analyzes sector-specific metrics and comparisons.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        dict: Sector analysis containing:
            - 'sector_name': Name of the sector
            - 'sector_metrics': Dict of key sector metrics
            - 'peer_rankings': Company's rank among peers
            - 'sector_average_multiple': Average revenue multiple for sector
            - 'sector_median_multiple': Median revenue multiple for sector
            - 'percentile_rank': Company's percentile in sector
    
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
    """
    pass

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

