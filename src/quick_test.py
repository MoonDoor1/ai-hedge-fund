from src.market_agents import (
    RevenueAnalysisAgent, 
    TechnicalAnalysisAgent,
    SectorAnalysisAgent,
    MarketSentimentAgent
)
from src.tools import make_fmp_request, analyze_sector_metrics
import pandas as pd
import time

def test_revenue_agent():
    """
    Tests the RevenueAnalysisAgent's ability to:
    1. Calculate revenue multiples
    2. Compare against sector averages
    3. Analyze growth rates
    4. Generate recommendations
    
    Expected outcomes:
    - Should return revenue multiple and growth metrics
    - Should provide sector comparison
    - Should generate a clear recommendation
    """
    print("\n=== Testing Revenue Analysis Agent ===")
    agent = RevenueAnalysisAgent()
    test_tickers = ['AAPL', 'MSFT', 'WMT']  # Mix of tech and retail
    
    for ticker in test_tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            analysis = agent.analyze(ticker)
            
            if "error" in analysis:
                print(f"❌ Error analyzing {ticker}: {analysis['error']}")
                continue
            
            # Display core metrics
            metrics = analysis.get('metrics', {})
            print("\nCore Metrics:")
            print(f"Revenue Multiple: {metrics.get('revenue_multiple', 'N/A'):.2f}x")
            print(f"Sector Average: {metrics.get('sector_average', 'N/A'):.2f}x")
            print(f"Growth Rate: {metrics.get('growth_rate', 'N/A'):.1%}")
            print(f"Current Revenue: ${metrics.get('current_revenue', 'N/A'):,.0f}")
            
            # Display recommendation
            print(f"\nRecommendation: {analysis.get('recommendation', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error testing {ticker}: {str(e)}")
            
        time.sleep(1)  # Rate limiting

def test_technical_agent():
    """
    Tests the TechnicalAnalysisAgent's ability to:
    1. Calculate moving averages and trends
    2. Generate technical indicators (MACD, RSI, etc.)
    3. Compare with price targets
    4. Identify technical signals
    
    Expected outcomes:
    - Should provide MA crossover signals
    - Should calculate technical indicators
    - Should identify overbought/oversold conditions
    - Should generate actionable recommendations
    """
    print("\n=== Testing Technical Analysis Agent ===")
    agent = TechnicalAnalysisAgent()
    test_tickers = ['AAPL', 'MSFT']  # Liquid stocks for better technical analysis
    
    for ticker in test_tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            analysis = agent.analyze(ticker)
            
            if "error" in analysis:
                print(f"❌ Error analyzing {ticker}: {analysis['error']}")
                continue
            
            # Display moving averages
            ma_data = analysis.get('moving_averages', {})
            print("\nMoving Averages:")
            print(f"MA75: ${ma_data.get('ma_75', 'N/A'):,.2f}")
            print(f"MA200: ${ma_data.get('ma_200', 'N/A'):,.2f}")
            
            # Display technical indicators
            tech_data = analysis.get('technical_indicators', {})
            print("\nTechnical Indicators:")
            print(f"RSI: {tech_data.get('rsi', 'N/A'):.1f}")
            print(f"MACD Histogram: {tech_data.get('macd', {}).get('histogram', 'N/A'):.3f}")
            
            # Display recommendation with signals
            print(f"\nRecommendation: {analysis.get('recommendation', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error testing {ticker}: {str(e)}")
            
        time.sleep(1)  # Rate limiting

def test_sector_metrics():
    """
    Tests the sector metrics analysis functionality:
    1. Peer company identification
    2. Revenue multiple calculations
    3. Sector average calculations
    
    Expected outcomes:
    - Should identify relevant peers
    - Should calculate revenue multiples
    - Should provide sector averages and rankings
    """
    print("\n=== Testing Sector Metrics Analysis ===")
    test_tickers = ['AAPL', 'MSFT']  # Tech companies for comparison
    
    for ticker in test_tickers:
        print(f"\nAnalyzing sector metrics for {ticker}...")
        try:
            metrics = analyze_sector_metrics(ticker)
            
            # Debug raw response
            print("\nDEBUG - Raw metrics response:")
            print(metrics)
            
            if "error" in metrics:
                print(f"❌ Error: {metrics['error']}")
                continue
            
            print(f"\nSector: {metrics.get('sector_name')}")
            print(f"Industry: {metrics.get('industry')}")
            
            sector_metrics = metrics.get('sector_metrics', {})
            print("\nRevenue Multiples:")
            
            # Safe formatting for potentially None values
            company_multiple = sector_metrics.get('company_multiple')
            avg_multiple = sector_metrics.get('avg_revenue_multiple')
            median_multiple = sector_metrics.get('median_revenue_multiple')
            
            print(f"Company: {company_multiple:.2f}x" if company_multiple is not None else "Company: N/A")
            print(f"Sector Average: {avg_multiple:.2f}x" if avg_multiple is not None else "Sector Average: N/A")
            print(f"Sector Median: {median_multiple:.2f}x" if median_multiple is not None else "Sector Median: N/A")
            
            # Debug peer data
            peer_metrics = metrics.get('peer_metrics', [])
            print(f"\nPeers Analyzed: {metrics.get('total_peers_analyzed', 0)}")
            if peer_metrics:
                print("\nDEBUG - Peer Details:")
                for peer in peer_metrics[:3]:  # Show first 3 peers for debug
                    print(f"Symbol: {peer.get('symbol')}")
                    print(f"Multiple: {peer.get('multiple', 'N/A'):.2f}x")
                    print(f"Market Cap: ${peer.get('market_cap', 0):,.0f}")
                    print(f"Revenue: ${peer.get('revenue', 0):,.0f}")
                    print("---")
            
        except Exception as e:
            print(f"❌ Error testing {ticker}: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    print("Starting Agent Tests...\n")
    
    try:
        # Test basic API connectivity first
        print("Testing API connectivity...")
        test_response = make_fmp_request("/v3/stock/list?limit=1")
        if test_response:
            print("✓ API connection successful")
            print(f"Sample response: {test_response[:1]}")
        else:
            print("❌ API connection failed")
            exit(1)
        
        test_sector_metrics()
        
        print("\nAll tests completed!")
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in test suite: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
