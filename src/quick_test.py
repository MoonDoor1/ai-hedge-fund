from src.tools import (
    get_ticker_details, get_related_companies,
    get_industry_peers, get_comparable_peers,
    analyze_financial_metrics
)
from pprint import pprint
import json

def test_peer_analysis(ticker: str):
    """Test peer analysis functions with detailed debugging"""
    print(f"\n{'='*80}")
    print(f"Testing Peer Analysis for {ticker}")
    print(f"{'='*80}")
    
    # 1. Get base company details
    print("\n1. Base Company Details:")
    print("-" * 40)
    try:
        details = get_ticker_details(ticker)
        print("Raw company details:")
        pprint(details)
    except Exception as e:
        print(f"Error getting company details: {str(e)}")
        return
    
    # 2. Get related companies
    print("\n2. Related Companies:")
    print("-" * 40)
    try:
        related = get_related_companies(ticker)
        print("\nRaw related companies response:")
        pprint(related)
        
        if not related['similar_tickers']:
            print("\nWARNING: No similar tickers found!")
        if not related['peer_details']:
            print("\nWARNING: No peer details found!")
    except Exception as e:
        print(f"Error getting related companies: {str(e)}")
        return
    
    # 3. Get industry peers with SIC code
    print("\n3. Industry Peers (SIC Code Based):")
    print("-" * 40)
    try:
        industry_peers = get_industry_peers(ticker)
        print("\nRaw industry peers response:")
        pprint(industry_peers)
        
        if not industry_peers['peers']:
            print("\nWARNING: No industry peers found!")
            print(f"SIC Code used: {industry_peers['industry_classification']['value']}")
    except Exception as e:
        print(f"Error getting industry peers: {str(e)}")
        return
    
    # 4. Test financial metrics for a few peers
    if industry_peers['peers']:
        print("\n4. Testing Financial Metrics for First 2 Peers:")
        print("-" * 40)
        for peer in industry_peers['peers'][:2]:
            try:
                print(f"\nGetting metrics for {peer['ticker']}:")
                metrics = analyze_financial_metrics(peer['ticker'])
                print("Raw metrics response:")
                pprint(metrics)
            except Exception as e:
                print(f"Error getting metrics for {peer['ticker']}: {str(e)}")

def run_tests():
    """Run tests for a few companies"""
    test_tickers = [
        'JPM',   # Financial
        'AAPL',  # Tech
        'XOM'    # Energy
    ]
    
    for ticker in test_tickers:
        test_peer_analysis(ticker)
        print("\nPress Enter to continue to next company...")
        input()

if __name__ == "__main__":
    run_tests() 