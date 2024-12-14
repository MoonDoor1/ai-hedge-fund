from src.tools import get_related_tickers, get_comparable_peers
from pprint import pprint

def test_peer_analysis(ticker: str):
    """Test peer analysis with multiple filters"""
    print(f"\n{'='*80}")
    print(f"Testing Peer Analysis for {ticker}")
    print(f"{'='*80}\n")
    
    # 1. Get raw related companies
    print("1. Raw Related Companies:")
    print("-" * 40)
    try:
        related = get_related_tickers(ticker)
        pprint(related)
    except Exception as e:
        print(f"Error getting related companies: {str(e)}")
        return
    
    # 2. Get filtered comparable peers
    print("\n2. Filtered Comparable Peers:")
    print("-" * 40)
    try:
        peers = get_comparable_peers(
            ticker,
            sic_tolerance=0.0,  # Exact SIC match
            min_market_cap_pct=0.2,  # At least 20% of base company
            max_market_cap_pct=5.0   # Up to 5x base company
        )
        pprint(peers)
        
        if peers.get('error'):
            print(f"\nERROR: {peers['error']}")
        else:
            print(f"\nFound {peers['peer_count']} comparable peers")
            
    except Exception as e:
        print(f"Error getting comparable peers: {str(e)}")

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