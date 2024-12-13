from src.tools import get_related_tickers
from pprint import pprint

def test_related_companies(ticker: str):
    """Test the related companies endpoint for a given ticker"""
    print(f"\n{'='*80}")
    print(f"Testing Related Companies for {ticker}")
    print(f"{'='*80}\n")
    
    try:
        related = get_related_tickers(ticker)
        print("Related Companies Response:")
        print("-" * 40)
        pprint(related)
        
        if not related:
            print("\nWARNING: No related companies found!")
        else:
            print(f"\nFound {len(related)} related companies")
            
    except Exception as e:
        print(f"Error getting related companies: {str(e)}")

def run_tests():
    """Run tests for a few companies"""
    test_tickers = [
        'JPM',   # Financial
        'AAPL',  # Tech
        'XOM'    # Energy
    ]
    
    for ticker in test_tickers:
        test_related_companies(ticker)
        print("\nPress Enter to continue to next company...")
        input()

if __name__ == "__main__":
    run_tests() 