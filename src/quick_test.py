from src.market_agents import (
    RevenueAnalysisAgent,
    SectorAnalysisAgent,
    TechnicalAnalysisAgent,
    MarketSentimentAgent
)
from src.tools import make_fmp_request
import logging
import time
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def test_sector_analysis():
    """
    Tests the SectorAnalysisAgent's functionality:
    1. Sector classification
    2. Peer identification
    3. Multiple calculations
    4. Recommendations
    
    Logs detailed information about each step and any errors encountered.
    """
    logging.info("\n=== Testing Sector Analysis Agent ===")
    agent = SectorAnalysisAgent()
    test_tickers = ['APP', 'MSOS']  # Test with known tech companies
    
    for ticker in test_tickers:
        logging.info(f"\nAnalyzing {ticker}...")
        try:
            analysis = agent.analyze(ticker)
            
            # Log raw response for debugging
            logging.debug(f"Raw analysis response: {json.dumps(analysis, indent=2)}")
            
            if "error" in analysis:
                logging.error(f"Analysis failed for {ticker}: {analysis['error']}")
                continue
            
            # Log sector information
            logging.info("\nSector Classification:")
            logging.info(f"Sector: {analysis['sector_info']['sector']}")
            logging.info(f"Industry: {analysis['sector_info']['industry']}")
            
            # Log valuation metrics
            logging.info("\nValuation Metrics:")
            metrics = analysis['valuation_metrics']
            logging.info(f"Company P/S Multiple: {metrics['company_multiple']:.2f}x" if metrics['company_multiple'] else "Company Multiple: N/A")
            logging.info(f"Sector Average: {metrics['sector_average']:.2f}x" if metrics['sector_average'] else "Sector Average: N/A")
            logging.info(f"Market Cap: ${metrics['market_cap']:,.0f}")
            logging.info(f"P/E Ratio: {metrics['pe_ratio']}")
            
            # Log peer comparison
            logging.info("\nPeer Analysis:")
            peers = analysis['peer_comparison']
            logging.info(f"Total Peers Analyzed: {peers['total_peers']}")
            
            if peers['peer_metrics']:
                logging.info("\nTop 3 Peers by Market Cap:")
                sorted_peers = sorted(peers['peer_metrics'], 
                                   key=lambda x: x['market_cap'] if x['market_cap'] else 0, 
                                   reverse=True)[:3]
                for peer in sorted_peers:
                    logging.info(f"\nPeer: {peer['symbol']}")
                    logging.info(f"P/S Multiple: {peer['price_to_sales']:.2f}x" if peer['price_to_sales'] else "P/S Multiple: N/A")
                    logging.info(f"Market Cap: ${peer['market_cap']:,.0f}" if peer['market_cap'] else "Market Cap: N/A")
            
            # Log recommendation
            logging.info(f"\nRecommendation: {analysis['recommendation']}")
            
        except Exception as e:
            logging.error(f"Error testing {ticker}: {str(e)}", exc_info=True)
        
        time.sleep(1)  # Rate limiting

def test_api_connectivity():
    """Tests basic API connectivity and response format."""
    logging.info("\nTesting API Connectivity...")
    
    try:
        test_response = make_fmp_request("/v3/stock/list?limit=1")
        if test_response:
            logging.info("✓ API connection successful")
            logging.debug(f"Sample response: {json.dumps(test_response[:1], indent=2)}")
            return True
        else:
            logging.error("❌ API connection failed")
            return False
    except Exception as e:
        logging.error(f"API test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.info("Starting Test Suite...")
    
    try:
        # Test API connectivity first
        if not test_api_connectivity():
            logging.error("Aborting tests due to API connectivity failure")
            exit(1)
        
        # Run tests
        test_sector_analysis()
        
        logging.info("\nAll tests completed!")
        
    except KeyboardInterrupt:
        logging.info("\nTests interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in test suite: {str(e)}", exc_info=True)
