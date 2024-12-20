from src.tools import (
    get_company_classification,
    get_peer_companies,
    calculate_revenue_multiple,
    analyze_peer_metrics,
    analyze_sector_metrics,
    make_fmp_request,
    get_stock_screener,
    get_enterprise_values,
    get_company_profile_detailed,
    get_financial_ratios_ttm,
    get_key_metrics_ttm
)
import logging
import json
from datetime import datetime
from src.market_agents import PeerAnalysisAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def test_sector_components(ticker: str):
    """Test each component of sector analysis individually."""
    logging.info(f"\n=== Testing Sector Analysis Components for {ticker} ===")
    
    # 1. Test Company Classification
    logging.info("\n1. Testing Company Classification:")
    classification = get_company_classification(ticker)
    logging.info(json.dumps(classification, indent=2))
    
    # 2. Test Peer Companies
    logging.info("\n2. Testing Peer Companies:")
    peers = get_peer_companies(ticker)
    logging.info(json.dumps(peers, indent=2))
    
    # 3. Test Revenue Multiple
    logging.info("\n3. Testing Revenue Multiple:")
    metrics = calculate_revenue_multiple(ticker)
    logging.info(json.dumps(metrics, indent=2))
    
    # 4. Test Peer Metrics (only if we got peers)
    if not "error" in peers:
        logging.info("\n4. Testing Peer Metrics:")
        peer_analysis = analyze_peer_metrics(peers['peers'])
        logging.info(json.dumps(peer_analysis, indent=2))
    
    # 5. Test Full Sector Analysis
    logging.info("\n5. Testing Full Sector Analysis:")
    full_analysis = analyze_sector_metrics(ticker)
    logging.info(json.dumps(full_analysis, indent=2))

def test_api_functions(ticker: str):
    """Test all new FMP API functions and log their responses."""
    logging.info(f"\n=== Testing API Functions for {ticker} ===")
    
    # Test Stock Screener
    logging.info("\n1. Testing Stock Screener:")
    screener_results = get_stock_screener(
        market_cap_min=1000000000,  # $1B minimum
        sector="Technology"
    )
    logging.info(json.dumps(screener_results[:3] if screener_results else None, indent=2))
    
    # Test Enterprise Values
    logging.info("\n2. Testing Enterprise Values:")
    enterprise_data = get_enterprise_values(ticker)
    logging.info(json.dumps(enterprise_data[:1] if enterprise_data else None, indent=2))
    
    # Test Company Profile
    logging.info("\n3. Testing Company Profile:")
    profile = get_company_profile_detailed(ticker)
    logging.info(json.dumps(profile[0] if profile else None, indent=2))
    
    # Test Financial Ratios
    logging.info("\n4. Testing Financial Ratios TTM:")
    ratios = get_financial_ratios_ttm(ticker)
    logging.info(json.dumps(ratios[0] if ratios else None, indent=2))
    
    # Test Key Metrics
    logging.info("\n5. Testing Key Metrics TTM:")
    metrics = get_key_metrics_ttm(ticker)
    logging.info(json.dumps(metrics[0] if metrics else None, indent=2))

def test_peer_analysis_components(ticker: str):
    """Test the complete peer analysis pipeline."""
    logging.info(f"\n=== Testing Complete Peer Analysis for {ticker} ===")
    
    agent = PeerAnalysisAgent()
    analysis = agent.analyze_peers(ticker)
    
    if "error" in analysis:
        logging.error(f"Analysis failed: {analysis['error']}")
        return
    
    # Log base company info
    logging.info("\nBase Company Analysis:")
    logging.info(json.dumps({
        'ticker': analysis['base_company']['ticker'],
        'market_cap': analysis['base_company']['market_cap'],
        'margins': analysis['base_company']['margins'],
        'multiples': analysis['base_company']['multiples']
    }, indent=2))
    
    # Log peer analysis
    logging.info("\nSelected Peers:")
    for peer in analysis['peer_analysis']['selected_peers']:
        logging.info(f"\n{peer['ticker']} (Relevance: {peer['relevance_score']:.2f})")
        logging.info(f"Reasoning: {peer['reasoning']}")
        logging.info("Key Metrics:")
        logging.info(json.dumps({
            'market_cap': peer['market_cap'],
            'margins': peer['margins'],
            'multiples': peer['multiples']
        }, indent=2))
    
    # Log summary metrics
    logging.info("\nPeer Group Summary:")
    logging.info(json.dumps(analysis['peer_analysis']['metrics_summary'], indent=2))
    
    # Log analysis summary
    logging.info("\nAnalysis Summary:")
    logging.info(analysis['peer_analysis']['analysis_summary'])

if __name__ == "__main__":
    logging.info("Starting Peer Analysis Test...")
    
    try:
        test_peer_analysis_components('AAPL')
    except KeyboardInterrupt:
        logging.info("\nTest interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
