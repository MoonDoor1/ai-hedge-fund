from typing import Dict, List
from src.tools import (
    get_revenue_data,
    analyze_sector_metrics,
    analyze_margin_trends,
    analyze_growth_metrics,
    analyze_price_targets,
    make_fmp_request,
    calculate_moving_averages,
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_obv,
    get_analyst_ratings,
    get_news_events,
    analyze_volatility_events,
    get_company_profile_detailed,
    get_stock_screener,
    get_enterprise_values,
    get_financial_ratios_ttm,
    get_key_metrics_ttm,
    get_peer_companies
)
import pandas as pd
import logging
import time
import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class RevenueAnalysisAgent:
    """Agent responsible for analyzing revenue metrics and identifying stocks trading below revenue."""
    
    def __init__(self):
        self.tracked_tickers = set()
        
    def analyze(self, ticker: str) -> Dict:
        """
        Analyzes revenue metrics for a given ticker.
        
        Returns:
            Dict containing analysis results and recommendations
        """
        try:
            # Get revenue data using our new function
            revenue_data = get_revenue_data(ticker)
            if "error" in revenue_data:
                return {"error": f"Failed to get revenue data: {revenue_data['error']}"}
            
            # Get current market data
            quote = make_fmp_request(f"/v3/quote/{ticker}")
            if not quote or not isinstance(quote, list):
                return {"error": "Failed to get current market data"}
            
            market_cap = quote[0].get('marketCap', 0)
            current_revenue = revenue_data.get('revenue', 0)
            
            # Calculate key metrics
            revenue_multiple = market_cap / current_revenue if current_revenue else None
            growth_rate = revenue_data.get('growth')
            
            # Get sector comparison with better error handling
            try:
                sector_data = analyze_sector_metrics(ticker)
                sector_avg_multiple = (sector_data.get('sector_metrics', {}).get('avg_revenue_multiple') 
                                     if not isinstance(sector_data, dict) or "error" not in sector_data 
                                     else None)
            except Exception as e:
                logging.error(f"Sector analysis failed: {str(e)}")
                sector_avg_multiple = None
            
            return {
                "analysis_type": "revenue",
                "metrics": {
                    "revenue_multiple": revenue_multiple,
                    "sector_average": sector_avg_multiple,
                    "growth_rate": growth_rate,
                    "current_revenue": current_revenue,
                    "market_cap": market_cap,
                    "revenue_per_share": revenue_data.get('revenue_per_share'),
                    "peer_comparison": sector_data.get('peer_metrics', []) if isinstance(sector_data, dict) else []
                },
                "historical_data": revenue_data.get('historical_data', []),
                "recommendation": self._generate_recommendation(
                    revenue_multiple=revenue_multiple,
                    sector_avg=sector_avg_multiple,
                    growth_rate=growth_rate
                ),
                "analysis_date": revenue_data.get('period_end')
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_recommendation(self, revenue_multiple: float, sector_avg: float, growth_rate: float) -> str:
        """Generate revenue-based recommendation."""
        if not all([revenue_multiple, sector_avg, growth_rate]):
            return "HOLD - Insufficient data"
            
        # Score based on multiple vs sector
        multiple_score = 1 if revenue_multiple < sector_avg else -1
        
        # Adjust for growth
        growth_score = 1 if growth_rate > 0.1 else (0 if growth_rate > 0 else -1)
        
        # Combine scores
        total_score = multiple_score + growth_score
        
        if total_score >= 1.5:
            return "STRONG BUY - Undervalued with strong growth"
        elif total_score > 0:
            return "BUY - Relatively attractive valuation"
        elif total_score < -1:
            return "SELL - Overvalued with weak growth"
        else:
            return "HOLD - Fair valuation"

class SectorAnalysisAgent:
    """Agent responsible for sector-specific analysis and comparisons."""
    
    def analyze(self, ticker: str) -> Dict:
        """
        Performs comprehensive sector analysis using multiple metrics.
        
        Returns:
            Dict containing:
            - Sector classification
            - Revenue multiples (P/S ratio)
            - Peer comparisons
            - Growth metrics
            - Recommendations
        """
        try:
            # Get sector analysis using our new modular functions
            sector_data = analyze_sector_metrics(ticker)
            if "error" in sector_data:
                return {"error": f"Sector analysis failed: {sector_data['error']}"}
            
            # Get growth metrics
            growth_data = analyze_growth_metrics(ticker)
            
            # Get margin trends
            margin_data = analyze_margin_trends(ticker)
            
            # Extract key metrics for recommendation
            company_multiple = sector_data['company_metrics']['price_to_sales']
            sector_avg = sector_data['peer_analysis']['sector_average']
            sector_median = sector_data['peer_analysis']['sector_median']
            
            return {
                "analysis_type": "sector",
                "sector_info": {
                    "sector": sector_data['sector_name'],
                    "industry": sector_data['industry']
                },
                "valuation_metrics": {
                    "company_multiple": company_multiple,
                    "sector_average": sector_avg,
                    "sector_median": sector_median,
                    "market_cap": sector_data['company_metrics']['market_cap'],
                    "pe_ratio": sector_data['company_metrics']['pe_ratio']
                },
                "peer_comparison": {
                    "total_peers": sector_data['peer_analysis']['total_peers_analyzed'],
                    "peer_metrics": sector_data['peer_analysis']['peer_metrics']
                },
                "growth_metrics": growth_data.get('growth_rates', {}),
                "margin_analysis": margin_data.get('current_margins', {}),
                "recommendation": self._generate_recommendation(
                    company_multiple=company_multiple,
                    sector_avg=sector_avg,
                    growth_data=growth_data,
                    margin_data=margin_data
                )
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_recommendation(self, company_multiple: float, sector_avg: float,
                               growth_data: Dict, margin_data: Dict) -> str:
        """Generate sector-based recommendation."""
        try:
            score = 0
            signals = []
            
            # Score based on multiple vs sector
            if company_multiple and sector_avg:
                multiple_discount = (sector_avg - company_multiple) / sector_avg
                if multiple_discount > 0.2:  # Trading at >20% discount
                    score += 2
                    signals.append("Significant valuation discount")
                elif multiple_discount > 0.1:  # Trading at >10% discount
                    score += 1
                    signals.append("Moderate valuation discount")
                elif multiple_discount < -0.2:  # Trading at >20% premium
                    score -= 2
                    signals.append("Significant valuation premium")
                elif multiple_discount < -0.1:  # Trading at >10% premium
                    score -= 1
                    signals.append("Moderate valuation premium")
            
            # Score growth metrics
            recent_growth = growth_data.get('recent_yoy', {}).get('revenue')
            if recent_growth:
                if recent_growth > 0.2:  # >20% growth
                    score += 2
                    signals.append("Strong revenue growth")
                elif recent_growth > 0.1:  # >10% growth
                    score += 1
                    signals.append("Moderate revenue growth")
                elif recent_growth < 0:
                    score -= 1
                    signals.append("Negative revenue growth")
            
            # Generate final recommendation
            if score >= 3:
                return f"STRONG BUY - {'; '.join(signals)}"
            elif score > 0:
                return f"BUY - {'; '.join(signals)}"
            elif score < -2:
                return f"SELL - {'; '.join(signals)}"
            elif score < 0:
                return f"UNDERWEIGHT - {'; '.join(signals)}"
            else:
                return f"HOLD - {'; '.join(signals)}"
                
        except Exception as e:
            return f"Unable to generate recommendation: {str(e)}"

class TechnicalAnalysisAgent:
    """Agent responsible for technical analysis and price targets."""
    
    def analyze(self, ticker: str) -> Dict:
        """
        Performs comprehensive technical analysis including moving averages,
        price targets, and technical indicators.
        
        Returns:
            Dict containing technical analysis results and recommendations
        """
        try:
            # Get moving averages and support/resistance
            ma_data = calculate_moving_averages(ticker)
            if "error" in ma_data:
                return {"error": f"Failed to get MA data: {ma_data['error']}"}
            
            # Get price targets
            target_data = analyze_price_targets(ticker)
            if "error" in target_data:
                return {"error": f"Failed to get price targets: {target_data['error']}"}
            
            # Get historical prices for technical indicators
            historical = make_fmp_request(f"/v3/historical-price-full/{ticker}")
            if not historical or 'historical' not in historical:
                return {"error": "Failed to get historical data"}
            
            # Convert to DataFrame for technical analysis
            df = pd.DataFrame(historical['historical'])
            df = df.sort_values('date')  # Ensure chronological order
            
            # Calculate technical indicators
            macd_line, signal_line = calculate_macd(df)
            rsi = calculate_rsi(df)
            upper_band, lower_band = calculate_bollinger_bands(df)
            obv = calculate_obv(df)
            
            # Get latest values
            latest_data = {
                'macd': {
                    'macd_line': macd_line.iloc[-1],
                    'signal_line': signal_line.iloc[-1],
                    'histogram': macd_line.iloc[-1] - signal_line.iloc[-1]
                },
                'rsi': rsi.iloc[-1],
                'bollinger_bands': {
                    'upper': upper_band.iloc[-1],
                    'lower': lower_band.iloc[-1],
                    'current_price': df['close'].iloc[-1]
                },
                'obv': obv.iloc[-1],
                'obv_change': (obv.iloc[-1] - obv.iloc[-5]) / obv.iloc[-5]  # 5-day OBV change
            }
            
            # Generate comprehensive recommendation
            recommendation = self._generate_recommendation(
                ma_data=ma_data,
                target_data=target_data,
                technical_data=latest_data
            )
            
            return {
                "analysis_type": "technical",
                "moving_averages": {
                    "ma_75": ma_data.get('ma_75'),
                    "ma_200": ma_data.get('ma_200'),
                    "trend_signals": ma_data.get('trend_signals'),
                    "support_resistance": ma_data.get('support_resistance')
                },
                "price_targets": {
                    "consensus": target_data.get('targetConsensus'),
                    "high": target_data.get('targetHigh'),
                    "low": target_data.get('targetLow')
                },
                "technical_indicators": latest_data,
                "recommendation": recommendation
            }
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def _generate_recommendation(self, ma_data: Dict, target_data: Dict, technical_data: Dict) -> str:
        """Generate technical analysis based recommendation."""
        try:
            score = 0
            signals = []
            
            # Score moving averages
            if ma_data.get('trend_signals', {}).get('golden_cross'):
                score += 1
                signals.append("Golden Cross detected")
            elif ma_data.get('trend_signals', {}).get('death_cross'):
                score -= 1
                signals.append("Death Cross detected")
            
            # Score RSI
            rsi = technical_data.get('rsi')
            if rsi:
                if rsi < 30:
                    score += 1
                    signals.append("Oversold (RSI)")
                elif rsi > 70:
                    score -= 1
                    signals.append("Overbought (RSI)")
            
            # Score MACD
            macd = technical_data.get('macd', {})
            if macd.get('histogram', 0) > 0 and macd.get('macd_line', 0) > 0:
                score += 1
                signals.append("Positive MACD crossover")
            elif macd.get('histogram', 0) < 0 and macd.get('macd_line', 0) < 0:
                score -= 1
                signals.append("Negative MACD crossover")
            
            # Score price targets
            current_price = ma_data.get('current_price', 0)
            target_consensus = target_data.get('targetConsensus')
            if current_price and target_consensus:
                upside = (target_consensus - current_price) / current_price
                if upside > 0.15:
                    score += 1
                    signals.append(f"Price target upside: {upside:.1%}")
                elif upside < -0.15:
                    score -= 1
                    signals.append(f"Price target downside: {upside:.1%}")
            
            # Generate final recommendation
            if score >= 2:
                return f"STRONG BUY - {'; '.join(signals)}"
            elif score == 1:
                return f"BUY - {'; '.join(signals)}"
            elif score == -1:
                return f"SELL - {'; '.join(signals)}"
            elif score <= -2:
                return f"STRONG SELL - {'; '.join(signals)}"
            else:
                return f"HOLD - Mixed signals: {'; '.join(signals)}"
                
        except Exception as e:
            return f"Unable to generate recommendation: {str(e)}"

class MarketSentimentAgent:
    """Agent responsible for analyzing market sentiment, news, and volatility."""
    
    def analyze(self, ticker: str) -> Dict:
        """
        Analyzes market sentiment and news impact.
        
        Returns:
            Dict containing:
            - analyst_ratings: Current ratings
            - news_impact: Recent news analysis
            - volatility_events: Significant volatility events
        """
        ratings = get_analyst_ratings(ticker)
        news = get_news_events(ticker)
        volatility = analyze_volatility_events(ticker)
        
        return {
            "analysis_type": "sentiment",
            "ratings": ratings,
            "news_analysis": news,
            "volatility": volatility,
            "recommendation": self._generate_recommendation(ratings, news, volatility)
        }

    def _generate_recommendation(self, ratings: Dict, news: List, volatility: Dict) -> str:
        """
        Generate market sentiment recommendation based on analyst ratings,
        news sentiment, and volatility patterns.
        
        Customization Points:
        1. News Analysis:
           - Source credibility weighting
           - Event type classification
           - Impact magnitude scoring
        
        2. Analyst Ratings:
           - Historical accuracy weighting
           - Institution reputation factors
           - Rating change momentum
        
        3. Volatility Patterns:
           - Event classification
           - Volume correlation
           - Historical pattern matching
        """
        score = 0
        reasons = []
        
        # CUSTOMIZATION: Adjust weights for different factors
        weights = {
            'analyst_ratings': 0.4,
            'news_sentiment': 0.3,
            'volatility': 0.3
        }
        
        # Analyze Analyst Ratings
        buy_ratio = ratings.get('buy', 0) / ratings.get('total_analysts', 1)
        if buy_ratio > 0.7:
            score += weights['analyst_ratings'] * 2
            reasons.append(f"{int(buy_ratio*100)}% analyst buy ratings")
        elif buy_ratio > 0.5:
            score += weights['analyst_ratings']
            reasons.append("Majority analyst buy ratings")
        
        # Analyze News Sentiment
        recent_news = news[:5]  # Focus on most recent news
        positive_news = sum(1 for n in recent_news if n.get('sentiment_score', 0) > 0.2)
        if positive_news >= 3:
            score += weights['news_sentiment'] * 2
            reasons.append("Strong positive news sentiment")
        elif positive_news >= 2:
            score += weights['news_sentiment']
            reasons.append("Positive news sentiment")
        
        # Analyze Volatility Patterns
        vol_trend = volatility['summary_metrics'].get('volatility_trend')
        if vol_trend == 'decreasing':
            score += weights['volatility']
            reasons.append("Decreasing volatility")
        elif vol_trend == 'increasing':
            score -= weights['volatility']
            reasons.append("Increasing volatility")
        
        # Generate final recommendation
        if score >= 1.5:
            return f"STRONG BUY - {'; '.join(reasons)}"
        elif score >= 0.5:
            return f"BUY - {'; '.join(reasons)}"
        elif score <= -1.0:
            return f"SELL - Negative sentiment indicators"
        else:
            return f"HOLD - Mixed sentiment signals"

class PortfolioRecommendationAgent:
    """Master agent that combines all analyses to make final recommendations."""
    
    def __init__(self):
        self.revenue_agent = RevenueAnalysisAgent()
        self.sector_agent = SectorAnalysisAgent()
        self.technical_agent = TechnicalAnalysisAgent()
        self.sentiment_agent = MarketSentimentAgent()
    
    def analyze(self, ticker: str) -> Dict:
        """
        Generates comprehensive analysis and recommendations.
        
        Returns:
            Dict containing combined analysis and final recommendation
        """
        revenue_analysis = self.revenue_agent.analyze(ticker)
        sector_analysis = self.sector_agent.analyze(ticker)
        technical_analysis = self.technical_agent.analyze(ticker)
        sentiment_analysis = self.sentiment_agent.analyze(ticker)
        
        return {
            "ticker": ticker,
            "analyses": {
                "revenue": revenue_analysis,
                "sector": sector_analysis,
                "technical": technical_analysis,
                "sentiment": sentiment_analysis
            },
            "final_recommendation": self._generate_final_recommendation(
                revenue_analysis,
                sector_analysis,
                technical_analysis,
                sentiment_analysis
            )
        }
    
    def _generate_final_recommendation(self, revenue_analysis: Dict, 
                                     sector_analysis: Dict,
                                     technical_analysis: Dict,
                                     sentiment_analysis: Dict) -> Dict:
        """
        Generate final portfolio recommendation by combining all analyses.
        
        The confidence score measures how strongly the different signals agree,
        regardless of buy/sell direction. High confidence means strong agreement
        across different analysis types.
        """
        # CUSTOMIZATION: Adjust weights based on market conditions
        weights = {
            'revenue': 0.30,    # Fundamental factor
            'sector': 0.25,     # Peer comparison
            'technical': 0.25,  # Price action
            'sentiment': 0.20   # Market mood
        }
        
        score = 0
        reasons = []
        
        def extract_signal(recommendation: str) -> float:
            if 'STRONG BUY' in recommendation: return 2.0
            if 'BUY' in recommendation: return 1.0
            if 'SELL' in recommendation: return -1.0
            if 'STRONG SELL' in recommendation: return -2.0
            return 0  # HOLD
        
        # Get all signals
        signals = {
            'revenue': extract_signal(revenue_analysis['recommendation']),
            'sector': extract_signal(sector_analysis['recommendation']),
            'technical': extract_signal(technical_analysis['recommendation']),
            'sentiment': extract_signal(sentiment_analysis['recommendation'])
        }
        
        # Calculate weighted score for buy/sell decision
        for source, signal in signals.items():
            score += signal * weights[source]
            if abs(signal) >= 1:
                reasons.append(f"{source.title()}: {revenue_analysis['recommendation']}")
        
        # Calculate enhanced confidence score
        def calculate_enhanced_confidence(signals: Dict[str, float]) -> tuple:
            """
            Calculate confidence based on signal agreement and strength,
            independent of buy/sell direction.
            
            Returns:
                tuple: (confidence, agreement_ratio, strength_score, clustering_score)
            """
            signal_values = list(signals.values())
            
            # Direction agreement (are signals pointing the same way?)
            signal_directions = [1 if s > 0 else -1 if s < 0 else 0 for s in signal_values]
            agreement_ratio = abs(sum(signal_directions)) / len(signal_directions)
            
            # Signal strength (how strong are the signals?)
            strength_score = sum(abs(s) for s in signal_values) / (len(signal_values) * 2)
            
            # Signal clustering (how close are signals to each other?)
            mean_signal = sum(abs(s) for s in signal_values) / len(signal_values)
            variance = sum((abs(s) - mean_signal) ** 2 for s in signal_values) / len(signal_values)
            clustering_score = 1 - min(variance, 1)  # Lower variance = higher clustering
            
            # Combine scores with weights
            confidence = (
                agreement_ratio * 0.4 +    # Direction agreement most important
                strength_score * 0.35 +    # Signal strength next
                clustering_score * 0.25    # Clustering least important
            )
            
            return (
                round(confidence, 2),
                round(agreement_ratio, 2),
                round(strength_score, 2),
                round(clustering_score, 2)
            )
        
        confidence, agreement_ratio, strength_score, clustering_score = calculate_enhanced_confidence(signals)
        
        # Generate final recommendation
        return {
            'action': 'STRONG BUY' if score >= 1.5 else
                     'BUY' if score >= 0.5 else
                     'SELL' if score <= -0.5 else
                     'STRONG SELL' if score <= -1.5 else
                     'HOLD',
            'confidence': confidence,
            'score': round(score, 2),
            'reasons': reasons,
            'risk_level': 'HIGH' if confidence < 0.3 else
                         'LOW' if confidence > 0.7 else
                         'MEDIUM',
            'signal_breakdown': {
                'agreement_strength': agreement_ratio,
                'signal_strength': strength_score,
                'signal_clustering': clustering_score
            }
        } 

class PeerAnalysisAgent:
    """Agent responsible for identifying and validating relevant peer companies."""
    
    def __init__(self):
        self.llm = None  # We'll add LLM setup later
        self.cache = {}  # Optional: Cache results to avoid repeat API calls
        
    def _get_initial_peers(self, ticker: str) -> Dict:
        """Get initial peer list using multiple sources."""
        try:
            # Get direct peers from API
            peers = get_peer_companies(ticker)
            if "error" in peers:
                return peers
            
            # Get company profile for context
            company_profile = get_company_profile_detailed(ticker)
            if not company_profile:
                return {"error": "Failed to get company profile"}
            
            # Add major tech companies for large tech firms
            tech_giants = ['MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA']
            if company_profile[0].get('mktCap', 0) > 100000000000:  # $100B+
                peers['peers'] = list(set(peers['peers'] + tech_giants))
            
            # Add platform/ecosystem companies
            platform_companies = ['NFLX', 'SPOT', 'ADBE', 'CRM', 'SHOP']
            if 'software' in company_profile[0].get('sector', '').lower():
                peers['peers'] = list(set(peers['peers'] + platform_companies))
            
            return {
                'base_company': company_profile[0],
                'peers': peers['peers'],
                'total_peers': len(peers['peers'])
            }
            
        except Exception as e:
            return {"error": f"Peer retrieval failed: {str(e)}"}

    def _enrich_peer_data(self, ticker: str, peer_candidates: List[str]) -> Dict:
        """
        Gather detailed metrics for each peer candidate.
        
        Args:
            ticker: Base company ticker
            peer_candidates: List of potential peer tickers
            
        Returns:
            Dict containing enriched data for base company and peers
        """
        try:
            enriched_data = {
                'base_company': {},
                'peers': []
            }
            
            # Get base company metrics first
            base_metrics = {
                'profile': get_company_profile_detailed(ticker),
                'enterprise': get_enterprise_values(ticker),
                'ratios': get_financial_ratios_ttm(ticker),
                'metrics': get_key_metrics_ttm(ticker)
            }
            
            if all(base_metrics.values()):
                enriched_data['base_company'] = {
                    'ticker': ticker,
                    'description': base_metrics['profile'][0].get('description', ''),
                    'business_model': base_metrics['profile'][0].get('description', ''),
                    'market_cap': base_metrics['profile'][0].get('mktCap'),
                    'revenue': base_metrics['metrics'][0].get('revenuePerShareTTM', 0) * 
                              base_metrics['profile'][0].get('sharesOutstanding', 0),
                    'margins': {
                        'gross': base_metrics['ratios'][0].get('grossProfitMarginTTM'),
                        'operating': base_metrics['ratios'][0].get('operatingProfitMarginTTM'),
                        'net': base_metrics['ratios'][0].get('netProfitMarginTTM')
                    },
                    'growth': {
                        'revenue': base_metrics['metrics'][0].get('revenueGrowthTTM'),
                        'earnings': base_metrics['metrics'][0].get('epsgrowthTTM')
                    },
                    'multiples': {
                        'ev_revenue': base_metrics['enterprise'][0].get('enterpriseValueMultipleTTM'),
                        'pe': base_metrics['ratios'][0].get('peRatioTTM'),
                        'ps': base_metrics['ratios'][0].get('priceToSalesRatioTTM')
                    }
                }
            
            # Get peer metrics
            for peer in peer_candidates:
                try:
                    peer_metrics = {
                        'profile': get_company_profile_detailed(peer),
                        'enterprise': get_enterprise_values(peer),
                        'ratios': get_financial_ratios_ttm(peer),
                        'metrics': get_key_metrics_ttm(peer)
                    }
                    
                    if all(peer_metrics.values()):
                        enriched_data['peers'].append({
                            'ticker': peer,
                            'description': peer_metrics['profile'][0].get('description', ''),
                            'business_model': peer_metrics['profile'][0].get('description', ''),
                            'market_cap': peer_metrics['profile'][0].get('mktCap'),
                            'revenue': peer_metrics['metrics'][0].get('revenuePerShareTTM', 0) * 
                                     peer_metrics['profile'][0].get('sharesOutstanding', 0),
                            'margins': {
                                'gross': peer_metrics['ratios'][0].get('grossProfitMarginTTM'),
                                'operating': peer_metrics['ratios'][0].get('operatingProfitMarginTTM'),
                                'net': peer_metrics['ratios'][0].get('netProfitMarginTTM')
                            },
                            'growth': {
                                'revenue': peer_metrics['metrics'][0].get('revenueGrowthTTM'),
                                'earnings': peer_metrics['metrics'][0].get('epsgrowthTTM')
                            },
                            'multiples': {
                                'ev_revenue': peer_metrics['enterprise'][0].get('enterpriseValueMultipleTTM'),
                                'pe': peer_metrics['ratios'][0].get('peRatioTTM'),
                                'ps': peer_metrics['ratios'][0].get('priceToSalesRatioTTM')
                            }
                        })
                    time.sleep(0.12)  # Rate limiting
                    
                except Exception as e:
                    logging.error(f"Failed to enrich peer {peer}: {str(e)}")
                    continue
            
            return enriched_data
            
        except Exception as e:
            return {"error": f"Data enrichment failed: {str(e)}"}

    def _validate_peers_with_llm(self, base_company: Dict, peer_tickers: List[str]) -> Dict:
        """Use LLM to validate and select most relevant peers."""
        try:
            # Get company profiles for all peers
            peer_profiles = []
            for ticker in peer_tickers:
                profile = get_company_profile_detailed(ticker)
                if profile and len(profile) > 0:
                    peer_profiles.append({
                        'ticker': ticker,
                        'description': profile[0].get('description', ''),
                        'market_cap': profile[0].get('mktCap', 0)
                    })
                time.sleep(0.12)  # Rate limiting
            
            # Construct the prompt
            prompt = f"""
            You are a professional equity analyst specializing in competitive analysis. 
            Given the following company and a list of potential peers, identify and rank the top 5 most similar companies.

            Evaluate similarity based on:
            - Core business model
            - Revenue sources
            - Market position
            - Competitive overlap

            Target Company:
            {base_company['symbol']}
            {base_company['description']}
            Market Cap: ${base_company['mktCap']:,.0f}

            Potential Peers:
            {json.dumps(peer_profiles, indent=2)}

            Instructions:
            1. Select the 5 most similar companies
            2. Score each on a scale of 0-1 (1 being most similar)
            3. Explain key similarities and differences
            4. Focus on business fundamentals, not just industry classification

            Return ONLY a JSON response with this exact structure:
            {{
                "selected_peers": [
                    {{
                        "ticker": "symbol",
                        "relevance_score": float (0-1),
                        "reasoning": "clear explanation of similarity and differences"
                    }}
                ],
                "analysis_summary": "brief overview of the peer group selection logic"
            }}
            """

            # Call GPT-4
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in peer company analysis. You must respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            # Parse the response
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                return {"error": "Failed to parse LLM response"}

        except Exception as e:
            return {"error": f"LLM validation failed: {str(e)}"}

    def analyze_peers(self, ticker: str) -> Dict:
        """Complete peer analysis pipeline."""
        try:
            logging.info(f"\nStarting peer analysis for {ticker}")
            
            # 1. Get initial peers
            logging.info("\n1. Getting initial peers...")
            initial_results = self._get_initial_peers(ticker)
            if "error" in initial_results:
                logging.error(f"Failed to get initial peers: {initial_results['error']}")
                return initial_results
            
            logging.info(f"Found {initial_results['total_peers']} initial peers: {initial_results['peers']}")
            
            # 2. Filter with LLM
            logging.info("\n2. Validating peers with LLM...")
            validated_peers = self._validate_peers_with_llm(
                initial_results['base_company'],
                initial_results['peers']
            )
            if "error" in validated_peers:
                logging.error(f"LLM validation failed: {validated_peers['error']}")
                return validated_peers
            
            # 3. Get selected peer tickers
            selected_tickers = [peer['ticker'] for peer in validated_peers['selected_peers']]
            logging.info(f"LLM selected peers: {selected_tickers}")
            
            # 4. Enrich the filtered peer data
            logging.info("\n3. Enriching peer data...")
            enriched_data = self._enrich_peer_data(ticker, selected_tickers)
            if "error" in enriched_data:
                logging.error(f"Data enrichment failed: {enriched_data['error']}")
                return enriched_data
            
            logging.info(f"Enriched data received for {len(enriched_data['peers'])} peers")
            
            # 5. Calculate metrics summary with error handling
            logging.info("\n4. Calculating metrics summary...")
            try:
                # Log raw data for debugging
                logging.info("Checking peer data validity...")
                for peer in enriched_data['peers']:
                    logging.info(f"\nValidating data for {peer['ticker']}:")
                    logging.info(f"Market Cap: {peer.get('market_cap')}")
                    logging.info(f"Margins: {peer.get('margins')}")
                    logging.info(f"Multiples: {peer.get('multiples')}")
                
                valid_peers = [p for p in enriched_data['peers'] if all([
                    # Required metrics
                    p.get('market_cap'),
                    p.get('margins', {}).get('gross') is not None,
                    p.get('margins', {}).get('operating') is not None,
                    p.get('margins', {}).get('net') is not None,
                    # Optional metrics (remove from validation)
                    # p.get('multiples', {}).get('ev_revenue'),
                    p.get('multiples', {}).get('pe') is not None,
                    p.get('multiples', {}).get('ps') is not None
                ])]
                
                logging.info(f"\nFound {len(valid_peers)} valid peers out of {len(enriched_data['peers'])} total")
                
                if not valid_peers:
                    logging.error("No peers had complete metrics")
                    return {"error": "No valid peer data available for comparison"}
                    
                metrics_summary = {
                    'avg_market_cap': sum(p['market_cap'] for p in valid_peers) / len(valid_peers),
                    'avg_margins': {
                        'gross': sum(p['margins']['gross'] for p in valid_peers) / len(valid_peers),
                        'operating': sum(p['margins']['operating'] for p in valid_peers) / len(valid_peers),
                        'net': sum(p['margins']['net'] for p in valid_peers) / len(valid_peers)
                    },
                    'avg_multiples': {
                        'pe': sum(p['multiples']['pe'] for p in valid_peers) / len(valid_peers),
                        'ps': sum(p['multiples']['ps'] for p in valid_peers) / len(valid_peers)
                    }
                }
                logging.info("Successfully calculated metrics summary")
                
            except Exception as e:
                logging.error(f"Failed to calculate metrics: {str(e)}")
                metrics_summary = {"error": f"Failed to calculate metrics: {str(e)}"}
            
            # 6. Return combined analysis
            logging.info("\n5. Combining final analysis...")
            return {
                'base_company': enriched_data['base_company'],
                'peer_analysis': {
                    'selected_peers': [
                        {
                            **next((p for p in validated_peers['selected_peers'] if p['ticker'] == peer['ticker']), {}),
                            **peer
                        }
                        for peer in enriched_data['peers']
                    ],
                    'analysis_summary': validated_peers['analysis_summary'],
                    'metrics_summary': metrics_summary
                }
            }
            
        except Exception as e:
            logging.error(f"Peer analysis failed: {str(e)}")
            return {"error": f"Peer analysis failed: {str(e)}"}