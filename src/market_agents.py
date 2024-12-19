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
    analyze_volatility_events
)
import pandas as pd

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
            
            # Get sector comparison
            sector_data = analyze_sector_metrics(ticker)
            sector_avg_multiple = sector_data.get('sector_metrics', {}).get('avg_revenue_multiple')
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                revenue_multiple=revenue_multiple,
                sector_avg=sector_avg_multiple,
                growth_rate=growth_rate
            )
            
            return {
                "analysis_type": "revenue",
                "metrics": {
                    "revenue_multiple": revenue_multiple,
                    "sector_average": sector_avg_multiple,
                    "growth_rate": growth_rate,
                    "current_revenue": current_revenue,
                    "market_cap": market_cap,
                    "revenue_per_share": revenue_data.get('revenue_per_share'),
                    "peer_comparison": sector_data.get('peer_metrics', [])
                },
                "historical_data": revenue_data.get('historical_data', []),
                "recommendation": recommendation,
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
        """
        try:
            # Get sector metrics
            sector_data = analyze_sector_metrics(ticker)
            if "error" in sector_data:
                return {"error": f"Sector analysis failed: {sector_data['error']}"}
            
            # Get margin trends
            margin_data = analyze_margin_trends(ticker)
            
            # Get growth metrics
            growth_data = analyze_growth_metrics(ticker)
            
            return {
                "analysis_type": "sector",
                "sector_info": {
                    "sector": sector_data.get('sector_name'),
                    "industry": sector_data.get('industry'),
                    "peer_count": len(sector_data.get('peer_metrics', []))
                },
                "metrics": {
                    "margins": margin_data.get('current_margins'),
                    "margin_trends": margin_data.get('margin_trends'),
                    "growth_metrics": growth_data,
                    "sector_metrics": sector_data.get('sector_metrics'),
                    "peer_comparison": sector_data.get('peer_metrics')
                },
                "recommendation": self._generate_sector_recommendation(
                    margin_data=margin_data,
                    growth_data=growth_data,
                    sector_data=sector_data
                )
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_sector_recommendation(self, margin_data: Dict, growth_data: Dict, sector_data: Dict) -> str:
        """Generate sector-based recommendation."""
        try:
            # Score margins vs peers
            margin_score = 0
            if margin_data and 'peer_comparison' in margin_data:
                peer_comp = margin_data['peer_comparison']
                margin_score += 1 if peer_comp.get('operating_margin_percentile', 0) > 75 else -1
                
            # Score growth vs sector
            growth_score = 0
            if growth_data and not isinstance(growth_data, str):
                revenue_growth = growth_data.get('revenue', [])
                if revenue_growth and len(revenue_growth) > 0:
                    growth_score += 1 if revenue_growth[0] > 0.1 else -1
            
            total_score = margin_score + growth_score
            
            if total_score >= 1.5:
                return "STRONG SECTOR POSITION - Above average margins and growth"
            elif total_score > 0:
                return "FAVORABLE SECTOR POSITION - Competitive in sector"
            elif total_score < -1:
                return "WEAK SECTOR POSITION - Below peer performance"
            else:
                return "NEUTRAL SECTOR POSITION - Average performance"
                
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