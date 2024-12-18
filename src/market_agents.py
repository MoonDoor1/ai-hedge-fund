from typing import Dict, List
import json
from src.tools import (
    get_market_cap_data, get_revenue_data, calculate_revenue_multiple,
    get_sector_classification, analyze_sector_metrics,
    calculate_moving_averages, analyze_price_targets,
    get_analyst_ratings, get_news_events,
    analyze_volatility_events, calculate_event_impact
)

class RevenueAnalysisAgent:
    """Agent responsible for analyzing revenue metrics and identifying stocks trading below revenue."""
    
    def __init__(self):
        self.tracked_tickers = set()
        
    def analyze(self, ticker: str) -> Dict:
        """
        Analyzes revenue metrics for a given ticker.
        
        Returns:
            Dict containing:
            - revenue_multiple: Current revenue multiple
            - below_revenue: Boolean indicating if trading below revenue
            - pe_ratio: Current P/E ratio
            - recommendation: Buy/Hold/Sell based on revenue metrics
        """
        market_cap = get_market_cap_data(ticker)
        revenue_data = get_revenue_data(ticker)
        multiple_data = calculate_revenue_multiple(ticker)
        
        return {
            "analysis_type": "revenue",
            "metrics": multiple_data,
            "recommendation": self._generate_recommendation(multiple_data)
        }

    def _generate_recommendation(self, metrics: Dict) -> str:
        """
        Generate revenue-based recommendation by analyzing revenue multiples
        and comparing to sector averages.
        
        Customization Points:
        1. Multiple thresholds (1.0x, 0.7x, 1.3x) - adjust based on:
           - Sector norms (e.g., higher for SaaS)
           - Market conditions
           - Growth rates
        
        2. Revenue quality factors to consider adding:
           - Recurring revenue %
           - YoY growth rate
           - Gross margin
        """
        # Extract key metrics
        revenue_multiple = metrics.get('multiple')
        sector_avg_multiple = metrics.get('sector_average')
        growth_rate = metrics.get('growth_rate', 0)
        
        if not revenue_multiple or not sector_avg_multiple:
            return "HOLD - Insufficient data"
        
        # CUSTOMIZATION: Adjust multiple thresholds based on growth
        # Example: Higher growth = higher acceptable multiple
        growth_adjustment = min(max(growth_rate, 0), 0.5)  # Cap at 50%
        threshold_adjustment = 1 + growth_adjustment
        
        # CUSTOMIZATION: Add sector-specific adjustments
        # sector_type = metrics.get('sector_type')
        # if sector_type == 'SaaS':
        #     threshold_adjustment *= 1.5
        
        # Generate recommendation
        if revenue_multiple < (1.0 * threshold_adjustment):
            return "STRONG BUY - Trading below revenue"
        elif revenue_multiple < (sector_avg_multiple * 0.7 * threshold_adjustment):
            return "BUY - Trading significantly below sector average"
        elif revenue_multiple > (sector_avg_multiple * 1.3 * threshold_adjustment):
            return "SELL - Trading significantly above sector average"
        else:
            return "HOLD - Trading within normal range"

class SectorAnalysisAgent:
    """Agent responsible for sector-specific analysis and comparisons."""
    
    def analyze(self, ticker: str) -> Dict:
        """
        Performs sector-specific analysis.
        
        Returns:
            Dict containing:
            - sector_info: Sector classification
            - peer_comparison: Comparison with peer companies
            - sector_metrics: Key sector-specific metrics
        """
        sector_data = get_sector_classification(ticker)
        sector_metrics = analyze_sector_metrics(ticker)
        
        return {
            "analysis_type": "sector",
            "sector_data": sector_data,
            "metrics": sector_metrics,
            "recommendation": self._generate_recommendation(sector_metrics)
        }

    def _generate_recommendation(self, metrics: Dict) -> str:
        """
        Generate sector-based recommendation by comparing company metrics
        against peer group.
        
        Customization Points:
        1. Scoring weights - adjust importance of:
           - Margin comparison
           - Growth rates
           - Market position
        
        2. Peer group selection:
           - Size similarity
           - Business model
           - Geographic exposure
        """
        sector_metrics = metrics.get('sector_metrics', {})
        peer_rankings = metrics.get('peer_rankings', {})
        
        # Initialize scoring system
        score = 0
        reasons = []
        
        # CUSTOMIZATION: Adjust weights for different metrics
        weights = {
            'margin_rank': 2.0,  # Higher weight for profitability
            'growth_rank': 1.5,  # Medium weight for growth
            'multiple_rank': 1.0  # Lower weight for valuation
        }
        
        # Score: Net Margin Ranking
        margin_rank = peer_rankings.get('net_margin_rank', 0)
        if margin_rank <= 3:
            score += weights['margin_rank'] * 2
            reasons.append("Top 3 in sector by net margin")
        elif margin_rank <= 5:
            score += weights['margin_rank']
            reasons.append("Top 5 in sector by net margin")
        
        # Score: Growth Ranking
        growth_rank = peer_rankings.get('growth_rank', 0)
        if growth_rank <= 3:
            score += weights['growth_rank'] * 2
            reasons.append("Top 3 in sector by growth")
        elif growth_rank <= 5:
            score += weights['growth_rank']
            reasons.append("Top 5 in sector by growth")
        
        # Score: Multiple Comparison
        multiple_percentile = sector_metrics.get('revenue_multiple', {}).get('percentile', 50)
        if multiple_percentile < 25:
            score += weights['multiple_rank'] * 2
            reasons.append("Attractive valuation vs peers")
        elif multiple_percentile < 40:
            score += weights['multiple_rank']
            reasons.append("Reasonable valuation vs peers")
        
        # CUSTOMIZATION: Adjust thresholds for final recommendation
        # Consider market conditions, sector momentum, etc.
        if score >= 6:
            return f"STRONG BUY - {'; '.join(reasons)}"
        elif score >= 3:
            return f"BUY - {'; '.join(reasons)}"
        elif score >= 0:
            return f"HOLD - Mixed sector comparison"
        else:
            return "SELL - Underperforming sector peers"

class TechnicalAnalysisAgent:
    """Agent responsible for technical analysis and price targets."""
    
    def analyze(self, ticker: str) -> Dict:
        """
        Performs technical analysis.
        
        Returns:
            Dict containing:
            - moving_averages: MA analysis
            - price_targets: Target price analysis
            - technical_signals: Buy/Sell signals
        """
        ma_data = calculate_moving_averages(ticker)
        target_data = analyze_price_targets(ticker)
        
        return {
            "analysis_type": "technical",
            "moving_averages": ma_data,
            "price_targets": target_data,
            "recommendation": self._generate_recommendation(ma_data, target_data)
        }

    def _generate_recommendation(self, ma_data: Dict, target_data: Dict) -> str:
        """
        Generate technical analysis recommendation based on moving averages,
        price targets, and technical indicators.
        
        Customization Points:
        1. Moving Average Signals:
           - Adjust MA periods (75/200 default)
           - Add different MA types (EMA, WMA)
           - Volume-weighted adjustments
        
        2. Price Target Analysis:
           - Analyst credibility weighting
           - Target dispersion impact
           - Recent revision weighting
        
        3. Signal Strength:
           - Adjust threshold levels
           - Add momentum indicators
           - Consider volatility regime
        """
        signals = []
        
        # CUSTOMIZATION: Adjust MA thresholds based on volatility
        volatility_factor = ma_data.get('volatility', 1.0)
        ma_threshold = 0.05 * (1 + volatility_factor)  # Dynamic threshold
        
        # Moving Average Signals
        if ma_data['price_to_75ma'] > (1 + ma_threshold):
            signals.append(("bullish", "Trading above 75-day MA"))
        elif ma_data['price_to_75ma'] < (1 - ma_threshold):
            signals.append(("bearish", "Trading below 75-day MA"))
        
        if ma_data['price_to_200ma'] > (1 + ma_threshold):
            signals.append(("bullish", "Trading above 200-day MA"))
        elif ma_data['price_to_200ma'] < (1 - ma_threshold):
            signals.append(("bearish", "Trading below 200-day MA"))
        
        # CUSTOMIZATION: Price Target Analysis
        # Add analyst credibility weights
        upside = target_data.get('upside_potential', 0)
        target_confidence = target_data.get('analyst_confidence', 0.5)
        
        if upside > 20 and target_confidence > 0.6:
            signals.append(("bullish", f"Strong upside potential: {upside}%"))
        elif upside < -10 and target_confidence > 0.6:
            signals.append(("bearish", f"Significant downside risk: {upside}%"))
        
        # Count and weight signals
        bullish = sum(1 for signal, _ in signals if signal == "bullish")
        bearish = sum(1 for signal, _ in signals if signal == "bearish")
        
        # Generate recommendation with reasoning
        if bullish > bearish:
            reasons = [reason for _, reason in signals if _ == "bullish"]
            return f"BUY - {'; '.join(reasons)}"
        elif bearish > bullish:
            reasons = [reason for _, reason in signals if _ == "bearish"]
            return f"SELL - {'; '.join(reasons)}"
        else:
            return "HOLD - Mixed technical signals"

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