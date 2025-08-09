#!/usr/bin/env python3
"""
📊 Real-time Market Sentiment Integration Engine
Autor: mad4cyber
Version: 1.0 - Real-time Sentiment Edition

🚀 FEATURES:
- Enhanced Social Media Sentiment Analysis (Twitter, Reddit, Discord)
- Live News Impact Scoring & Event Detection
- Whale Movement & On-Chain Analytics
- Real-time Fear & Greed Index Integration
- Multi-Source Sentiment Aggregation
- Event-Driven Sentiment Alerts
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
import asyncio
import os
import warnings
warnings.filterwarnings('ignore')

# Text analysis libraries
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_SENTIMENT_LIBS = True
except ImportError:
    HAS_SENTIMENT_LIBS = False
    print("⚠️ Sentiment analysis libraries not installed - using simplified analysis")

# Web scraping for news
try:
    from bs4 import BeautifulSoup
    HAS_SCRAPING = True
except ImportError:
    HAS_SCRAPING = False
    print("⚠️ Web scraping libraries not available")

@dataclass
class SentimentData:
    """📊 Unified Sentiment Data Structure"""
    timestamp: str
    source: str  # 'twitter', 'reddit', 'news', 'onchain', 'fear_greed'
    coin_symbol: str
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1
    volume: int  # Number of mentions/posts
    impact_score: float  # Expected market impact 0-1
    raw_data: Dict[str, Any]
    keywords: List[str]
    influence_level: str  # 'low', 'medium', 'high', 'viral'

@dataclass
class SentimentEvent:
    """🚨 Significant Sentiment Event"""
    timestamp: str
    event_type: str  # 'whale_movement', 'news_spike', 'social_viral', 'fear_extreme'
    coin_symbol: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    sentiment_change: float  # Change in sentiment
    expected_impact: float  # Expected price impact
    confidence: float
    sources: List[str]
    duration_estimate: int  # Minutes

@dataclass
class OnChainMetrics:
    """🔗 On-Chain Analytics Data"""
    timestamp: str
    coin_symbol: str
    large_transactions: int  # Number of whale transactions
    exchange_flows: Dict[str, float]  # Net flow to/from exchanges
    holder_distribution: Dict[str, float]  # Distribution changes
    network_activity: float  # Transaction count/volume
    fear_greed_onchain: float  # On-chain fear & greed

class RealtimeSentimentEngine:
    """📊 Real-time Market Sentiment Integration Engine"""
    
    def __init__(self, data_file: str = "realtime_sentiment_data.json"):
        self.data_file = data_file
        
        # Sentiment analyzers
        if HAS_SENTIMENT_LIBS:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Data storage
        self.sentiment_data: List[SentimentData] = []
        self.sentiment_events: List[SentimentEvent] = []
        self.onchain_metrics: List[OnChainMetrics] = []
        
        # Configuration
        self.update_interval = 300  # 5 minutes
        self.max_history_hours = 72  # 3 days
        self.viral_threshold = 1000  # Mentions for viral classification
        self.whale_threshold = 1000000  # USD threshold for whale transactions
        
        # API endpoints (would use real APIs in production)
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.news_sources = [
            "https://cointelegraph.com/",
            "https://coindesk.com/",
            "https://decrypt.co/"
        ]
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Load existing data
        self.load_sentiment_data()
    
    def load_sentiment_data(self):
        """📁 Load historical sentiment data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load sentiment data
                for item in data.get('sentiment_data', []):
                    sentiment = SentimentData(**item)
                    self.sentiment_data.append(sentiment)
                
                # Load sentiment events
                for item in data.get('sentiment_events', []):
                    event = SentimentEvent(**item)
                    self.sentiment_events.append(event)
                
                # Load on-chain metrics
                for item in data.get('onchain_metrics', []):
                    metrics = OnChainMetrics(**item)
                    self.onchain_metrics.append(metrics)
                    
            except Exception as e:
                print(f"❌ Error loading sentiment data: {e}")
    
    def save_sentiment_data(self):
        """💾 Save sentiment data"""
        try:
            data = {
                'sentiment_data': [asdict(item) for item in self.sentiment_data[-1000:]],  # Keep last 1000
                'sentiment_events': [asdict(item) for item in self.sentiment_events[-200:]],  # Keep last 200
                'onchain_metrics': [asdict(item) for item in self.onchain_metrics[-500:]],  # Keep last 500
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving sentiment data: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """📝 Analyze sentiment from text"""
        if not HAS_SENTIMENT_LIBS:
            # Simple keyword-based analysis
            positive_words = ['bull', 'bullish', 'moon', 'pump', 'buy', 'hodl', 'rocket', 'gain', 'profit']
            negative_words = ['bear', 'bearish', 'dump', 'sell', 'crash', 'dip', 'loss', 'fear', 'panic']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count + negative_count == 0:
                return 0.0, 0.3  # Neutral with low confidence
            
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            confidence = min(1.0, (positive_count + negative_count) / 5)
            return sentiment, confidence
        
        try:
            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Combine both methods
            combined_sentiment = (vader_scores['compound'] + textblob_sentiment) / 2
            confidence = abs(vader_scores['compound']) * 0.7 + abs(textblob_sentiment) * 0.3
            
            return combined_sentiment, confidence
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            return 0.0, 0.1
    
    def fetch_social_sentiment(self, coin_symbol: str) -> List[SentimentData]:
        """📱 Fetch social media sentiment (simulated)"""
        print(f"📱 Fetching social sentiment for {coin_symbol}...")
        
        try:
            # Simulated social media data (in production, would use real APIs)
            social_posts = [
                f"{coin_symbol} is looking bullish! Great momentum building 🚀",
                f"Might be time to take profits on {coin_symbol}, seems overbought",
                f"HODL {coin_symbol}! Long term this is going to moon 🌙",
                f"{coin_symbol} chart looks bearish, expecting a pullback",
                f"Just bought more {coin_symbol}! Diamond hands 💎🙌",
                f"Selling my {coin_symbol} bag, too much volatility for me",
                f"{coin_symbol} breaking resistance! Next stop $100k",
                f"FUD around {coin_symbol} is getting intense, but I'm staying strong"
            ]
            
            sentiment_data = []
            current_time = datetime.now()
            
            for i, post in enumerate(social_posts):
                # Simulate different sources and timestamps
                sources = ['twitter', 'reddit', 'discord', 'telegram']
                source = sources[i % len(sources)]
                
                # Analyze sentiment
                sentiment_score, confidence = self.analyze_text_sentiment(post)
                
                # Simulate volume and influence
                volume = np.random.randint(50, 2000)
                influence = 'high' if volume > 1500 else 'medium' if volume > 800 else 'low'
                
                # Calculate impact score based on volume and sentiment strength
                impact_score = min(1.0, (volume / 1000) * abs(sentiment_score) * confidence)
                
                sentiment_item = SentimentData(
                    timestamp=(current_time - timedelta(minutes=i*30)).isoformat(),
                    source=source,
                    coin_symbol=coin_symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    volume=volume,
                    impact_score=impact_score,
                    raw_data={'post': post, 'engagement': volume},
                    keywords=self.extract_keywords(post),
                    influence_level=influence
                )
                
                sentiment_data.append(sentiment_item)
            
            return sentiment_data
            
        except Exception as e:
            print(f"❌ Error fetching social sentiment: {e}")
            return []
    
    def extract_keywords(self, text: str) -> List[str]:
        """🔍 Extract relevant keywords from text"""
        # Simple keyword extraction
        crypto_keywords = ['bull', 'bear', 'moon', 'hodl', 'pump', 'dump', 'dip', 'rocket', 'diamond', 'hands']
        text_lower = text.lower()
        
        found_keywords = [keyword for keyword in crypto_keywords if keyword in text_lower]
        return found_keywords[:5]  # Return max 5 keywords
    
    def fetch_news_sentiment(self, coin_symbol: str) -> List[SentimentData]:
        """📰 Fetch news sentiment"""
        print(f"📰 Fetching news sentiment for {coin_symbol}...")
        
        try:
            # Simulated news headlines (in production, would scrape real news)
            news_headlines = [
                f"{coin_symbol} Institutional Adoption Surges as Major Banks Enter Market",
                f"Regulatory Uncertainty Clouds {coin_symbol} Short-term Outlook",
                f"Technical Analysis: {coin_symbol} Forms Bullish Continuation Pattern",
                f"Market Correction: {coin_symbol} Tests Key Support Levels",
                f"Breaking: Major {coin_symbol} Whale Transfers $100M to Exchange",
                f"{coin_symbol} Network Upgrade Promises Enhanced Scalability"
            ]
            
            sentiment_data = []
            current_time = datetime.now()
            
            for i, headline in enumerate(news_headlines):
                sentiment_score, confidence = self.analyze_text_sentiment(headline)
                
                # News has higher impact than social media
                impact_multiplier = 1.5
                impact_score = min(1.0, abs(sentiment_score) * confidence * impact_multiplier)
                
                # Simulate source credibility
                sources = ['CoinTelegraph', 'CoinDesk', 'Decrypt', 'BlockWorks']
                source = sources[i % len(sources)]
                
                sentiment_item = SentimentData(
                    timestamp=(current_time - timedelta(hours=i)).isoformat(),
                    source=f"news_{source.lower()}",
                    coin_symbol=coin_symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    volume=1,  # Single news article
                    impact_score=impact_score,
                    raw_data={'headline': headline, 'source': source},
                    keywords=self.extract_keywords(headline),
                    influence_level='high'  # News generally has high influence
                )
                
                sentiment_data.append(sentiment_item)
            
            return sentiment_data
            
        except Exception as e:
            print(f"❌ Error fetching news sentiment: {e}")
            return []
    
    def fetch_fear_greed_index(self) -> float:
        """😰 Fetch Fear & Greed Index"""
        try:
            # Simulate Fear & Greed Index (in production, would use real API)
            # Index ranges from 0 (Extreme Fear) to 100 (Extreme Greed)
            
            # Simulate based on current time to add variability
            base_value = 50
            time_factor = (datetime.now().hour - 12) * 2  # -24 to +24
            random_factor = np.random.randint(-15, 15)
            
            fear_greed_value = max(0, min(100, base_value + time_factor + random_factor))
            
            print(f"😰 Fear & Greed Index: {fear_greed_value}")
            return fear_greed_value
            
        except Exception as e:
            print(f"❌ Error fetching Fear & Greed Index: {e}")
            return 50  # Neutral default
    
    def simulate_onchain_metrics(self, coin_symbol: str) -> OnChainMetrics:
        """🔗 Simulate on-chain analytics"""
        print(f"🔗 Analyzing on-chain metrics for {coin_symbol}...")
        
        try:
            # Simulate whale transactions
            large_transactions = np.random.randint(0, 20)
            
            # Simulate exchange flows (positive = inflow, negative = outflow)
            exchange_flows = {
                'binance': np.random.uniform(-1000, 1000),
                'coinbase': np.random.uniform(-500, 500),
                'kraken': np.random.uniform(-300, 300)
            }
            
            # Simulate holder distribution changes
            holder_distribution = {
                'whales_pct': np.random.uniform(0.6, 0.8),  # 60-80% held by whales
                'retail_pct': np.random.uniform(0.15, 0.35),  # 15-35% retail
                'institutions_pct': np.random.uniform(0.05, 0.15)  # 5-15% institutions
            }
            
            # Network activity simulation
            network_activity = np.random.uniform(0.3, 1.0)
            
            # On-chain fear & greed
            fear_greed_onchain = np.random.uniform(0.2, 0.8)
            
            return OnChainMetrics(
                timestamp=datetime.now().isoformat(),
                coin_symbol=coin_symbol,
                large_transactions=large_transactions,
                exchange_flows=exchange_flows,
                holder_distribution=holder_distribution,
                network_activity=network_activity,
                fear_greed_onchain=fear_greed_onchain
            )
            
        except Exception as e:
            print(f"❌ Error simulating on-chain metrics: {e}")
            return OnChainMetrics(
                timestamp=datetime.now().isoformat(),
                coin_symbol=coin_symbol,
                large_transactions=0,
                exchange_flows={},
                holder_distribution={},
                network_activity=0.5,
                fear_greed_onchain=0.5
            )
    
    def detect_sentiment_events(self, coin_symbol: str) -> List[SentimentEvent]:
        """🚨 Detect significant sentiment events"""
        events = []
        current_time = datetime.now()
        
        try:
            # Get recent sentiment data for this coin
            recent_data = [
                s for s in self.sentiment_data 
                if s.coin_symbol == coin_symbol and 
                datetime.fromisoformat(s.timestamp.replace('Z', '')) >= current_time - timedelta(hours=1)
            ]
            
            if not recent_data:
                return events
            
            # Check for viral social activity
            total_volume = sum(s.volume for s in recent_data if 'social' in s.source or s.source in ['twitter', 'reddit'])
            if total_volume > self.viral_threshold:
                avg_sentiment = np.mean([s.sentiment_score for s in recent_data])
                events.append(SentimentEvent(
                    timestamp=current_time.isoformat(),
                    event_type='social_viral',
                    coin_symbol=coin_symbol,
                    severity='high' if total_volume > self.viral_threshold * 2 else 'medium',
                    description=f"Viral social activity detected: {total_volume} mentions with {avg_sentiment:.2f} sentiment",
                    sentiment_change=avg_sentiment,
                    expected_impact=min(0.15, total_volume / 10000),  # Max 15% expected impact
                    confidence=0.7,
                    sources=['twitter', 'reddit', 'discord'],
                    duration_estimate=120  # 2 hours estimated duration
                ))
            
            # Check for news spikes
            news_data = [s for s in recent_data if 'news' in s.source]
            if len(news_data) >= 3:  # Multiple news articles
                avg_impact = np.mean([s.impact_score for s in news_data])
                if avg_impact > 0.6:
                    events.append(SentimentEvent(
                        timestamp=current_time.isoformat(),
                        event_type='news_spike',
                        coin_symbol=coin_symbol,
                        severity='high',
                        description=f"Major news spike detected with {avg_impact:.2f} impact score",
                        sentiment_change=np.mean([s.sentiment_score for s in news_data]),
                        expected_impact=avg_impact * 0.1,  # 10% of impact score
                        confidence=0.8,
                        sources=[s.source for s in news_data],
                        duration_estimate=360  # 6 hours
                    ))
            
            # Simulate whale movement detection
            if np.random.random() < 0.1:  # 10% chance
                events.append(SentimentEvent(
                    timestamp=current_time.isoformat(),
                    event_type='whale_movement',
                    coin_symbol=coin_symbol,
                    severity='medium',
                    description=f"Large whale transaction detected: ${np.random.randint(5, 50)}M moved",
                    sentiment_change=np.random.uniform(-0.3, 0.3),
                    expected_impact=np.random.uniform(0.02, 0.08),  # 2-8% impact
                    confidence=0.9,
                    sources=['onchain'],
                    duration_estimate=60  # 1 hour
                ))
            
            return events
            
        except Exception as e:
            print(f"❌ Error detecting sentiment events: {e}")
            return []
    
    def aggregate_sentiment_score(self, coin_symbol: str, hours_back: int = 24) -> Dict[str, float]:
        """📊 Aggregate sentiment score from multiple sources"""
        print(f"📊 Aggregating sentiment for {coin_symbol} ({hours_back}h)...")
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter recent data
            recent_data = [
                s for s in self.sentiment_data
                if s.coin_symbol == coin_symbol and 
                datetime.fromisoformat(s.timestamp.replace('Z', '')) >= cutoff_time
            ]
            
            if not recent_data:
                return {
                    'overall_sentiment': 0.0,
                    'confidence': 0.3,
                    'social_sentiment': 0.0,
                    'news_sentiment': 0.0,
                    'volume_weighted': 0.0,
                    'impact_weighted': 0.0,
                    'trend_direction': 'neutral'
                }
            
            # Separate by source type
            social_data = [s for s in recent_data if s.source in ['twitter', 'reddit', 'discord', 'telegram']]
            news_data = [s for s in recent_data if 'news' in s.source]
            
            # Calculate source-specific sentiments
            social_sentiment = np.mean([s.sentiment_score for s in social_data]) if social_data else 0.0
            news_sentiment = np.mean([s.sentiment_score for s in news_data]) if news_data else 0.0
            
            # Volume-weighted sentiment
            total_volume = sum(s.volume for s in recent_data)
            volume_weighted = sum(s.sentiment_score * s.volume for s in recent_data) / total_volume if total_volume > 0 else 0.0
            
            # Impact-weighted sentiment
            total_impact = sum(s.impact_score for s in recent_data)
            impact_weighted = sum(s.sentiment_score * s.impact_score for s in recent_data) / total_impact if total_impact > 0 else 0.0
            
            # Overall sentiment (weighted combination)
            weights = {
                'social': 0.4,
                'news': 0.3,
                'volume': 0.2,
                'impact': 0.1
            }
            
            overall_sentiment = (
                social_sentiment * weights['social'] +
                news_sentiment * weights['news'] +
                volume_weighted * weights['volume'] +
                impact_weighted * weights['impact']
            )
            
            # Aggregate confidence
            avg_confidence = np.mean([s.confidence for s in recent_data])
            
            # Trend direction
            if overall_sentiment > 0.2:
                trend = 'bullish'
            elif overall_sentiment < -0.2:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            result = {
                'overall_sentiment': overall_sentiment,
                'confidence': avg_confidence,
                'social_sentiment': social_sentiment,
                'news_sentiment': news_sentiment,
                'volume_weighted': volume_weighted,
                'impact_weighted': impact_weighted,
                'trend_direction': trend,
                'data_points': len(recent_data),
                'total_volume': total_volume
            }
            
            print(f"✅ Sentiment aggregated: {overall_sentiment:.3f} ({trend})")
            return result
            
        except Exception as e:
            print(f"❌ Error aggregating sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.1,
                'social_sentiment': 0.0,
                'news_sentiment': 0.0,
                'volume_weighted': 0.0,
                'impact_weighted': 0.0,
                'trend_direction': 'neutral'
            }
    
    def run_realtime_analysis(self, coin_symbols: List[str] = None) -> Dict[str, Any]:
        """🔄 Run comprehensive real-time sentiment analysis"""
        if coin_symbols is None:
            coin_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
        
        print(f"🔄 Running real-time sentiment analysis for {len(coin_symbols)} coins...")
        
        results = {}
        
        for coin_symbol in coin_symbols:
            try:
                print(f"\n📊 Analyzing {coin_symbol}...")
                
                # Fetch all sentiment data
                social_data = self.fetch_social_sentiment(coin_symbol)
                news_data = self.fetch_news_sentiment(coin_symbol)
                
                # Add to storage
                self.sentiment_data.extend(social_data + news_data)
                
                # Fetch Fear & Greed Index (global)
                fear_greed = self.fetch_fear_greed_index()
                
                # Get on-chain metrics
                onchain = self.simulate_onchain_metrics(coin_symbol)
                self.onchain_metrics.append(onchain)
                
                # Detect events
                events = self.detect_sentiment_events(coin_symbol)
                self.sentiment_events.extend(events)
                
                # Aggregate sentiment
                aggregated = self.aggregate_sentiment_score(coin_symbol, hours_back=24)
                
                # Compile results
                results[coin_symbol] = {
                    'aggregated_sentiment': aggregated,
                    'fear_greed_index': fear_greed,
                    'onchain_metrics': asdict(onchain),
                    'recent_events': [asdict(event) for event in events],
                    'social_mentions': len(social_data),
                    'news_articles': len(news_data),
                    'sentiment_strength': abs(aggregated['overall_sentiment']),
                    'market_mood': self.classify_market_mood(aggregated['overall_sentiment'], fear_greed)
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {coin_symbol}: {e}")
                results[coin_symbol] = {'error': str(e)}
        
        # Clean up old data
        self.cleanup_old_data()
        
        # Save data
        self.save_sentiment_data()
        
        print(f"✅ Real-time sentiment analysis completed for {len(results)} coins")
        return results
    
    def classify_market_mood(self, sentiment: float, fear_greed: float) -> str:
        """🌡️ Classify overall market mood"""
        # Combine sentiment and fear & greed
        combined_score = (sentiment + (fear_greed - 50) / 50) / 2
        
        if combined_score > 0.5:
            return "Extreme Greed"
        elif combined_score > 0.2:
            return "Greed"
        elif combined_score > -0.2:
            return "Neutral"
        elif combined_score > -0.5:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def cleanup_old_data(self):
        """🧹 Clean up old data to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
        
        # Filter sentiment data
        self.sentiment_data = [
            s for s in self.sentiment_data
            if datetime.fromisoformat(s.timestamp.replace('Z', '')) >= cutoff_time
        ]
        
        # Filter events
        self.sentiment_events = [
            e for e in self.sentiment_events
            if datetime.fromisoformat(e.timestamp.replace('Z', '')) >= cutoff_time
        ]
        
        # Filter on-chain metrics
        self.onchain_metrics = [
            m for m in self.onchain_metrics
            if datetime.fromisoformat(m.timestamp.replace('Z', '')) >= cutoff_time
        ]
    
    def get_sentiment_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """📈 Get sentiment summary for dashboard"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent data
        recent_data = [
            s for s in self.sentiment_data
            if datetime.fromisoformat(s.timestamp.replace('Z', '')) >= cutoff_time
        ]
        
        recent_events = [
            e for e in self.sentiment_events
            if datetime.fromisoformat(e.timestamp.replace('Z', '')) >= cutoff_time
        ]
        
        if not recent_data:
            return {'message': 'No recent sentiment data'}
        
        # Summary statistics
        total_mentions = sum(s.volume for s in recent_data)
        avg_sentiment = np.mean([s.sentiment_score for s in recent_data])
        avg_confidence = np.mean([s.confidence for s in recent_data])
        
        # Source breakdown
        source_counts = {}
        for s in recent_data:
            source_counts[s.source] = source_counts.get(s.source, 0) + s.volume
        
        # Event severity breakdown
        event_severity = {}
        for e in recent_events:
            event_severity[e.severity] = event_severity.get(e.severity, 0) + 1
        
        return {
            'timeframe_hours': hours_back,
            'total_data_points': len(recent_data),
            'total_mentions': total_mentions,
            'average_sentiment': avg_sentiment,
            'average_confidence': avg_confidence,
            'sentiment_trend': 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral',
            'source_breakdown': source_counts,
            'recent_events_count': len(recent_events),
            'event_severity_breakdown': event_severity,
            'top_events': [asdict(e) for e in recent_events[-5:]]  # Last 5 events
        }

# Test der Real-time Sentiment Engine
def main():
    """📊 Test des Real-time Sentiment Systems"""
    engine = RealtimeSentimentEngine()
    
    print("📊 Real-time Sentiment Engine - Demo")
    print("=" * 50)
    
    # Run analysis for sample coins
    coins = ['BTC', 'ETH']
    results = engine.run_realtime_analysis(coins)
    
    for coin, data in results.items():
        if 'error' not in data:
            print(f"\n📊 {coin} Sentiment Analysis:")
            agg = data['aggregated_sentiment']
            print(f"   Overall Sentiment: {agg['overall_sentiment']:.3f} ({agg['trend_direction']})")
            print(f"   Confidence: {agg['confidence']:.2f}")
            print(f"   Fear & Greed Index: {data['fear_greed_index']}")
            print(f"   Market Mood: {data['market_mood']}")
            print(f"   Social Mentions: {data['social_mentions']}")
            print(f"   News Articles: {data['news_articles']}")
            
            if data['recent_events']:
                print(f"   Recent Events: {len(data['recent_events'])}")
                for event in data['recent_events']:
                    print(f"     • {event['event_type']}: {event['description']}")

if __name__ == "__main__":
    main()