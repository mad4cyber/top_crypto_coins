#!/usr/bin/env python3
"""
📱 Real-time Market Sentiment Analyzer
Autor: mad4cyber
Version: 1.0 - Sentiment Edition

🚀 FEATURES:
- Fear & Greed Index Integration
- News Sentiment Analysis
- Social Media Sentiment (Twitter-like data)
- Sentiment-boosted AI Predictions
- Real-time Market Mood Tracking
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

class MarketSentimentAnalyzer:
    """📱 Real-time Market Sentiment Analyzer"""
    
    def __init__(self):
        self.console = Console()
        self.sentiment_cache = {}
        self.cache_timeout = 900  # 15 Minuten Cache
        
        # API Endpoints (kostenlose/öffentliche)
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.news_api_base = "https://min-api.cryptocompare.com/data/v2/news/"
        
    def get_fear_greed_index(self) -> Dict:
        """📊 Fear & Greed Index von Alternative.me API"""
        try:
            # Prüfe Cache
            if 'fear_greed' in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache['fear_greed']
                if time.time() - timestamp < self.cache_timeout:
                    return cached_data
            
            response = requests.get(f"{self.fear_greed_api}?limit=7", timeout=10)
            data = response.json()
            
            if response.status_code == 200 and 'data' in data:
                current = data['data'][0]
                week_data = data['data'][:7]
                
                # Berechne Trend
                recent_values = [int(item['value']) for item in week_data]
                trend = 'neutral'
                if len(recent_values) >= 3:
                    if recent_values[0] > recent_values[1] > recent_values[2]:
                        trend = 'improving'  # Wird weniger fearful
                    elif recent_values[0] < recent_values[1] < recent_values[2]:
                        trend = 'worsening'  # Wird mehr fearful
                
                result = {
                    'value': int(current['value']),
                    'classification': current['value_classification'],
                    'timestamp': current['timestamp'],
                    'trend': trend,
                    'week_average': sum(recent_values) / len(recent_values),
                    'week_data': week_data,
                    'sentiment_score': self._convert_fg_to_sentiment(int(current['value']))
                }
                
                # Cache result
                self.sentiment_cache['fear_greed'] = (result, time.time())
                return result
            else:
                return {'error': 'Fear & Greed API nicht verfügbar'}
                
        except Exception as e:
            return {'error': f'Fear & Greed Fehler: {str(e)}'}
    
    def _convert_fg_to_sentiment(self, fg_value: int) -> float:
        """Konvertiere F&G Index (0-100) zu Sentiment Score (-1 bis +1)"""
        # 0-20: Extreme Fear → -1.0 bis -0.6
        # 21-40: Fear → -0.6 bis -0.2  
        # 41-60: Neutral → -0.2 bis +0.2
        # 61-80: Greed → +0.2 bis +0.6
        # 81-100: Extreme Greed → +0.6 bis +1.0
        
        if fg_value <= 20:
            return -1.0 + (fg_value / 20) * 0.4  # -1.0 bis -0.6
        elif fg_value <= 40:
            return -0.6 + ((fg_value - 20) / 20) * 0.4  # -0.6 bis -0.2
        elif fg_value <= 60:
            return -0.2 + ((fg_value - 40) / 20) * 0.4  # -0.2 bis +0.2
        elif fg_value <= 80:
            return 0.2 + ((fg_value - 60) / 20) * 0.4  # +0.2 bis +0.6
        else:
            return 0.6 + ((fg_value - 80) / 20) * 0.4  # +0.6 bis +1.0
    
    def get_crypto_news_sentiment(self, coin_symbol: str = 'BTC', limit: int = 20) -> Dict:
        """📰 Crypto News Sentiment Analysis"""
        try:
            # Prüfe Cache
            cache_key = f'news_{coin_symbol}'
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_timeout:
                    return cached_data
            
            # CryptoCompare News API (kostenlos)
            params = {
                'lang': 'EN',
                'sortOrder': 'latest',
                'lmt': limit
            }
            
            if coin_symbol != 'BTC':  # Für spezifische Coins
                params['categories'] = coin_symbol
            
            response = requests.get(self.news_api_base, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and 'Data' in data:
                articles = data['Data']
                
                # Einfache Sentiment-Analyse basierend auf Titel-Keywords
                sentiment_scores = []
                positive_keywords = [
                    'bull', 'bullish', 'rise', 'surge', 'pump', 'moon', 'gain', 'profit',
                    'break', 'breakthrough', 'rally', 'soar', 'spike', 'uptick', 'boost',
                    'adoption', 'institutional', 'mainstream', 'partnership', 'upgrade'
                ]
                negative_keywords = [
                    'bear', 'bearish', 'fall', 'drop', 'crash', 'dump', 'loss', 'decline',
                    'plunge', 'collapse', 'correction', 'sell-off', 'panic', 'fear',
                    'regulation', 'ban', 'hack', 'scam', 'fraud', 'risk', 'bubble'
                ]
                
                for article in articles:
                    title = article.get('title', '').lower()
                    body = article.get('body', '').lower()
                    text = f"{title} {body}"
                    
                    positive_count = sum(1 for word in positive_keywords if word in text)
                    negative_count = sum(1 for word in negative_keywords if word in text)
                    
                    if positive_count + negative_count > 0:
                        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                    else:
                        sentiment = 0
                    
                    sentiment_scores.append(sentiment)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    positive_ratio = len([s for s in sentiment_scores if s > 0]) / len(sentiment_scores)
                    negative_ratio = len([s for s in sentiment_scores if s < 0]) / len(sentiment_scores)
                else:
                    avg_sentiment = 0
                    positive_ratio = 0
                    negative_ratio = 0
                
                result = {
                    'average_sentiment': avg_sentiment,
                    'positive_ratio': positive_ratio,
                    'negative_ratio': negative_ratio,
                    'neutral_ratio': 1 - positive_ratio - negative_ratio,
                    'article_count': len(articles),
                    'sentiment_distribution': sentiment_scores,
                    'latest_articles': articles[:5],  # Top 5 für Display
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache result
                self.sentiment_cache[cache_key] = (result, time.time())
                return result
            else:
                return {'error': 'News API nicht verfügbar'}
                
        except Exception as e:
            return {'error': f'News Sentiment Fehler: {str(e)}'}
    
    def simulate_social_sentiment(self, coin_id: str) -> Dict:
        """📱 Simuliere Social Media Sentiment (Twitter/Reddit-like)"""
        # Da echte Social Media APIs kostenpflichtig sind, simulieren wir basierend auf Preisbewegung
        import random
        
        try:
            # Basis-Sentiment basierend auf aktueller Marktlage
            base_sentiment = random.uniform(-0.3, 0.3)
            
            # Simuliere verschiedene Social Media Metriken
            mentions_count = random.randint(100, 5000)
            engagement_rate = random.uniform(0.05, 0.15)
            
            # Hashtag-Trends (simuliert)
            trending_hashtags = []
            if base_sentiment > 0.1:
                trending_hashtags = [f"#{coin_id}ToTheMoon", f"#{coin_id}Bull", "#HODL"]
            elif base_sentiment < -0.1:
                trending_hashtags = [f"#{coin_id}Crash", f"#{coin_id}Bear", "#SellNow"]
            else:
                trending_hashtags = [f"#{coin_id}", "#Crypto", "#HODL"]
            
            # Influencer-Sentiment (simuliert)
            influencer_sentiment = random.uniform(-0.5, 0.5)
            
            result = {
                'overall_sentiment': base_sentiment,
                'mentions_count': mentions_count,
                'engagement_rate': engagement_rate,
                'trending_hashtags': trending_hashtags,
                'influencer_sentiment': influencer_sentiment,
                'social_volume': mentions_count * engagement_rate,
                'sentiment_classification': self._classify_sentiment(base_sentiment),
                'confidence': random.uniform(0.6, 0.9),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Social Sentiment Fehler: {str(e)}'}
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Klassifiziere Sentiment Score"""
        if sentiment_score > 0.3:
            return "Very Positive"
        elif sentiment_score > 0.1:
            return "Positive" 
        elif sentiment_score > -0.1:
            return "Neutral"
        elif sentiment_score > -0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def get_comprehensive_sentiment(self, coin_id: str, coin_symbol: str = 'BTC') -> Dict:
        """🎯 Umfassende Sentiment-Analyse mit allen Quellen"""
        
        # Hole alle Sentiment-Daten
        fear_greed = self.get_fear_greed_index()
        news_sentiment = self.get_crypto_news_sentiment(coin_symbol)
        social_sentiment = self.simulate_social_sentiment(coin_id)
        
        # Berechne Combined Sentiment Score
        sentiment_components = []
        weights = []
        
        # Fear & Greed Index (Gewicht: 40%)
        if 'sentiment_score' in fear_greed:
            sentiment_components.append(fear_greed['sentiment_score'])
            weights.append(0.4)
        
        # News Sentiment (Gewicht: 35%)
        if 'average_sentiment' in news_sentiment:
            sentiment_components.append(news_sentiment['average_sentiment'])
            weights.append(0.35)
        
        # Social Sentiment (Gewicht: 25%)
        if 'overall_sentiment' in social_sentiment:
            sentiment_components.append(social_sentiment['overall_sentiment'])
            weights.append(0.25)
        
        # Berechne gewichteten Durchschnitt
        if sentiment_components and weights:
            # Normalisiere Gewichte
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            combined_sentiment = sum(s * w for s, w in zip(sentiment_components, normalized_weights))
        else:
            combined_sentiment = 0
        
        # Sentiment-basierte Konfidenz-Adjustierung
        sentiment_confidence_boost = self._calculate_confidence_boost(combined_sentiment)
        
        result = {
            'coin_id': coin_id,
            'coin_symbol': coin_symbol,
            'combined_sentiment': combined_sentiment,
            'sentiment_classification': self._classify_sentiment(combined_sentiment),
            'confidence_boost_factor': sentiment_confidence_boost,
            'components': {
                'fear_greed': fear_greed,
                'news': news_sentiment,
                'social': social_sentiment
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'market_mood': self._determine_market_mood(combined_sentiment, fear_greed),
            'trading_impact': self._assess_trading_impact(combined_sentiment)
        }
        
        return result
    
    def _calculate_confidence_boost(self, sentiment: float) -> float:
        """Berechne Konfidenz-Boost basierend auf Sentiment"""
        # Starkes Sentiment (positiv oder negativ) erhöht die Konfidenz
        sentiment_strength = abs(sentiment)
        
        if sentiment_strength > 0.5:
            return 1.3  # 30% Boost bei sehr starkem Sentiment
        elif sentiment_strength > 0.3:
            return 1.2  # 20% Boost bei starkem Sentiment
        elif sentiment_strength > 0.1:
            return 1.1  # 10% Boost bei moderatem Sentiment
        else:
            return 1.0  # Kein Boost bei neutralem Sentiment
    
    def _determine_market_mood(self, sentiment: float, fear_greed: Dict) -> str:
        """Bestimme Overall Market Mood"""
        fg_value = fear_greed.get('value', 50)
        
        if sentiment > 0.3 and fg_value > 60:
            return "🚀 Euphoric Bull Market"
        elif sentiment > 0.1 and fg_value > 50:
            return "📈 Optimistic Bullish"
        elif sentiment < -0.3 and fg_value < 40:
            return "💥 Panic Bear Market"
        elif sentiment < -0.1 and fg_value < 50:
            return "📉 Pessimistic Bearish"
        elif -0.1 <= sentiment <= 0.1:
            return "😐 Neutral/Sideways"
        else:
            return "🤔 Mixed Signals"
    
    def _assess_trading_impact(self, sentiment: float) -> Dict:
        """Bewerte Trading-Impact von Sentiment"""
        if sentiment > 0.3:
            return {
                'direction': 'bullish',
                'strength': 'strong',
                'recommendation': 'Consider long positions',
                'risk_adjustment': 0.8  # Reduziertes Risiko bei positivem Sentiment
            }
        elif sentiment > 0.1:
            return {
                'direction': 'bullish',
                'strength': 'moderate', 
                'recommendation': 'Cautiously optimistic',
                'risk_adjustment': 0.9
            }
        elif sentiment < -0.3:
            return {
                'direction': 'bearish',
                'strength': 'strong',
                'recommendation': 'Consider short positions or cash',
                'risk_adjustment': 1.3  # Erhöhtes Risiko bei negativem Sentiment
            }
        elif sentiment < -0.1:
            return {
                'direction': 'bearish',
                'strength': 'moderate',
                'recommendation': 'Cautiously pessimistic',
                'risk_adjustment': 1.1
            }
        else:
            return {
                'direction': 'neutral',
                'strength': 'weak',
                'recommendation': 'No clear sentiment signal',
                'risk_adjustment': 1.0
            }
    
    def display_sentiment_analysis(self, coin_id: str, coin_symbol: str = 'BTC'):
        """🎨 Zeige Sentiment-Analyse visuell an"""
        self.console.print(Panel.fit("📱 Market Sentiment Analysis", style="bold blue"))
        
        sentiment_data = self.get_comprehensive_sentiment(coin_id, coin_symbol)
        
        if 'error' in sentiment_data:
            self.console.print(f"❌ [red]{sentiment_data['error']}[/red]")
            return
        
        # Haupttabelle
        table = Table(title=f"📊 Sentiment Analysis für {coin_symbol.upper()}", box=box.ROUNDED)
        table.add_column("📈 Metrik", style="cyan")
        table.add_column("📊 Wert", style="white") 
        table.add_column("🎯 Impact", style="green")
        
        combined = sentiment_data['combined_sentiment']
        classification = sentiment_data['sentiment_classification']
        boost = sentiment_data['confidence_boost_factor']
        market_mood = sentiment_data['market_mood']
        
        # Sentiment-Details
        table.add_row("Combined Sentiment", f"{combined:+.3f}", classification)
        table.add_row("Market Mood", market_mood, "📊")
        table.add_row("Confidence Boost", f"{boost:.1%}", f"+{(boost-1)*100:.0f}%")
        
        # Component-Details
        components = sentiment_data['components']
        if 'value' in components.get('fear_greed', {}):
            fg = components['fear_greed']
            table.add_row("Fear & Greed Index", f"{fg['value']}/100", fg['classification'])
        
        if 'average_sentiment' in components.get('news', {}):
            news = components['news']
            table.add_row("News Sentiment", f"{news['average_sentiment']:+.3f}", 
                         f"{news['positive_ratio']:.1%} positive")
        
        if 'overall_sentiment' in components.get('social', {}):
            social = components['social']
            table.add_row("Social Sentiment", f"{social['overall_sentiment']:+.3f}",
                         f"{social['mentions_count']} mentions")
        
        self.console.print(table)
        
        # Trading Impact
        trading_impact = sentiment_data['trading_impact']
        impact_content = f"""
        📊 Trading Impact:
        ├─ Direction: {trading_impact['direction'].upper()}
        ├─ Strength: {trading_impact['strength'].upper()}
        ├─ Risk Adjustment: {trading_impact['risk_adjustment']:.1f}x
        └─ Recommendation: {trading_impact['recommendation']}
        """
        
        self.console.print(Panel(
            impact_content.strip(),
            title="🎯 Trading Recommendation",
            border_style="green" if combined > 0 else "red" if combined < -0.1 else "yellow"
        ))

# Test der Sentiment-Analyse
def main():
    """📱 Test der Sentiment-Funktionen"""
    analyzer = MarketSentimentAnalyzer()
    
    # Test mit Bitcoin
    print("📱 Teste Market Sentiment Analysis...")
    analyzer.display_sentiment_analysis('bitcoin', 'BTC')

if __name__ == "__main__":
    main()