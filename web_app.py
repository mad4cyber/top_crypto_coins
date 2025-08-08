#!/usr/bin/env python3
"""
🌐 Crypto AI Web Interface
Autor: mad4cyber
Version: 1.0 - Web Edition

🚀 FEATURES:
- Flask Web Application
- AI-Prognosen über Browser
- Interactive Charts
- Real-time Updates
- REST API Endpoints
"""

from flask import Flask, render_template, jsonify, request
import json
import numpy as np
from datetime import datetime
import asyncio
import threading
import time
import warnings
warnings.filterwarnings('ignore')

from ai_predictor import CryptoAIPredictor
from multi_coin_ai_analysis import MultiCoinAIAnalysis
from sentiment_analyzer import MarketSentimentAnalyzer
from portfolio_manager import PortfolioManager
from alert_manager import AlertManager

# Flask App
app = Flask(__name__)
app.secret_key = 'crypto_ai_secret_key_2025'

# Global Instances
predictor = CryptoAIPredictor()
multi_analyzer = MultiCoinAIAnalysis()
sentiment_analyzer = MarketSentimentAnalyzer()
portfolio_manager = PortfolioManager()
alert_manager = AlertManager()

# Cache für Predictions (um API-Limits zu schonen)
prediction_cache = {}
cache_timeout = 300  # 5 Minuten

def convert_numpy_types(obj):
    """Konvertiere NumPy Typen zu Python Standardtypen für JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def add_confidence_analysis(prediction_result):
    """Füge Konfidenz-Analyse und Warnungen hinzu"""
    if 'error' in prediction_result:
        return prediction_result
    
    confidence = prediction_result.get('confidence', 0)
    price_change_pct = prediction_result.get('price_change_pct', 0)
    
    # Konfidenz-Bewertung
    if confidence < 0.3:
        confidence_warning = "⚠️ SEHR NIEDRIGE KONFIDENZ"
        confidence_level = "very_low"
        risk_assessment = "SEHR HOCH"
        recommendation = "❌ Prognose nicht empfohlen - Weitere Analyse nötig"
        reliability_score = 1
    elif confidence < 0.5:
        confidence_warning = "⚠️ NIEDRIGE KONFIDENZ"
        confidence_level = "low"
        risk_assessment = "HOCH"
        recommendation = "⚠️ Mit großer Vorsicht verwenden - Kleine Position"
        reliability_score = 2
    elif confidence < 0.65:
        confidence_warning = "⚡ MODERATE KONFIDENZ"
        confidence_level = "moderate"
        risk_assessment = "MITTEL"
        recommendation = "📊 Akzeptabel - Normale Position möglich"
        reliability_score = 3
    elif confidence < 0.8:
        confidence_warning = "✅ GUTE KONFIDENZ"
        confidence_level = "good"
        risk_assessment = "NIEDRIG"
        recommendation = "✅ Verlässlich - Standard Position empfohlen"
        reliability_score = 4
    else:
        confidence_warning = "🎯 SEHR HOHE KONFIDENZ"
        confidence_level = "very_high"
        risk_assessment = "SEHR NIEDRIG"
        recommendation = "🚀 Sehr verlässlich - Erhöhte Position möglich"
        reliability_score = 5
    
    # Trading-Signal basierend auf Konfidenz und Preisänderung
    signal_strength = abs(price_change_pct) * confidence
    
    if signal_strength > 5 and price_change_pct > 0:
        trading_signal = "🚀 STRONG BUY"
        signal_color = "success"
    elif signal_strength > 2 and price_change_pct > 0:
        trading_signal = "🟢 BUY"
        signal_color = "success"
    elif signal_strength > 5 and price_change_pct < 0:
        trading_signal = "🔴 STRONG SELL"
        signal_color = "danger"
    elif signal_strength > 2 and price_change_pct < 0:
        trading_signal = "🟠 SELL"
        signal_color = "warning"
    else:
        trading_signal = "⚪ HOLD"
        signal_color = "secondary"
    
    # Position Size Empfehlung (basierend auf Konfidenz)
    if confidence > 0.7:
        max_position = 10  # 10% des Portfolios
    elif confidence > 0.5:
        max_position = 5   # 5% des Portfolios
    elif confidence > 0.3:
        max_position = 2   # 2% des Portfolios
    else:
        max_position = 0   # Keine Position empfohlen
    
    # Stop-Loss und Take-Profit Levels
    if abs(price_change_pct) > 0:
        stop_loss_pct = abs(price_change_pct) * 0.5    # 50% der erwarteten Bewegung
        take_profit_pct = abs(price_change_pct) * 1.5  # 150% der erwarteten Bewegung
    else:
        stop_loss_pct = 3  # Default 3%
        take_profit_pct = 6  # Default 6%
    
    # Erweitere die Prognose um Konfidenz-Analyse
    enhanced_result = {
        **prediction_result,
        'confidence_analysis': {
            'warning': confidence_warning,
            'level': confidence_level,
            'risk_assessment': risk_assessment,
            'recommendation': recommendation,
            'reliability_score': reliability_score
        },
        'trading_analysis': {
            'signal': trading_signal,
            'signal_color': signal_color,
            'signal_strength': round(signal_strength, 2),
            'max_position_pct': max_position,
            'stop_loss_pct': round(stop_loss_pct, 2),
            'take_profit_pct': round(take_profit_pct, 2)
        }
    }
    
    return enhanced_result

def get_cached_prediction(coin_id: str):
    """Hole gecachte Prognose oder erstelle neue mit Konfidenz-Analyse und Sentiment"""
    now = time.time()
    
    if coin_id in prediction_cache:
        cached_data, timestamp = prediction_cache[coin_id]
        if now - timestamp < cache_timeout:
            return cached_data
    
    # Neue Prognose erstellen
    try:
        prediction = predictor.predict_future_prices(coin_id)
        # Konvertiere NumPy Typen für JSON-Serialisierung
        prediction = convert_numpy_types(prediction)
        
        # Füge Sentiment-Analyse hinzu
        coin_symbol = get_coin_symbol(coin_id)
        sentiment_data = sentiment_analyzer.get_comprehensive_sentiment(coin_id, coin_symbol)
        prediction = add_sentiment_boost(prediction, sentiment_data)
        
        # Füge Konfidenz-Analyse hinzu
        prediction = add_confidence_analysis(prediction)
        
        prediction_cache[coin_id] = (prediction, now)
        return prediction
    except Exception as e:
        return {'error': str(e)}

def get_coin_symbol(coin_id: str) -> str:
    """Konvertiere Coin ID zu Symbol für APIs"""
    symbol_mapping = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH', 
        'binancecoin': 'BNB',
        'ripple': 'XRP',
        'solana': 'SOL',
        'cardano': 'ADA',
        'dogecoin': 'DOGE',
        'tether': 'USDT',
        'usd-coin': 'USDC',
        'staked-ether': 'stETH',
        'tron': 'TRX',
        'shiba-inu': 'SHIB',
        'avalanche-2': 'AVAX',
        'chainlink': 'LINK',
        'polygon': 'MATIC'
    }
    return symbol_mapping.get(coin_id, coin_id.upper())

def add_sentiment_boost(prediction_result, sentiment_data):
    """Booste AI-Prognose basierend auf Sentiment-Analyse"""
    if 'error' in prediction_result or 'error' in sentiment_data:
        return prediction_result
    
    original_confidence = prediction_result.get('confidence', 0)
    boost_factor = sentiment_data.get('confidence_boost_factor', 1.0)
    
    # Booste Konfidenz (maximal auf 0.95 begrenzt)
    boosted_confidence = min(0.95, original_confidence * boost_factor)
    
    # Sentiment-Adjustierung der Prognose
    sentiment_score = sentiment_data.get('combined_sentiment', 0)
    price_change_pct = prediction_result.get('price_change_pct', 0)
    
    # Sentiment-verstärkte Prognose
    sentiment_adjustment = sentiment_score * 0.1  # Max 10% Adjustment
    adjusted_price_change = price_change_pct + (price_change_pct * sentiment_adjustment)
    
    # Berechne adjusted predicted price
    current_price = prediction_result.get('current_price', 0)
    adjusted_predicted_price = current_price * (1 + adjusted_price_change / 100)
    
    # Erweitere Prognose um Sentiment-Daten
    enhanced_result = {
        **prediction_result,
        'confidence': boosted_confidence,
        'sentiment_adjusted_price_change': adjusted_price_change,
        'sentiment_adjusted_predicted_price': adjusted_predicted_price,
        'sentiment_analysis': {
            'combined_sentiment': sentiment_data.get('combined_sentiment', 0),
            'sentiment_classification': sentiment_data.get('sentiment_classification', 'Neutral'),
            'market_mood': sentiment_data.get('market_mood', 'Unknown'),
            'confidence_boost_applied': boost_factor,
            'sentiment_adjustment': sentiment_adjustment,
            'fear_greed_index': sentiment_data.get('components', {}).get('fear_greed', {}).get('value', 50),
            'news_sentiment': sentiment_data.get('components', {}).get('news', {}).get('average_sentiment', 0),
            'social_sentiment': sentiment_data.get('components', {}).get('social', {}).get('overall_sentiment', 0)
        }
    }
    
    return enhanced_result

@app.route('/')
def index():
    """Hauptseite"""
    return render_template('index.html')

@app.route('/api/predict/<coin_id>')
def api_predict(coin_id):
    """API Endpoint für AI-Prognose"""
    prediction = get_cached_prediction(coin_id.lower())
    return jsonify(prediction)

@app.route('/api/multi-analysis')
def api_multi_analysis():
    """API Endpoint für Multi-Coin Analyse"""
    try:
        # Da asyncio in Flask komplex ist, verwenden wir eine vereinfachte Version
        results = {}
        coins = ['bitcoin', 'ethereum', 'tether', 'binancecoin', 'solana', 'ripple', 'usd-coin', 'staked-ether', 'dogecoin', 'cardano', 'tron', 'shiba-inu', 'avalanche-2', 'chainlink', 'polygon']
        
        for coin in coins:
            prediction = get_cached_prediction(coin)
            if 'error' not in prediction:
                results[coin] = prediction
        
        # Konvertiere auch Multi-Analysis Ergebnisse
        results = convert_numpy_types(results)
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/coins')
def api_coins():
    """Verfügbare Coins - Top 15 nach Marktkapitalisierung"""
    coins = [
        {'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'BTC'},
        {'id': 'ethereum', 'name': 'Ethereum', 'symbol': 'ETH'},
        {'id': 'tether', 'name': 'Tether', 'symbol': 'USDT'},
        {'id': 'binancecoin', 'name': 'Binance Coin', 'symbol': 'BNB'},
        {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL'},
        {'id': 'ripple', 'name': 'Ripple', 'symbol': 'XRP'},
        {'id': 'usd-coin', 'name': 'USD Coin', 'symbol': 'USDC'},
        {'id': 'staked-ether', 'name': 'Lido Staked Ether', 'symbol': 'stETH'},
        {'id': 'dogecoin', 'name': 'Dogecoin', 'symbol': 'DOGE'},
        {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA'},
        {'id': 'tron', 'name': 'TRON', 'symbol': 'TRX'},
        {'id': 'shiba-inu', 'name': 'Shiba Inu', 'symbol': 'SHIB'},
        {'id': 'avalanche-2', 'name': 'Avalanche', 'symbol': 'AVAX'},
        {'id': 'chainlink', 'name': 'Chainlink', 'symbol': 'LINK'},
        {'id': 'polygon', 'name': 'Polygon', 'symbol': 'MATIC'}
    ]
    return jsonify(coins)

@app.route('/dashboard')
def dashboard():
    """AI Dashboard Seite"""
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    """Multi-Coin Analyse Seite"""
    return render_template('analysis.html')

@app.route('/charts')
def charts():
    """Professional Trading Charts Seite"""
    return render_template('charts.html')

@app.route('/portfolio')
def portfolio():
    """Portfolio Manager Seite"""
    return render_template('portfolio.html')

@app.route('/alerts')
def alerts():
    """Alert Manager Seite"""
    return render_template('alerts.html')

# Portfolio API Endpoints
@app.route('/api/portfolio/<portfolio_name>')
def api_get_portfolio(portfolio_name):
    """API Endpoint für Portfolio-Daten"""
    try:
        portfolio_data = portfolio_manager.calculate_portfolio_value(portfolio_name)
        
        # Konvertiere NumPy Typen
        portfolio_data = convert_numpy_types(portfolio_data)
        
        # Hole auch Risk Assessment
        risk_data = portfolio_manager.get_portfolio_risk_metrics(portfolio_name)
        portfolio_data['risk_assessment'] = convert_numpy_types(risk_data)
        
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio/suggest/<portfolio_name>/<coin_id>')
def api_suggest_position(portfolio_name, coin_id):
    """API Endpoint für Position Size Empfehlung"""
    try:
        suggestion = portfolio_manager.suggest_position_size(portfolio_name, coin_id)
        suggestion = convert_numpy_types(suggestion)
        return jsonify(suggestion)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio/add/<portfolio_name>', methods=['POST'])
def api_add_position(portfolio_name):
    """API Endpoint zum Hinzufügen einer Position"""
    try:
        data = request.get_json()
        
        coin_id = data.get('coin_id')
        entry_price = float(data.get('entry_price', 0))
        quantity = float(data.get('quantity', 0))
        notes = data.get('notes', '')
        
        # Hole Symbol für Coin
        coin_symbol = get_coin_symbol(coin_id)
        
        # Hole AI Konfidenz
        prediction = get_cached_prediction(coin_id)
        confidence = prediction.get('confidence', 0) if 'error' not in prediction else 0
        
        # Füge Position hinzu
        success = portfolio_manager.add_position(
            portfolio_name, coin_id, coin_symbol, 
            quantity, entry_price, confidence, notes
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Position added successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to add position'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio/sell/<portfolio_name>/<coin_id>', methods=['POST'])
def api_sell_position(portfolio_name, coin_id):
    """API Endpoint zum Verkaufen einer Position"""
    try:
        data = request.get_json() if request.is_json else {}
        quantity = data.get('quantity') if data else None
        
        # Hole aktuellen Preis
        current_prices = portfolio_manager.get_current_prices([coin_id])
        current_price = current_prices.get(coin_id, 0)
        
        if current_price == 0:
            return jsonify({'success': False, 'error': 'Could not get current price'})
        
        # Verkaufe Position
        success = portfolio_manager.remove_position(portfolio_name, coin_id, current_price, quantity)
        
        if success:
            # Berechne P&L für Response
            portfolio = portfolio_manager.portfolios.get(portfolio_name)
            if portfolio:
                # Finde verkaufte Position für P&L Berechnung
                pnl = 0  # Vereinfachte P&L Berechnung
                return jsonify({
                    'success': True, 
                    'message': 'Position sold successfully',
                    'pnl': pnl,
                    'current_price': current_price
                })
            else:
                return jsonify({'success': True, 'message': 'Position sold successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to sell position'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Alert API Endpoints
@app.route('/api/alerts/stats')
def api_alert_stats():
    """API Endpoint für Alert-Statistiken"""
    try:
        stats = alert_manager.get_alert_statistics()
        return jsonify(convert_numpy_types(stats))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts/active')
def api_active_alerts():
    """API Endpoint für aktive Alerts"""
    try:
        alerts = alert_manager.get_active_alerts()
        return jsonify(convert_numpy_types(alerts))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts/history')
def api_alert_history():
    """API Endpoint für Alert-History"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = alert_manager.get_alert_history(limit)
        return jsonify(convert_numpy_types(history))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts/notifications')
def api_alert_notifications():
    """API Endpoint für Recent Notifications"""
    try:
        notifications = alert_manager.get_alert_history(20)
        return jsonify(convert_numpy_types(notifications))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts/create', methods=['POST'])
def api_create_alert():
    """API Endpoint zum Erstellen von Alerts"""
    try:
        data = request.get_json()
        
        alert_type = data.get('alert_type')
        coin_id = data.get('coin_id', '')
        user_email = data.get('user_email', '')
        
        if not alert_type:
            return jsonify({'success': False, 'error': 'Alert type required'})
        
        # Hole coin_symbol
        coin_symbol = get_coin_symbol(coin_id) if coin_id else ''
        
        alert_id = None
        
        if alert_type == 'confidence':
            min_confidence = data.get('min_confidence', 0.7)
            alert_id = alert_manager.create_confidence_alert(
                coin_id, coin_symbol, min_confidence, user_email
            )
            
        elif alert_type == 'price':
            target_price = data.get('target_price', 0)
            direction = data.get('direction', 'above')
            if target_price <= 0:
                return jsonify({'success': False, 'error': 'Valid target price required'})
            alert_id = alert_manager.create_price_alert(
                coin_id, coin_symbol, target_price, direction, user_email
            )
            
        elif alert_type == 'stop_loss':
            stop_loss_price = data.get('stop_loss_price', 0)
            if stop_loss_price <= 0:
                return jsonify({'success': False, 'error': 'Valid stop loss price required'})
            alert_id = alert_manager.create_stop_loss_alert(
                coin_id, coin_symbol, stop_loss_price, user_email
            )
            
        elif alert_type == 'take_profit':
            take_profit_price = data.get('take_profit_price', 0)
            if take_profit_price <= 0:
                return jsonify({'success': False, 'error': 'Valid take profit price required'})
            alert_id = alert_manager.create_take_profit_alert(
                coin_id, coin_symbol, take_profit_price, user_email
            )
            
        elif alert_type == 'portfolio_risk':
            max_loss_pct = data.get('max_loss_pct', -10.0)
            alert_id = alert_manager.create_portfolio_risk_alert(
                'Main Portfolio', max_loss_pct, user_email
            )
        
        if alert_id:
            return jsonify({'success': True, 'alert_id': alert_id})
        else:
            return jsonify({'success': False, 'error': 'Failed to create alert'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/delete/<alert_id>', methods=['DELETE'])
def api_delete_alert(alert_id):
    """API Endpoint zum Löschen von Alerts"""
    try:
        success = alert_manager.delete_alert(alert_id)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Alert not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/start-monitoring', methods=['POST'])
def api_start_monitoring():
    """API Endpoint zum Starten des Monitorings"""
    try:
        alert_manager.start_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/stop-monitoring', methods=['POST'])
def api_stop_monitoring():
    """API Endpoint zum Stoppen des Monitorings"""
    try:
        alert_manager.stop_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/status')
def api_monitoring_status():
    """API Endpoint für Monitoring-Status"""
    try:
        stats = alert_manager.get_alert_statistics()
        return jsonify({
            'monitoring_active': stats.get('monitoring_active', False),
            'active_alerts': stats.get('active_alerts', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 Crypto AI Web Interface startet...")
    print("📊 Verfügbar auf: http://localhost:5001")
    print("🚀 Dashboard: http://localhost:5001/dashboard")
    print("📈 Analyse: http://localhost:5001/analysis")
    
    app.run(debug=True, host='0.0.0.0', port=5001)