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
from performance_tracker import PerformanceTracker
from ml_portfolio_optimizer import MLPortfolioOptimizer
from dynamic_allocator import DynamicAllocator
from portfolio_backtester import PortfolioBacktester
from realtime_sentiment_engine import RealtimeSentimentEngine
from trading_interface import TradingInterface
from multi_exchange_integration import MultiExchangeIntegration
from advanced_technical_analysis import AdvancedTechnicalAnalysis

# Flask App
app = Flask(__name__)
app.secret_key = 'crypto_ai_secret_key_2025'

# Global Instances
predictor = CryptoAIPredictor()
multi_analyzer = MultiCoinAIAnalysis()
sentiment_analyzer = MarketSentimentAnalyzer()
portfolio_manager = PortfolioManager()
alert_manager = AlertManager()
performance_tracker = PerformanceTracker()
ml_optimizer = MLPortfolioOptimizer()
dynamic_allocator = DynamicAllocator()
portfolio_backtester = PortfolioBacktester()
sentiment_engine = RealtimeSentimentEngine()
trading_interface = TradingInterface()

# Multi-Exchange Integration
multi_exchange = None
try:
    multi_exchange = MultiExchangeIntegration()
    print("✅ Multi-Exchange Integration loaded")
except Exception as e:
    print(f"❌ Failed to load Multi-Exchange Integration: {e}")

# Advanced Technical Analysis
tech_analyzer = None
try:
    tech_analyzer = AdvancedTechnicalAnalysis()
    print("✅ Advanced Technical Analysis loaded")
except Exception as e:
    print(f"❌ Failed to load Advanced Technical Analysis: {e}")

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
        
        # Record prediction for performance tracking
        coin_symbol = get_coin_symbol(coin_id)
        performance_tracker.record_prediction(coin_id, coin_symbol, prediction)
        
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

@app.route('/performance')
def performance():
    """Performance Analytics Seite"""
    return render_template('performance.html')

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

# Performance Tracking API Endpoints
@app.route('/api/performance/summary')
def api_performance_summary():
    """API Endpoint für Performance-Zusammenfassung"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        # Update predictions first
        performance_tracker.update_prediction_actuals()
        
        summary = performance_tracker.get_performance_summary(days_back)
        return jsonify(convert_numpy_types(summary))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance/rankings')
def api_performance_rankings():
    """API Endpoint für Coin Performance Rankings"""
    try:
        days_back = request.args.get('days', 30, type=int)
        rankings = performance_tracker.get_coin_performance_ranking(days_back)
        return jsonify(convert_numpy_types(rankings))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance/backtests')
def api_performance_backtests():
    """API Endpoint für Backtest-Ergebnisse"""
    try:
        results = []
        for backtest in performance_tracker.backtest_results:
            results.append({
                'strategy_name': backtest.strategy_name,
                'coin_id': backtest.coin_id,
                'start_date': backtest.start_date,
                'end_date': backtest.end_date,
                'total_return_pct': backtest.total_return_pct,
                'total_trades': backtest.total_trades,
                'win_rate': backtest.win_rate,
                'max_drawdown': backtest.max_drawdown,
                'sharpe_ratio': backtest.sharpe_ratio
            })
        return jsonify(convert_numpy_types(results))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance/recent')
def api_performance_recent():
    """API Endpoint für Recent Predictions"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Get recent predictions
        recent_predictions = list(performance_tracker.prediction_records.values())
        recent_predictions.sort(key=lambda x: x.prediction_date, reverse=True)
        recent_predictions = recent_predictions[:limit]
        
        result = []
        for pred in recent_predictions:
            result.append({
                'prediction_date': pred.prediction_date,
                'coin_symbol': pred.coin_symbol,
                'predicted_change_pct': pred.predicted_change_pct,
                'actual_change_pct': pred.actual_change_pct,
                'confidence': pred.confidence,
                'accuracy_score': pred.accuracy_score,
                'is_resolved': pred.is_resolved
            })
        
        return jsonify(convert_numpy_types(result))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance/backtest', methods=['POST'])
def api_run_backtest():
    """API Endpoint zum Ausführen eines Backtests"""
    try:
        data = request.get_json()
        
        coin_id = data.get('coin_id')
        strategy_name = data.get('strategy_name', 'high_confidence')
        initial_capital = data.get('initial_capital', 10000)
        days_back = data.get('days_back', 30)
        
        if not coin_id:
            return jsonify({'success': False, 'error': 'Coin ID required'})
        
        # Run backtest
        result = performance_tracker.run_backtest_strategy(
            coin_id, strategy_name, days_back, initial_capital
        )
        
        return jsonify({
            'success': True,
            'strategy_name': result.strategy_name,
            'coin_id': result.coin_id,
            'total_return_pct': result.total_return_pct,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/performance/update', methods=['POST'])
def api_update_predictions():
    """API Endpoint zum manuellen Update der Predictions"""
    try:
        updated_count = performance_tracker.update_prediction_actuals()
        return jsonify({
            'success': True,
            'updated_predictions': updated_count,
            'message': f'Updated {updated_count} predictions with actual results'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ML Portfolio Optimization Routes
@app.route('/ml-optimization')
def ml_optimization():
    """ML Portfolio Optimization Seite"""
    return render_template('ml_optimization.html')

@app.route('/api/ml-optimization/optimize', methods=['POST'])
def api_ml_optimize():
    """API Endpoint für ML Portfolio Optimierung"""
    try:
        data = request.get_json()
        
        portfolio_name = data.get('portfolio_name', 'Main Portfolio')
        optimization_type = data.get('optimization_type', 'risk_adjusted')
        
        # Run ML optimization
        result = ml_optimizer.optimize_portfolio(portfolio_name, optimization_type)
        
        # Convert to dict for JSON response
        response_data = {
            'timestamp': result.timestamp,
            'portfolio_name': result.portfolio_name,
            'optimization_type': result.optimization_type,
            'recommended_weights': result.recommended_weights,
            'expected_return': result.expected_return,
            'expected_risk': result.expected_risk,
            'sharpe_ratio': result.sharpe_ratio,
            'rebalancing_required': result.rebalancing_required,
            'confidence_score': result.confidence_score,
            'market_regime': result.market_regime,
            'reasoning': result.reasoning
        }
        
        return jsonify(convert_numpy_types(response_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml-optimization/rebalancing-trades', methods=['POST'])
def api_rebalancing_trades():
    """API Endpoint für Rebalancing-Trades"""
    try:
        data = request.get_json()
        
        portfolio_name = data.get('portfolio_name', 'Main Portfolio')
        optimization_result_data = data.get('optimization_result', {})
        
        # Reconstruct OptimizationResult from dict
        from ml_portfolio_optimizer import OptimizationResult
        optimization_result = OptimizationResult(**optimization_result_data)
        
        trades = ml_optimizer.get_rebalancing_trades(portfolio_name, optimization_result)
        
        return jsonify(convert_numpy_types(trades))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml-optimization/history')
def api_optimization_history():
    """API Endpoint für Optimierungs-Historie"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Get recent optimization history
        history = ml_optimizer.optimization_history[-limit:]
        
        # Convert to dict format
        history_data = []
        for result in reversed(history):  # Most recent first
            history_data.append({
                'timestamp': result.timestamp,
                'portfolio_name': result.portfolio_name,
                'optimization_type': result.optimization_type,
                'expected_return': result.expected_return,
                'expected_risk': result.expected_risk,
                'sharpe_ratio': result.sharpe_ratio,
                'market_regime': result.market_regime,
                'confidence_score': result.confidence_score,
                'rebalancing_required': result.rebalancing_required
            })
        
        return jsonify(convert_numpy_types(history_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml-optimization/performance')
def api_optimization_performance():
    """API Endpoint für ML Optimization Performance"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        summary = ml_optimizer.get_optimization_summary(days_back)
        
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml-optimization/market-regime')
def api_market_regime():
    """API Endpoint für aktuelle Marktregime-Analyse"""
    try:
        # Get all available coins for regime analysis
        coins = ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana']
        
        regime = ml_optimizer.detect_market_regime(coins)
        
        regime_data = {
            'regime_type': regime.regime_type,
            'confidence': regime.confidence,
            'indicators': regime.indicators,
            'duration_days': regime.duration_days,
            'volatility_level': regime.volatility_level
        }
        
        return jsonify(convert_numpy_types(regime_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Dynamic Allocation API Endpoints
@app.route('/api/dynamic-allocation/execute', methods=['POST'])
def api_execute_dynamic_allocation():
    """API Endpoint für Dynamic Portfolio Rebalancing"""
    try:
        data = request.get_json()
        portfolio_name = data.get('portfolio_name', 'Main Portfolio')
        
        result = dynamic_allocator.execute_dynamic_rebalancing(portfolio_name)
        
        return jsonify(convert_numpy_types(result))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/dynamic-allocation/market-analysis')
def api_market_analysis():
    """API Endpoint für Market Condition Analysis"""
    try:
        # Get coins for analysis
        coins = ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana']
        
        market_condition = dynamic_allocator.analyze_market_conditions(coins)
        
        condition_data = {
            'timestamp': market_condition.timestamp,
            'overall_trend': market_condition.overall_trend,
            'volatility_regime': market_condition.volatility_regime,
            'correlation_level': market_condition.correlation_level,
            'momentum_strength': market_condition.momentum_strength,
            'fear_greed_level': market_condition.fear_greed_level,
            'recommended_allocation_style': market_condition.recommended_allocation_style
        }
        
        return jsonify(convert_numpy_types(condition_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/dynamic-allocation/signals')
def api_allocation_signals():
    """API Endpoint für Recent Allocation Signals"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get recent signals
        recent_signals = dynamic_allocator.allocation_signals[-limit:]
        
        signals_data = []
        for signal in reversed(recent_signals):  # Most recent first
            signals_data.append({
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'strength': signal.strength,
                'direction': signal.direction,
                'affected_assets': signal.affected_assets,
                'reasoning': signal.reasoning,
                'confidence': signal.confidence,
                'timeframe': signal.timeframe
            })
        
        return jsonify(convert_numpy_types(signals_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/dynamic-allocation/performance')
def api_allocation_performance():
    """API Endpoint für Dynamic Allocation Performance"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        performance = dynamic_allocator.get_allocation_performance(days_back)
        
        return jsonify(convert_numpy_types(performance))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/dynamic-allocation/history')
def api_allocation_history():
    """API Endpoint für Allocation History"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        # Get recent allocation history
        recent_history = dynamic_allocator.allocation_history[-limit:]
        
        return jsonify(convert_numpy_types(list(reversed(recent_history))))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Portfolio Backtesting API Endpoints
@app.route('/api/portfolio-backtest/run', methods=['POST'])
def api_run_portfolio_backtest():
    """API Endpoint für Portfolio Backtesting"""
    try:
        data = request.get_json()
        
        coin_ids = data.get('coin_ids', ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana'])
        days_back = data.get('days_back', 30)
        initial_capital = data.get('initial_capital', 10000)
        
        # Run comprehensive backtest
        result = portfolio_backtester.run_comprehensive_backtest(
            coin_ids=coin_ids,
            days_back=days_back,
            initial_capital=initial_capital
        )
        
        # Convert result to dict format
        strategies_data = []
        for strategy in result.strategies:
            strategy_dict = {
                'strategy_name': strategy.strategy_name,
                'total_return_pct': strategy.total_return_pct,
                'annualized_return_pct': strategy.annualized_return_pct,
                'volatility_pct': strategy.volatility_pct,
                'sharpe_ratio': strategy.sharpe_ratio,
                'max_drawdown_pct': strategy.max_drawdown_pct,
                'recovery_days': strategy.recovery_days,
                'win_rate': strategy.win_rate,
                'rebalancing_frequency': strategy.rebalancing_frequency,
                'transaction_costs': strategy.transaction_costs,
                'net_return_pct': strategy.net_return_pct,
                'risk_adjusted_return': strategy.risk_adjusted_return,
                'final_capital': strategy.final_capital,
                'performance_metrics': strategy.performance_metrics
            }
            strategies_data.append(strategy_dict)
        
        response_data = {
            'timestamp': result.timestamp,
            'strategies': strategies_data,
            'best_strategy': result.best_strategy,
            'performance_ranking': result.performance_ranking,
            'market_conditions': result.market_conditions,
            'insights': result.insights
        }
        
        return jsonify(convert_numpy_types(response_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio-backtest/summary')
def api_backtest_summary():
    """API Endpoint für Backtesting Summary"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        summary = portfolio_backtester.get_backtest_summary(limit)
        
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio-backtest/history')
def api_backtest_history():
    """API Endpoint für Backtesting History"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get recent backtest results
        recent_results = portfolio_backtester.backtest_results[-limit:]
        
        history_data = []
        for result in reversed(recent_results):  # Most recent first
            # Simplified version for history view
            history_item = {
                'timestamp': result.timestamp,
                'best_strategy': result.best_strategy,
                'strategies_count': len(result.strategies),
                'market_conditions': result.market_conditions,
                'top_insights': result.insights[:3] if result.insights else [],
                'performance_ranking': result.performance_ranking[:3]  # Top 3 only
            }
            history_data.append(history_item)
        
        return jsonify(convert_numpy_types(history_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio-backtest/strategy-comparison')
def api_strategy_comparison():
    """API Endpoint für Strategy Performance Comparison"""
    try:
        # Get latest backtest result for comparison
        if not portfolio_backtester.backtest_results:
            return jsonify({'message': 'No backtest results available'})
        
        latest_result = portfolio_backtester.backtest_results[-1]
        
        # Create comparison data
        comparison_data = {
            'timestamp': latest_result.timestamp,
            'strategies': [],
            'performance_metrics': ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate'],
            'best_strategy': latest_result.best_strategy,
            'insights': latest_result.insights
        }
        
        for strategy in latest_result.strategies:
            strategy_data = {
                'name': strategy.strategy_name,
                'metrics': {
                    'total_return_pct': strategy.total_return_pct,
                    'sharpe_ratio': strategy.sharpe_ratio,
                    'max_drawdown_pct': strategy.max_drawdown_pct,
                    'win_rate': strategy.win_rate * 100,
                    'volatility_pct': strategy.volatility_pct,
                    'final_capital': strategy.final_capital,
                    'recovery_days': strategy.recovery_days
                }
            }
            comparison_data['strategies'].append(strategy_data)
        
        return jsonify(convert_numpy_types(comparison_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Real-time Sentiment Analysis Routes
@app.route('/realtime-sentiment')
def realtime_sentiment():
    """Real-time Sentiment Analysis Seite"""
    return render_template('realtime_sentiment.html')

@app.route('/api/realtime-sentiment/analyze', methods=['POST'])
def api_realtime_sentiment_analyze():
    """API Endpoint für Real-time Sentiment Analysis"""
    try:
        data = request.get_json()
        
        coin_symbols = data.get('coin_symbols', ['BTC', 'ETH'])
        hours_back = data.get('hours_back', 24)
        
        # Run real-time sentiment analysis
        results = sentiment_engine.run_realtime_analysis(coin_symbols)
        
        return jsonify(convert_numpy_types(results))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/realtime-sentiment/summary')
def api_sentiment_summary():
    """API Endpoint für Sentiment Summary"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        
        summary = sentiment_engine.get_sentiment_summary(hours_back)
        
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/realtime-sentiment/events')
def api_sentiment_events():
    """API Endpoint für Recent Sentiment Events"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get recent events
        recent_events = sentiment_engine.sentiment_events[-limit:]
        
        events_data = []
        for event in reversed(recent_events):  # Most recent first
            events_data.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'coin_symbol': event.coin_symbol,
                'severity': event.severity,
                'description': event.description,
                'sentiment_change': event.sentiment_change,
                'expected_impact': event.expected_impact,
                'confidence': event.confidence,
                'sources': event.sources,
                'duration_estimate': event.duration_estimate
            })
        
        return jsonify(convert_numpy_types(events_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/realtime-sentiment/fear-greed')
def api_fear_greed_index():
    """API Endpoint für Fear & Greed Index"""
    try:
        fear_greed_value = sentiment_engine.fetch_fear_greed_index()
        
        # Classify the value
        if fear_greed_value >= 80:
            classification = 'Extreme Greed'
            color = '#28a745'
        elif fear_greed_value >= 60:
            classification = 'Greed'
            color = '#20c997'
        elif fear_greed_value >= 40:
            classification = 'Neutral'
            color = '#6c757d'
        elif fear_greed_value >= 20:
            classification = 'Fear'
            color = '#fd7e14'
        else:
            classification = 'Extreme Fear'
            color = '#dc3545'
        
        return jsonify({
            'value': fear_greed_value,
            'classification': classification,
            'color': color,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/realtime-sentiment/aggregated/<coin_symbol>')
def api_aggregated_sentiment(coin_symbol):
    """API Endpoint für Aggregated Sentiment eines spezifischen Coins"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        
        # Get aggregated sentiment for specific coin
        aggregated = sentiment_engine.aggregate_sentiment_score(coin_symbol.upper(), hours_back)
        
        return jsonify(convert_numpy_types(aggregated))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Trading Interface Routes
@app.route('/trading-interface')
def trading_interface_page():
    """Professional Trading Interface Seite"""
    return render_template('trading_interface.html')

@app.route('/professional-charts')
def professional_charts():
    """Professional Charts Seite"""
    return render_template('professional_charts.html')

@app.route('/risk-management')
def risk_management():
    """Risk Management Dashboard Seite"""
    return render_template('risk_management.html')

@app.route('/api/trading/create-order', methods=['POST'])
def api_create_order():
    """API Endpoint zum Erstellen einer Order"""
    try:
        data = request.get_json()
        
        # Import OrderType and OrderSide enums
        from trading_interface import OrderType, OrderSide
        
        order_id = trading_interface.create_order(
            coin_id=data.get('coin_id'),
            coin_symbol=data.get('coin_symbol'),
            order_type=OrderType(data.get('order_type')),
            side=OrderSide(data.get('side')),
            quantity=float(data.get('quantity')),
            price=float(data.get('price')) if data.get('price') else None,
            stop_price=float(data.get('stop_price')) if data.get('stop_price') else None,
            portfolio_name=data.get('portfolio_name', 'Main Portfolio'),
            notes=data.get('notes', '')
        )
        
        return jsonify({
            'success': True,
            'order_id': order_id,
            'message': 'Order created successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/orders')
def api_get_orders():
    """API Endpoint zum Abrufen aller offenen Orders"""
    try:
        portfolio_name = request.args.get('portfolio', None)
        open_orders = trading_interface.get_open_orders(portfolio_name)
        
        # Convert orders to dict format
        orders_data = [
            {
                'order_id': order.order_id,
                'coin_id': order.coin_id,
                'coin_symbol': order.coin_symbol,
                'order_type': order.order_type.value,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'created_at': order.created_at,
                'portfolio_name': order.portfolio_name,
                'notes': order.notes
            }
            for order in open_orders
        ]
        
        return jsonify({'orders': orders_data})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/positions')
def api_get_positions():
    """API Endpoint zum Abrufen aller offenen Positionen"""
    try:
        portfolio_name = request.args.get('portfolio', None)
        open_positions = trading_interface.get_open_positions(portfolio_name)
        
        # Convert positions to dict format
        positions_data = [
            {
                'position_id': pos.position_id,
                'coin_id': pos.coin_id,
                'coin_symbol': pos.coin_symbol,
                'side': pos.side,
                'quantity': pos.quantity,
                'avg_entry_price': pos.avg_entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'status': pos.status.value,
                'opened_at': pos.opened_at,
                'portfolio_name': pos.portfolio_name,
                'stop_loss_price': pos.stop_loss_price,
                'take_profit_price': pos.take_profit_price
            }
            for pos in open_positions
        ]
        
        return jsonify({'positions': positions_data})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/cancel-order/<order_id>', methods=['POST'])
def api_cancel_order(order_id):
    """API Endpoint zum Stornieren einer Order"""
    try:
        success = trading_interface.cancel_order(order_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Order cancelled successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to cancel order'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/update-pnl', methods=['POST'])
def api_update_pnl():
    """API Endpoint zum Aktualisieren der P&L für alle Positionen"""
    try:
        updated_count = trading_interface.update_positions_pnl()
        
        return jsonify({
            'success': True,
            'updated_positions': updated_count,
            'message': f'Updated P&L for {updated_count} positions'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/risk-metrics')
def api_get_risk_metrics():
    """API Endpoint zum Abrufen der Risk Metrics"""
    try:
        portfolio_name = request.args.get('portfolio', 'Main Portfolio')
        risk_metrics = trading_interface.calculate_risk_metrics(portfolio_name)
        
        # Convert to dict
        metrics_data = {
            'portfolio_name': risk_metrics.portfolio_name,
            'total_exposure': risk_metrics.total_exposure,
            'max_position_size': risk_metrics.max_position_size,
            'current_drawdown': risk_metrics.current_drawdown,
            'var_1_day': risk_metrics.var_1_day,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'risk_score': risk_metrics.risk_score,
            'margin_usage': risk_metrics.margin_usage,
            'open_positions': risk_metrics.open_positions,
            'total_unrealized_pnl': risk_metrics.total_unrealized_pnl
        }
        
        return jsonify(convert_numpy_types(metrics_data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/trading-summary')
def api_trading_summary():
    """API Endpoint für Trading Activity Summary"""
    try:
        days_back = request.args.get('days', 30, type=int)
        summary = trading_interface.get_trading_summary(days_back)
        
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/stop-loss/<position_id>', methods=['POST'])
def api_add_stop_loss(position_id):
    """API Endpoint zum Hinzufügen einer Stop-Loss Order"""
    try:
        data = request.get_json()
        stop_price = float(data.get('stop_price'))
        
        order_id = trading_interface.create_stop_loss_order(position_id, stop_price)
        
        if order_id:
            return jsonify({
                'success': True,
                'order_id': order_id,
                'message': 'Stop-loss order created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create stop-loss order'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/take-profit/<position_id>', methods=['POST'])
def api_add_take_profit(position_id):
    """API Endpoint zum Hinzufügen einer Take-Profit Order"""
    try:
        data = request.get_json()
        target_price = float(data.get('target_price'))
        
        order_id = trading_interface.create_take_profit_order(position_id, target_price)
        
        if order_id:
            return jsonify({
                'success': True,
                'order_id': order_id,
                'message': 'Take-profit order created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create take-profit order'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/automation/start', methods=['POST'])
def api_start_automation():
    """API Endpoint zum Starten der Trading Automation"""
    try:
        success = trading_interface.start_automation()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Trading automation started successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Automation already running'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/automation/stop', methods=['POST'])
def api_stop_automation():
    """API Endpoint zum Stoppen der Trading Automation"""
    try:
        success = trading_interface.stop_automation()
        
        return jsonify({
            'success': True,
            'message': 'Trading automation stopped successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/automation/status')
def api_automation_status():
    """API Endpoint für Automation Status"""
    try:
        status = trading_interface.get_automation_status()
        
        return jsonify(convert_numpy_types(status))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/automation/set-position', methods=['POST'])
def api_set_position_automation():
    """API Endpoint zum Setzen der Position Automation"""
    try:
        data = request.get_json()
        
        position_id = data.get('position_id')
        stop_loss_pct = data.get('stop_loss_pct')
        take_profit_pct = data.get('take_profit_pct')
        trailing_stop_pct = data.get('trailing_stop_pct')
        
        success = trading_interface.set_position_automation(
            position_id=position_id,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Position automation configured successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to configure position automation'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/realtime/start', methods=['POST'])
def api_start_realtime_tracking():
    """API Endpoint zum Starten des Real-time Tracking"""
    try:
        success = trading_interface.start_realtime_tracking()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Real-time tracking started successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Real-time tracking already running'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/realtime/stop', methods=['POST'])
def api_stop_realtime_tracking():
    """API Endpoint zum Stoppen des Real-time Tracking"""
    try:
        success = trading_interface.stop_realtime_tracking()
        
        return jsonify({
            'success': True,
            'message': 'Real-time tracking stopped successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/realtime/summary')
def api_realtime_summary():
    """API Endpoint für Real-time Tracking Summary"""
    try:
        summary = trading_interface.get_realtime_summary()
        
        return jsonify(convert_numpy_types(summary))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/realtime/positions')
def api_realtime_positions():
    """API Endpoint für Real-time Position Data"""
    try:
        position_id = request.args.get('position_id', None)
        data = trading_interface.get_position_realtime_data(position_id)
        
        return jsonify(convert_numpy_types(data))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/realtime/pnl-history')
def api_pnl_history():
    """API Endpoint für P&L History"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        history = trading_interface.get_pnl_history(hours_back)
        
        return jsonify(convert_numpy_types(history))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading/realtime/alerts')
def api_realtime_alerts():
    """API Endpoint für Recent Alerts"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        alerts = trading_interface.get_recent_alerts(hours_back)
        
        return jsonify(convert_numpy_types(alerts))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Multi-Exchange Integration Routes
@app.route('/multi-exchange')
def multi_exchange_page():
    """Multi-Exchange Integration Dashboard"""
    return render_template('multi_exchange.html')

@app.route('/api/multi-exchange/connections')
def api_test_connections():
    """API Endpoint zum Testen aller Exchange-Verbindungen"""
    try:
        if not multi_exchange:
            return jsonify({'error': 'Multi-exchange integration not available'})
        
        connections = multi_exchange.test_all_connections()
        health_status = multi_exchange.get_exchange_health()
        
        return jsonify({
            'connections': connections,
            'health_status': convert_numpy_types({
                name: {
                    'status': health.status.value,
                    'latency_ms': health.latency_ms,
                    'uptime_pct': health.uptime_pct,
                    'error_rate': health.error_rate,
                    'api_calls_remaining': health.api_calls_remaining,
                    'last_check': health.last_check,
                    'issues': health.issues
                } for name, health in health_status.items()
            })
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/multi-exchange/unified-balance')
def api_unified_balance():
    """API Endpoint für Unified Portfolio Balance"""
    try:
        if not multi_exchange:
            return jsonify({'error': 'Multi-exchange integration not available'})
        
        balance = multi_exchange.get_unified_balance()
        
        return jsonify(convert_numpy_types(balance))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/multi-exchange/arbitrage')
def api_arbitrage_opportunities():
    """API Endpoint für Arbitrage Opportunities"""
    try:
        if not multi_exchange:
            return jsonify({'error': 'Multi-exchange integration not available'})
        
        symbols = request.args.getlist('symbols') or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        opportunities = multi_exchange.scan_arbitrage_opportunities(symbols)
        
        # Convert opportunities to dict format
        opportunities_data = []
        for opp in opportunities:
            opportunities_data.append({
                'timestamp': opp.timestamp,
                'buy_exchange': opp.buy_exchange,
                'sell_exchange': opp.sell_exchange,
                'symbol': opp.symbol,
                'buy_price': opp.buy_price,
                'sell_price': opp.sell_price,
                'spread_pct': opp.spread_pct,
                'potential_profit': opp.potential_profit,
                'volume': opp.volume,
                'confidence': opp.confidence
            })
        
        return jsonify(convert_numpy_types(opportunities_data))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/multi-exchange/summary')
def api_exchange_summary():
    """API Endpoint für Exchange Integration Summary"""
    try:
        if not multi_exchange:
            return jsonify({'error': 'Multi-exchange integration not available'})
        
        summary = multi_exchange.get_exchange_summary()
        
        return jsonify(convert_numpy_types(summary))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/multi-exchange/monitoring/start', methods=['POST'])
def api_start_exchange_monitoring():
    """API Endpoint zum Starten des Exchange Monitoring"""
    try:
        if not multi_exchange:
            return jsonify({'success': False, 'error': 'Multi-exchange integration not available'})
        
        data = request.get_json() or {}
        health_monitoring = data.get('health_monitoring', True)
        arbitrage_monitoring = data.get('arbitrage_monitoring', True)
        
        if health_monitoring:
            multi_exchange.start_health_monitoring()
        
        if arbitrage_monitoring:
            multi_exchange.start_arbitrage_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Exchange monitoring started',
            'health_monitoring': health_monitoring,
            'arbitrage_monitoring': arbitrage_monitoring
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/multi-exchange/monitoring/stop', methods=['POST'])
def api_stop_exchange_monitoring():
    """API Endpoint zum Stoppen des Exchange Monitoring"""
    try:
        if not multi_exchange:
            return jsonify({'success': False, 'error': 'Multi-exchange integration not available'})
        
        multi_exchange.stop_health_monitoring()
        multi_exchange.stop_arbitrage_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Exchange monitoring stopped'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/multi-exchange/monitoring/status')
def api_monitoring_status():
    """API Endpoint für Exchange Monitoring Status"""
    try:
        if not multi_exchange:
            return jsonify({'error': 'Multi-exchange integration not available'})
        
        return jsonify({
            'health_monitoring_active': multi_exchange.health_monitor_active,
            'arbitrage_monitoring_active': multi_exchange.arbitrage_monitor_active,
            'total_exchanges': len(multi_exchange.exchanges),
            'arbitrage_opportunities': len(multi_exchange.arbitrage_opportunities)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Advanced Technical Analysis Routes
@app.route('/technical-analysis')
def technical_analysis_page():
    """Advanced Technical Analysis Dashboard"""
    return render_template('technical_analysis.html')

@app.route('/api/technical-analysis/signal/<coin_id>')
def api_technical_signal(coin_id):
    """API Endpoint für Comprehensive Technical Signal"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        coin_symbol = get_coin_symbol(coin_id)
        timeframe = request.args.get('timeframe', '1d')
        
        signal = tech_analyzer.generate_comprehensive_signal(coin_id, coin_symbol, timeframe)
        
        # Convert to dict format
        signal_data = {
            'timestamp': signal.timestamp,
            'coin_id': signal.coin_id,
            'coin_symbol': signal.coin_symbol,
            'timeframe': signal.timeframe,
            'overall_signal': signal.overall_signal,
            'signal_strength': signal.signal_strength,
            'confidence': signal.confidence,
            'indicators': [
                {
                    'name': ind.name,
                    'value': ind.value,
                    'signal': ind.signal,
                    'strength': ind.strength.value,
                    'interpretation': ind.interpretation,
                    'confidence': ind.confidence
                }
                for ind in signal.indicators
            ],
            'patterns': [
                {
                    'pattern_type': pattern.pattern_type.value,
                    'confidence': pattern.confidence,
                    'support_level': pattern.support_level,
                    'resistance_level': pattern.resistance_level,
                    'breakout_target': pattern.breakout_target,
                    'risk_reward_ratio': pattern.risk_reward_ratio,
                    'description': pattern.description,
                    'signal': pattern.signal
                }
                for pattern in signal.patterns
            ],
            'trend_analysis': {
                'direction': signal.trend_analysis.direction.value,
                'strength': signal.trend_analysis.strength,
                'duration_days': signal.trend_analysis.duration_days,
                'change_percent': signal.trend_analysis.change_percent,
                'confidence': signal.trend_analysis.confidence,
                'support_levels': signal.trend_analysis.support_levels,
                'resistance_levels': signal.trend_analysis.resistance_levels
            },
            'support_resistance': [
                {
                    'level': sr.level,
                    'level_type': sr.level_type,
                    'strength': sr.strength,
                    'confidence': sr.confidence
                }
                for sr in signal.support_resistance
            ],
            'price_targets': signal.price_targets,
            'risk_assessment': signal.risk_assessment,
            'recommended_action': signal.recommended_action
        }
        
        return jsonify(convert_numpy_types(signal_data))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/multi-signal', methods=['POST'])
def api_multi_technical_signal():
    """API Endpoint für Multi-Coin Technical Analysis"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        data = request.get_json()
        coin_ids = data.get('coin_ids', ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana'])
        timeframe = data.get('timeframe', '1d')
        
        results = {}
        
        for coin_id in coin_ids:
            try:
                coin_symbol = get_coin_symbol(coin_id)
                signal = tech_analyzer.generate_comprehensive_signal(coin_id, coin_symbol, timeframe)
                
                results[coin_id] = {
                    'coin_symbol': coin_symbol,
                    'overall_signal': signal.overall_signal,
                    'signal_strength': signal.signal_strength,
                    'confidence': signal.confidence,
                    'trend_direction': signal.trend_analysis.direction.value,
                    'trend_strength': signal.trend_analysis.strength,
                    'risk_assessment': signal.risk_assessment,
                    'recommended_action': signal.recommended_action,
                    'indicator_count': len(signal.indicators),
                    'pattern_count': len(signal.patterns)
                }
            except Exception as e:
                results[coin_id] = {'error': str(e)}
        
        return jsonify(convert_numpy_types(results))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/indicators/<coin_id>')
def api_technical_indicators(coin_id):
    """API Endpoint für Technical Indicators"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        timeframe = request.args.get('timeframe', '1d')
        indicators = tech_analyzer.analyze_indicators(coin_id, timeframe)
        
        indicators_data = [
            {
                'name': ind.name,
                'value': ind.value,
                'signal': ind.signal,
                'strength': ind.strength.value,
                'interpretation': ind.interpretation,
                'confidence': ind.confidence,
                'timestamp': ind.timestamp
            }
            for ind in indicators
        ]
        
        return jsonify(convert_numpy_types(indicators_data))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/patterns/<coin_id>')
def api_technical_patterns(coin_id):
    """API Endpoint für Pattern Detection"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        timeframe = request.args.get('timeframe', '1d')
        patterns = tech_analyzer.detect_patterns(coin_id, timeframe)
        
        patterns_data = [
            {
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence,
                'support_level': pattern.support_level,
                'resistance_level': pattern.resistance_level,
                'breakout_target': pattern.breakout_target,
                'risk_reward_ratio': pattern.risk_reward_ratio,
                'description': pattern.description,
                'signal': pattern.signal,
                'timeframe': pattern.timeframe
            }
            for pattern in patterns
        ]
        
        return jsonify(convert_numpy_types(patterns_data))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/trend/<coin_id>')
def api_trend_analysis(coin_id):
    """API Endpoint für Trend Analysis"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        timeframe = request.args.get('timeframe', '1d')
        trend = tech_analyzer.analyze_trend(coin_id, timeframe)
        
        trend_data = {
            'timeframe': trend.timeframe,
            'direction': trend.direction.value,
            'strength': trend.strength,
            'duration_days': trend.duration_days,
            'start_price': trend.start_price,
            'current_price': trend.current_price,
            'change_percent': trend.change_percent,
            'confidence': trend.confidence,
            'support_levels': trend.support_levels,
            'resistance_levels': trend.resistance_levels,
            'trend_line_slope': trend.trend_line_slope
        }
        
        return jsonify(convert_numpy_types(trend_data))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/summary')
def api_technical_summary():
    """API Endpoint für Technical Analysis Summary"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        days_back = request.args.get('days', 7, type=int)
        summary = tech_analyzer.get_analysis_summary(days_back)
        
        return jsonify(convert_numpy_types(summary))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-analysis/start-monitoring', methods=['POST'])
def api_start_technical_monitoring():
    """API Endpoint zum Starten der Technical Analysis"""
    try:
        if not tech_analyzer:
            return jsonify({'success': False, 'error': 'Technical analysis not available'})
        
        data = request.get_json() or {}
        coin_list = data.get('coin_list', ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana'])
        
        tech_analyzer.start_continuous_analysis(coin_list)
        
        return jsonify({
            'success': True,
            'message': 'Technical analysis monitoring started',
            'monitoring_coins': coin_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/technical-analysis/stop-monitoring', methods=['POST'])
def api_stop_technical_monitoring():
    """API Endpoint zum Stoppen der Technical Analysis"""
    try:
        if not tech_analyzer:
            return jsonify({'success': False, 'error': 'Technical analysis not available'})
        
        tech_analyzer.stop_continuous_analysis()
        
        return jsonify({
            'success': True,
            'message': 'Technical analysis monitoring stopped'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/technical-analysis/monitoring-status')
def api_technical_monitoring_status():
    """API Endpoint für Technical Analysis Monitoring Status"""
    try:
        if not tech_analyzer:
            return jsonify({'error': 'Technical analysis not available'})
        
        return jsonify({
            'analysis_active': tech_analyzer.analysis_active,
            'analysis_results_count': len(tech_analyzer.analysis_results),
            'pattern_history_count': len(tech_analyzer.pattern_history),
            'timeframes': tech_analyzer.timeframes
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 Crypto AI Web Interface startet...")
    print("📊 Verfügbar auf: http://localhost:5001")
    print("🚀 Dashboard: http://localhost:5001/dashboard")
    print("📈 Analyse: http://localhost:5001/analysis")
    print("🏦 Multi-Exchange: http://localhost:5001/multi-exchange")
    print("💼 Trading Interface: http://localhost:5001/trading-interface")
    print("📊 Technical Analysis: http://localhost:5001/technical-analysis")
    
    app.run(debug=True, host='0.0.0.0', port=5001)