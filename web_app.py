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

# Flask App
app = Flask(__name__)
app.secret_key = 'crypto_ai_secret_key_2025'

# Global Instances
predictor = CryptoAIPredictor()
multi_analyzer = MultiCoinAIAnalysis()

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

def get_cached_prediction(coin_id: str):
    """Hole gecachte Prognose oder erstelle neue"""
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
        prediction_cache[coin_id] = (prediction, now)
        return prediction
    except Exception as e:
        return {'error': str(e)}

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

if __name__ == '__main__':
    print("🌐 Crypto AI Web Interface startet...")
    print("📊 Verfügbar auf: http://localhost:5001")
    print("🚀 Dashboard: http://localhost:5001/dashboard")
    print("📈 Analyse: http://localhost:5001/analysis")
    
    app.run(debug=True, host='0.0.0.0', port=5001)