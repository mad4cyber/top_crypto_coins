#!/usr/bin/env python3
"""
🌐 FastAPI Web-Server für Kryptowährungs-Analyse
Autor: mad4cyber
Version: 3.0 - Web Edition
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
import asyncio
import uvicorn

from pycoingecko import CoinGeckoAPI
from crypto_analyzer import CryptoAnalyzer, PortfolioManager

# FastAPI App initialisieren
app = FastAPI(
    title="🚀 Krypto-Analyse API",
    description="Professionelle Kryptowährungs-Analyse mit Portfolio-Management",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware für Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globale Instanzen
analyzer = CryptoAnalyzer()
portfolio_manager = PortfolioManager()
cg = CoinGeckoAPI()

# Cache für bessere Performance
cache = {}
cache_ttl = 300  # 5 Minuten

# Pydantic Modelle für API
class CryptoData(BaseModel):
    id: str
    symbol: str
    name: str
    rang: int
    preis: float
    marktkapitalisierung: float
    change_24h: float
    change_7d: Optional[float] = None
    change_30d: Optional[float] = None
    volume: Optional[float] = None

class PortfolioItem(BaseModel):
    symbol: str
    amount: float
    purchase_price: Optional[float] = None

class PriceAlert(BaseModel):
    symbol: str
    target_price: float
    alert_type: str  # "above" or "below"
    active: bool = True

class MarketSummary(BaseModel):
    total_market_cap: float
    avg_24h_change: float
    bullish_count: int
    bearish_count: int
    sentiment: str
    top_gainer: Dict[str, Any]
    top_loser: Dict[str, Any]

# Utility Funktionen
def get_cached_data(key: str) -> Optional[Any]:
    """Cache-Daten abrufen"""
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < cache_ttl:
            return data
    return None

def set_cache(key: str, data: Any):
    """Daten in Cache speichern"""
    cache[key] = (data, time.time())

async def fetch_crypto_data(num: int = 10, currency: str = "eur") -> List[Dict]:
    """Async Kryptodaten abrufen"""
    cache_key = f"crypto_data_{num}_{currency}"
    
    # Cache prüfen
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        # API Call (in Thread-Pool da CoinGecko sync ist)
        loop = asyncio.get_event_loop()
        coins = await loop.run_in_executor(
            None, 
            lambda: cg.get_coins_markets(
                vs_currency=currency, 
                order="market_cap", 
                per_page=num,
                page=1, 
                sparkline=False, 
                price_change_percentage='24h,7d,30d'
            )
        )
        
        # Daten verarbeiten
        processed_data = []
        for coin in coins:
            processed_data.append({
                "id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "rang": coin["market_cap_rank"],
                "preis": coin["current_price"],
                "marktkapitalisierung": coin["market_cap"],
                "change_24h": coin.get("price_change_percentage_24h_in_currency", 0),
                "change_7d": coin.get("price_change_percentage_7d_in_currency", 0),
                "change_30d": coin.get("price_change_percentage_30d_in_currency", 0),
                "volume": coin.get("total_volume", 0)
            })
        
        set_cache(cache_key, processed_data)
        return processed_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Daten: {str(e)}")

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Haupt-Dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Krypto-Analyse Dashboard</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0f172a; color: white; }
            .header { text-align: center; margin-bottom: 30px; }
            .card { background: #1e293b; padding: 20px; border-radius: 10px; margin: 10px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .crypto-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #334155; }
            .positive { color: #10b981; }
            .negative { color: #ef4444; }
            .button { background: #3b82f6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .button:hover { background: #2563eb; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Krypto-Analyse Dashboard v3.0</h1>
            <p>Professionelle Kryptowährungs-Analyse</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Top Kryptowährungen</h3>
                <div id="crypto-list">Lade Daten...</div>
                <button class="button" onclick="loadCryptoData()">🔄 Aktualisieren</button>
            </div>
            
            <div class="card">
                <h3>📈 Markt-Übersicht</h3>
                <div id="market-summary">Lade Daten...</div>
            </div>
            
            <div class="card">
                <h3>💼 Portfolio</h3>
                <div id="portfolio">Kein Portfolio gefunden</div>
                <button class="button" onclick="loadPortfolio()">💼 Portfolio laden</button>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 Preisdiagramm</h3>
            <canvas id="priceChart" width="400" height="200"></canvas>
        </div>
        
        <script>
            async function loadCryptoData() {
                try {
                    const response = await fetch('/api/crypto?num=10');
                    const data = await response.json();
                    
                    const listElement = document.getElementById('crypto-list');
                    listElement.innerHTML = data.map(crypto => `
                        <div class="crypto-item">
                            <div>
                                <strong>${crypto.symbol}</strong> - ${crypto.name}
                                <br><small>#${crypto.rang}</small>
                            </div>
                            <div style="text-align: right;">
                                <div>€${crypto.preis.toFixed(2)}</div>
                                <div class="${crypto.change_24h >= 0 ? 'positive' : 'negative'}">
                                    ${crypto.change_24h >= 0 ? '📈' : '📉'} ${crypto.change_24h.toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    updateChart(data);
                } catch (error) {
                    console.error('Fehler beim Laden der Daten:', error);
                }
            }
            
            async function loadMarketSummary() {
                try {
                    const response = await fetch('/api/market-summary');
                    const data = await response.json();
                    
                    document.getElementById('market-summary').innerHTML = `
                        <div>💰 Gesamtmarktkapitalisierung: €${(data.total_market_cap/1e9).toFixed(0)}B</div>
                        <div>📈 Durchschnittliche 24h-Änderung: ${data.avg_24h_change.toFixed(1)}%</div>
                        <div>💭 Markt-Sentiment: ${data.sentiment}</div>
                        <div>🚀 Top Gewinner: ${data.top_gainer.symbol} (+${data.top_gainer.change_24h.toFixed(1)}%)</div>
                        <div>💥 Top Verlierer: ${data.top_loser.symbol} (${data.top_loser.change_24h.toFixed(1)}%)</div>
                    `;
                } catch (error) {
                    console.error('Fehler beim Laden der Marktdaten:', error);
                }
            }
            
            function updateChart(data) {
                const ctx = document.getElementById('priceChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: data.slice(0, 10).map(crypto => crypto.symbol),
                        datasets: [{
                            label: '24h Änderung (%)',
                            data: data.slice(0, 10).map(crypto => crypto.change_24h),
                            backgroundColor: data.slice(0, 10).map(crypto => 
                                crypto.change_24h >= 0 ? '#10b981' : '#ef4444'
                            )
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: 'white' } } },
                        scales: {
                            y: { ticks: { color: 'white' }, grid: { color: '#374151' } },
                            x: { ticks: { color: 'white' }, grid: { color: '#374151' } }
                        }
                    }
                });
            }
            
            // Initial laden
            loadCryptoData();
            loadMarketSummary();
            
            // Auto-refresh alle 60 Sekunden
            setInterval(() => {
                loadCryptoData();
                loadMarketSummary();
            }, 60000);
        </script>
    </body>
    </html>
    """

@app.get("/api/crypto")
async def get_crypto_data(num: int = 10, currency: str = "eur"):
    """Top Kryptowährungen abrufen"""
    return await fetch_crypto_data(num, currency)

@app.get("/api/market-summary", response_model=MarketSummary)
async def get_market_summary(num: int = 100, currency: str = "eur"):
    """Markt-Zusammenfassung"""
    data = await fetch_crypto_data(num, currency)
    
    if not data:
        raise HTTPException(status_code=404, detail="Keine Marktdaten verfügbar")
    
    # Statistiken berechnen
    total_market_cap = sum(item["marktkapitalisierung"] for item in data)
    avg_24h_change = sum(item["change_24h"] for item in data) / len(data)
    
    bullish_count = len([item for item in data if item["change_24h"] > 0])
    bearish_count = len([item for item in data if item["change_24h"] < 0])
    
    # Sentiment bestimmen
    bullish_pct = (bullish_count / len(data)) * 100
    if bullish_pct > 60:
        sentiment = "🚀 Sehr Bullish"
    elif bullish_pct > 50:
        sentiment = "📈 Bullish"
    elif bullish_pct < 40:
        sentiment = "📉 Bearish"
    else:
        sentiment = "⚖️ Neutral"
    
    # Top Gewinner/Verlierer
    top_gainer = max(data, key=lambda x: x["change_24h"])
    top_loser = min(data, key=lambda x: x["change_24h"])
    
    return MarketSummary(
        total_market_cap=total_market_cap,
        avg_24h_change=avg_24h_change,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        sentiment=sentiment,
        top_gainer=top_gainer,
        top_loser=top_loser
    )

@app.get("/api/portfolio")
async def get_portfolio():
    """Portfolio abrufen"""
    try:
        portfolio = portfolio_manager.portfolio
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen des Portfolios: {str(e)}")

@app.post("/api/portfolio/add")
async def add_to_portfolio(item: PortfolioItem):
    """Kryptowährung zum Portfolio hinzufügen"""
    try:
        portfolio_manager.add_holding(
            item.symbol.upper(), 
            item.amount, 
            item.purchase_price
        )
        return {"message": f"✅ {item.symbol.upper()} zum Portfolio hinzugefügt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Hinzufügen: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Gesundheitsprüfung"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0",
        "cache_entries": len(cache)
    }

@app.get("/api/stats")
async def get_stats():
    """API-Statistiken"""
    return {
        "cache_entries": len(cache),
        "uptime": "N/A",  # TODO: Implementieren
        "api_calls": "N/A",  # TODO: Implementieren
        "version": "3.0"
    }

# WebSocket für Live-Updates (Optional)
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket für Live-Updates"""
    await websocket.accept()
    try:
        while True:
            # Alle 30 Sekunden aktuelle Daten senden
            data = await fetch_crypto_data(10)
            await websocket.send_json({"type": "crypto_update", "data": data})
            await asyncio.sleep(30)
    except Exception as e:
        print(f"WebSocket Fehler: {e}")

if __name__ == "__main__":
    print("🚀 Starte Krypto-Analyse Web-Server...")
    print("🌐 Dashboard: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
