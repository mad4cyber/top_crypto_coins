#!/usr/bin/env python3
"""
🚀 Live Trading Demo mit Binance Testnet
Autor: mad4cyber
Version: 1.0 - Live Demo Edition

🎯 FEATURES:
- Automatische API-Key Integration
- AI-Powered Live Trading Signale
- Echte Binance Testnet Orders
- Live Portfolio-Tracking
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# API-Keys aus config laden
try:
    import config
    API_KEY = config.BINANCE_TESTNET_API_KEY
    API_SECRET = config.BINANCE_TESTNET_SECRET
    TRADING_SYMBOLS = config.TRADING_SYMBOLS
except ImportError:
    print("❌ config.py nicht gefunden!")
    exit(1)

# Trading Libraries
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Unsere AI-Module
from trading_signals_demo import SimpleTradingSignals

class LiveTradingDemo:
    """🚀 Live Trading Demo mit echten API-Calls"""
    
    def __init__(self):
        self.console = Console()
        
        # Binance Testnet Client
        try:
            self.client = Client(
                api_key=API_KEY,
                api_secret=API_SECRET,
                testnet=True
            )
            account = self.client.get_account()
            self.console.print("✅ [green]Binance Testnet verbunden![/green]")
        except Exception as e:
            self.console.print(f"❌ [red]API-Fehler: {e}[/red]")
            self.client = None
        
        # AI Signal Generator
        self.signal_generator = SimpleTradingSignals()
        
        # Trading Stats
        self.trades_executed = 0
        self.total_volume = 0.0
        self.start_time = datetime.now()
    
    def get_live_balance(self) -> Dict[str, float]:
        """💰 Live-Guthaben von Binance abrufen"""
        if not self.client:
            return {}
        
        try:
            account = self.client.get_account()
            balances = {}
            
            # Nur relevante Assets mit Guthaben > 0
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0 and asset in ['USDT', 'BTC', 'ETH', 'BNB', 'SOL', 'ADA']:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            
            return balances
        except Exception as e:
            self.console.print(f"❌ [red]Fehler beim Abrufen der Bilanz: {e}[/red]")
            return {}
    
    def get_live_prices(self) -> Dict[str, float]:
        """💲 Live-Preise von Binance abrufen"""
        if not self.client:
            return {}
        
        try:
            prices = {}
            for symbol in TRADING_SYMBOLS:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
            return prices
        except Exception as e:
            self.console.print(f"❌ [red]Fehler beim Abrufen der Preise: {e}[/red]")
            return {}
    
    async def execute_live_trade(self, symbol: str, side: str, quantity: float) -> bool:
        """📈 Echten Trade auf Testnet ausführen"""
        if not self.client:
            self.console.print("❌ [red]Kein API-Client verfügbar[/red]")
            return False
        
        try:
            # Kleine Testmenge für Demo
            min_qty = 0.001  # Minimale Testmenge
            test_quantity = min(quantity, min_qty)
            
            self.console.print(f"🔄 [yellow]Führe {side} Order aus: {test_quantity:.6f} {symbol}[/yellow]")
            
            # Market Order auf Testnet
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=test_quantity
            )
            
            # Erfolgreiche Order
            self.console.print(f"✅ [green]Order erfolgreich! ID: {order['orderId']}[/green]")
            self.console.print(f"📊 [green]Status: {order['status']} | Executed: {order['executedQty']}[/green]")
            
            # Stats aktualisieren
            self.trades_executed += 1
            self.total_volume += test_quantity
            
            return True
            
        except BinanceAPIException as e:
            self.console.print(f"❌ [red]Binance API Fehler: {e.message}[/red]")
            return False
        except Exception as e:
            self.console.print(f"❌ [red]Unbekannter Fehler: {e}[/red]")
            return False
    
    def display_live_dashboard(self, signals: List[Dict], balances: Dict, prices: Dict):
        """📊 Live-Dashboard anzeigen"""
        self.console.clear()
        self.console.print(Panel.fit("🚀 Live Trading Demo - Binance Testnet", style="bold magenta"))
        
        # Account Info
        account_table = Table(title="💰 Live Account Balance", box=box.SIMPLE)
        account_table.add_column("Asset", style="cyan")
        account_table.add_column("Free", style="green")
        account_table.add_column("Locked", style="yellow")
        account_table.add_column("Total", style="white")
        
        for asset, data in balances.items():
            account_table.add_row(
                asset,
                f"{data['free']:,.6f}",
                f"{data['locked']:,.6f}",
                f"{data['total']:,.6f}"
            )
        
        self.console.print(account_table)
        
        # Live Preise
        price_table = Table(title="💲 Live Prices (Binance)", box=box.SIMPLE)
        price_table.add_column("Symbol", style="cyan")
        price_table.add_column("Price", style="green")
        price_table.add_column("Change 24h", style="white")
        
        for symbol, price in prices.items():
            # Vereinfachte 24h Änderung (würde normalerweise von API kommen)
            change_24h = "+1.23%"  # Mock-Daten
            change_color = "green" if change_24h.startswith("+") else "red"
            
            price_table.add_row(
                symbol,
                f"${price:,.4f}",
                f"[{change_color}]{change_24h}[/{change_color}]"
            )
        
        self.console.print(price_table)
        
        # AI Signals
        if signals:
            signal_table = Table(title="🧠 AI Trading Signals", box=box.SIMPLE)
            signal_table.add_column("Symbol", style="cyan")
            signal_table.add_column("Signal", style="white")
            signal_table.add_column("Confidence", style="green")
            signal_table.add_column("Expected", style="yellow")
            
            for signal in signals[:3]:  # Top 3 Signale
                symbol_base = signal['symbol'].replace('COIN', '') + 'USDT'
                if symbol_base in TRADING_SYMBOLS:
                    signal_color = "green" if signal['signal'] in ['STRONG_BUY', 'BUY'] else "red"
                    signal_icon = "🚀" if signal['signal'] in ['STRONG_BUY', 'BUY'] else "📉"
                    
                    signal_table.add_row(
                        symbol_base,
                        f"[{signal_color}]{signal_icon} {signal['signal']}[/{signal_color}]",
                        f"{signal['confidence']:.1%}",
                        f"{signal['price_change_pct']:+.2f}%"
                    )
            
            self.console.print(signal_table)
        
        # Trading Stats
        runtime = datetime.now() - self.start_time
        stats_table = Table(title="📊 Trading Statistics", box=box.SIMPLE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("🕐 Runtime", str(runtime).split('.')[0])
        stats_table.add_row("📈 Trades Executed", str(self.trades_executed))
        stats_table.add_row("💹 Total Volume", f"{self.total_volume:.6f}")
        stats_table.add_row("🔄 Status", "🟢 Live Trading Active")
        
        self.console.print(stats_table)
    
    async def run_live_demo(self):
        """🚀 Live Trading Demo starten"""
        self.console.print("🚀 [green]Live Trading Demo gestartet![/green]")
        self.console.print("⚠️  [yellow]Verwendet Binance Testnet - kein echtes Geld![/yellow]")
        self.console.print("🔄 [cyan]Demo läuft bis mindestens ein Trade ausgeführt wurde![/cyan]\n")
        
        for cycle in range(10):  # 10 Demo-Zyklen für mehr Trading-Chancen
            try:
                self.console.print(f"🔄 [cyan]Demo-Zyklus {cycle + 1}/5[/cyan]")
                
                # 1. Live-Daten abrufen
                balances = self.get_live_balance()
                prices = self.get_live_prices()
                
                # 2. AI-Signale generieren
                self.console.print("🧠 [yellow]Generiere AI-Signale...[/yellow]")
                signals = await self.signal_generator.generate_signals()
                
                # 3. Dashboard aktualisieren
                self.display_live_dashboard(signals, balances, prices)
                
                # 4. Trading-Entscheidung
                trade_executed = False
                
                if signals:
                    best_signal = signals[0]
                    symbol_base = best_signal['symbol'].replace('COIN', '') + 'USDT'
                    
                    # Gelockerte Bedingungen für Demo - garantiert mindestens einen Trade
                    should_trade = (
                        symbol_base in TRADING_SYMBOLS and 
                        best_signal['confidence'] > 0.70 and  # Reduziert von 0.8 auf 0.7
                        best_signal['signal'] in ['STRONG_BUY', 'BUY', 'HOLD']
                    )
                    
                    # Nach 3 Zyklen: Erzwinge Trade mit dem besten verfügbaren Signal
                    if cycle >= 3 and self.trades_executed == 0:
                        should_trade = True
                        self.console.print(f"\n🔥 [red]DEMO-MODUS: Erzwinge Trade nach {cycle + 1} Zyklen![/red]")
                    
                    if should_trade:
                        self.console.print(f"\n🎯 [green]Trading-Signal erkannt: {symbol_base}[/green]")
                        self.console.print(f"📊 Signal: {best_signal['signal']} ({best_signal['confidence']:.1%})")
                        self.console.print(f"💰 Erwartete Änderung: {best_signal['price_change_pct']:+.2f}%")
                        
                        # Demo-Trade ausführen
                        success = await self.execute_live_trade(symbol_base, SIDE_BUY, 0.001)
                        
                        if success:
                            self.console.print("✅ [green]Demo-Trade erfolgreich ausgeführt![/green]")
                            trade_executed = True
                            
                            # Zeige Trade-Details
                            self.console.print("\n📈 [cyan]Trade-Details:[/cyan]")
                            self.console.print(f"🎯 Symbol: {symbol_base}")
                            self.console.print(f"💰 Menge: 0.001")
                            self.console.print(f"📊 AI-Konfidenz: {best_signal['confidence']:.1%}")
                            self.console.print(f"🎲 Basiert auf: {best_signal['signal']}")
                            
                            # Demo nach erfolgreichem Trade beenden
                            self.console.print("\n🎉 [bold green]TRADE ERFOLGREICH! Demo kann beendet werden.[/bold green]")
                            break
                        else:
                            self.console.print("❌ [red]Demo-Trade fehlgeschlagen[/red]")
                
                # Falls kein Trade nach 5 Zyklen: Verwende Bitcoin als Fallback
                if cycle >= 4 and self.trades_executed == 0:
                    self.console.print("\n🚨 [yellow]FALLBACK: Führe Bitcoin-Trade aus...[/yellow]")
                    success = await self.execute_live_trade('BTCUSDT', SIDE_BUY, 0.001)
                    if success:
                        self.console.print("✅ [green]Fallback Bitcoin-Trade erfolgreich![/green]")
                        break
                
                # 5. Warten bis zum nächsten Zyklus
                if cycle < 4:  # Nicht beim letzten Zyklus warten
                    self.console.print(f"\n⏳ [dim]Warte 30 Sekunden bis zum nächsten Zyklus...[/dim]")
                    await asyncio.sleep(30)
                
            except Exception as e:
                self.console.print(f"❌ [red]Fehler in Demo-Zyklus: {e}[/red]")
        
        # Demo beendet
        self.console.print("\n🏁 [green]Live Trading Demo beendet![/green]")
        self.console.print(f"📊 [cyan]Insgesamt {self.trades_executed} Trades ausgeführt[/cyan]")
        self.console.print("🎯 [yellow]Nächster Schritt: Vollautomatischer Bot oder Live Trading[/yellow]")


async def main():
    """🎯 Hauptfunktion"""
    console = Console()
    
    if not BINANCE_AVAILABLE:
        console.print("❌ [red]Binance Library fehlt. Installiere mit: pip install python-binance[/red]")
        return
    
    if not API_KEY or API_KEY == "dein_testnet_api_key_hier":
        console.print("❌ [red]API-Keys nicht konfiguriert. Prüfe config.py[/red]")
        return
    
    # Live Trading Demo starten
    demo = LiveTradingDemo()
    await demo.run_live_demo()


if __name__ == "__main__":
    asyncio.run(main())
