#!/usr/bin/env python3
"""
🔐 Secure Trading Bot mit API-Key Integration
Autor: mad4cyber
Version: 1.0 - Secure Edition

🚀 FEATURES:
- Sichere API-Key Verwaltung
- Binance Testnet Integration
- Vollständiges Risk Management
- Portfolio-Tracking
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

# Sichere API-Key Verwaltung
try:
    import config
    TESTNET_API_KEY = config.BINANCE_TESTNET_API_KEY
    TESTNET_SECRET = config.BINANCE_TESTNET_SECRET
    TRADING_SYMBOLS = config.TRADING_SYMBOLS
    INITIAL_BALANCE = config.INITIAL_BALANCE
    print("✅ Konfiguration geladen")
except ImportError:
    print("❌ config.py nicht gefunden!")
    print("💡 Kopiere config_template.py zu config.py und fülle deine API-Keys ein")
    TESTNET_API_KEY = None
    TESTNET_SECRET = None
    TRADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    INITIAL_BALANCE = 10000.0

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

from trading_signals_demo import SimpleTradingSignals

class TradingMode(Enum):
    PAPER = "paper"
    TESTNET = "testnet" 
    LIVE = "live"

class SecureTradingBot:
    """🔐 Sicherer Trading Bot mit API-Integration"""
    
    def __init__(self, mode: TradingMode = TradingMode.PAPER):
        self.console = Console()
        self.mode = mode
        
        # Binance Client Setup
        self.client = None
        if mode == TradingMode.TESTNET and TESTNET_API_KEY and TESTNET_SECRET:
            try:
                self.client = Client(
                    api_key=TESTNET_API_KEY,
                    api_secret=TESTNET_SECRET,
                    testnet=True
                )
                # API-Verbindung testen
                account = self.client.get_account()
                self.console.print("✅ [green]Binance Testnet verbunden![/green]")
                self.console.print(f"📊 Account Status: {account['accountType']}")
            except Exception as e:
                self.console.print(f"❌ [red]Testnet Verbindung fehlgeschlagen: {e}[/red]")
                self.mode = TradingMode.PAPER
        
        # Signal Generator
        self.signal_generator = SimpleTradingSignals()
        
        # Portfolio
        self.balance = INITIAL_BALANCE
        self.positions = {}
        self.trades = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SecureTradingBot')
    
    def get_testnet_balance(self) -> Dict[str, float]:
        """💰 Testnet-Guthaben abrufen"""
        if not self.client:
            return {"USDT": self.balance}
        
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                free = float(balance['free'])
                if free > 0:
                    balances[balance['asset']] = free
            
            return balances
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Testnet-Bilanz: {e}")
            return {"USDT": self.balance}
    
    def get_current_prices(self) -> Dict[str, float]:
        """💲 Aktuelle Preise abrufen"""
        if not self.client:
            # Fallback-Preise
            return {
                "BTCUSDT": 90000.0,
                "ETHUSDT": 2100.0,
                "BNBUSDT": 550.0,
                "ADAUSDT": 0.50,
                "SOLUSDT": 125.0
            }
        
        try:
            prices = {}
            for symbol in TRADING_SYMBOLS:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                prices[symbol] = float(ticker['price'])
            return prices
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Preise: {e}")
            return {}
    
    async def place_testnet_order(self, symbol: str, side: str, amount: float) -> bool:
        """📈 Testnet-Order platzieren"""
        if not self.client:
            self.console.print(f"📝 [yellow]PAPER: {side} {amount:.6f} {symbol}[/yellow]")
            return True
        
        try:
            # Order auf Testnet platzieren
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=amount
            )
            
            self.console.print(f"✅ [green]TESTNET: {side} Order platziert - ID: {order['orderId']}[/green]")
            return True
            
        except BinanceAPIException as e:
            self.console.print(f"❌ [red]Testnet Order Fehler: {e.message}[/red]")
            return False
        except Exception as e:
            self.console.print(f"❌ [red]Unbekannter Fehler: {e}[/red]")
            return False
    
    def display_account_info(self):
        """📊 Account-Informationen anzeigen"""
        self.console.print(Panel.fit("🔐 Secure Trading Bot Dashboard", style="bold magenta"))
        
        # Account Info
        info_table = Table(title="📊 Account Information", box=box.SIMPLE)
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Wert", style="white")
        
        info_table.add_row("🔧 Modus", f"{self.mode.value.upper()}")
        info_table.add_row("🌐 API Status", "✅ Verbunden" if self.client else "📝 Paper Trading")
        
        # Guthaben
        balances = self.get_testnet_balance()
        for asset, amount in balances.items():
            if amount > 0:
                info_table.add_row(f"💰 {asset}", f"{amount:,.6f}")
        
        self.console.print(info_table)
        
        # Aktuelle Preise
        prices = self.get_current_prices()
        if prices:
            price_table = Table(title="💲 Aktuelle Preise", box=box.SIMPLE)
            price_table.add_column("Symbol", style="cyan")
            price_table.add_column("Preis", style="green")
            
            for symbol, price in prices.items():
                price_table.add_row(symbol, f"${price:,.4f}")
            
            self.console.print(price_table)
    
    async def run_test_trade(self):
        """🧪 Test-Trade durchführen"""
        self.console.print("\n🧪 [cyan]Führe Test-Trade durch...[/cyan]")
        
        # Generiere AI-Signale
        signals = await self.signal_generator.generate_signals()
        
        if signals:
            # Nimm das beste Signal
            best_signal = signals[0]
            symbol = best_signal['symbol'].replace('COIN', '') + 'USDT'
            
            if symbol in TRADING_SYMBOLS:
                self.console.print(f"\n🎯 Test-Trade für {symbol}")
                self.console.print(f"📊 Signal: {best_signal['signal']}")
                self.console.print(f"🎯 Konfidenz: {best_signal['confidence']:.1%}")
                
                # Kleine Test-Order (0.001 für Testnet)
                if best_signal['signal'] in ['STRONG_BUY', 'BUY']:
                    test_amount = 0.001  # Sehr kleine Menge für Tests
                    
                    success = await self.place_testnet_order(symbol, SIDE_BUY, test_amount)
                    
                    if success:
                        self.console.print("✅ [green]Test-Trade erfolgreich![/green]")
                    else:
                        self.console.print("❌ [red]Test-Trade fehlgeschlagen[/red]")
        else:
            self.console.print("❌ [red]Keine Signale verfügbar für Test-Trade[/red]")


def check_setup():
    """🔍 Setup-Überprüfung"""
    console = Console()
    
    console.print(Panel.fit("🔧 Trading Bot Setup Check", style="bold blue"))
    
    # Prüfe Konfiguration
    if TESTNET_API_KEY and TESTNET_API_KEY != "dein_testnet_api_key_hier":
        console.print("✅ [green]API-Keys konfiguriert[/green]")
    else:
        console.print("❌ [red]API-Keys nicht konfiguriert[/red]")
        console.print("💡 [yellow]Folge diesen Schritten:[/yellow]")
        console.print("1. Gehe zu https://testnet.binance.vision/")
        console.print("2. Erstelle einen Account")
        console.print("3. Generiere API-Keys")
        console.print("4. Kopiere config_template.py zu config.py")
        console.print("5. Füge deine API-Keys in config.py ein")
        return False
    
    # Prüfe Binance Library
    if BINANCE_AVAILABLE:
        console.print("✅ [green]Binance Library verfügbar[/green]")
    else:
        console.print("❌ [red]Binance Library fehlt[/red]")
        console.print("💡 [yellow]Installiere mit: pip install python-binance[/yellow]")
        return False
    
    return True


async def main():
    """🎯 Hauptfunktion"""
    console = Console()
    
    console.print(Panel.fit("🔐 Secure Trading Bot", style="bold magenta"))
    console.print("⚠️  [yellow]Testnet-Modus für sichere Tests![/yellow]\n")
    
    # Setup prüfen
    if not check_setup():
        return
    
    # Modus wählen
    console.print("🔧 Wähle Trading-Modus:")
    console.print("1. Paper Trading (Simulation)")
    console.print("2. Testnet (Binance Testnet mit API-Keys)")
    
    choice = input("\nWähle (1-2): ").strip()
    
    if choice == "2" and TESTNET_API_KEY:
        mode = TradingMode.TESTNET
    else:
        mode = TradingMode.PAPER
    
    # Trading Bot starten
    bot = SecureTradingBot(mode)
    
    console.print(f"\n🚀 Trading Bot im {mode.value.upper()} Modus gestartet!")
    
    # Dashboard anzeigen
    bot.display_account_info()
    
    # Test-Trade Optionen
    console.print("\n🧪 [cyan]Test-Optionen:[/cyan]")
    console.print("1. Account-Info anzeigen")
    console.print("2. Test-Trade durchführen") 
    console.print("3. Preise abrufen")
    
    while True:
        choice = input("\nWähle Option (1-3, 'q' zum Beenden): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            bot.display_account_info()
        elif choice == '2':
            await bot.run_test_trade()
        elif choice == '3':
            prices = bot.get_current_prices()
            for symbol, price in prices.items():
                console.print(f"{symbol}: ${price:,.4f}")
        else:
            console.print("❌ Ungültige Auswahl")
    
    console.print("\n👋 Trading Bot beendet.")


if __name__ == "__main__":
    asyncio.run(main())
