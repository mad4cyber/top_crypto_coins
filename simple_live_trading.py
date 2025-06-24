#!/usr/bin/env python3
"""
🚀 Simple Live Trading Bot - Production Ready
Autor: mad4cyber
Version: 1.0 - Simplified Edition

🎯 SOFORT EINSATZBEREIT:
- Optimierte Parameter
- Sicherheitschecks
- Live Trading Ready
"""

import asyncio
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import config
    from trading_parameters import *
    from binance.client import Client
    from binance.enums import SIDE_BUY
    from trading_signals_demo import SimpleTradingSignals
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Installiere: pip install python-binance")
    exit(1)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class SimpleLiveTrading:
    """🚀 Vereinfachter Live Trading Bot"""
    
    def __init__(self, mode="testnet"):
        self.console = Console()
        self.mode = mode
        
        # Conservative Parameters für Live Trading
        self.params = LIVE_DEMO_PARAMS if mode == "live_demo" else CONSERVATIVE_PARAMS
        
        # API Setup
        if mode == "testnet":
            self.client = Client(
                api_key=config.BINANCE_TESTNET_API_KEY,
                api_secret=config.BINANCE_TESTNET_SECRET,
                testnet=True
            )
            self.console.print("✅ [green]Testnet verbunden[/green]")
        elif mode == "live_demo":
            # Live API (wenn konfiguriert)
            if hasattr(config, 'BINANCE_LIVE_API_KEY') and config.BINANCE_LIVE_API_KEY != "dein_live_api_key_hier":
                self.client = Client(
                    api_key=config.BINANCE_LIVE_API_KEY,
                    api_secret=config.BINANCE_LIVE_SECRET
                )
                self.console.print("⚠️ [red]LIVE API VERBUNDEN![/red]")
            else:
                self.console.print("❌ Live API nicht konfiguriert - verwende Testnet")
                self.mode = "testnet"
                self.__init__("testnet")
                return
        
        self.signal_generator = SimpleTradingSignals()
        self.trades_today = 0
        self.start_balance = self.get_balance()
    
    def get_balance(self) -> float:
        """💰 USDT Balance abrufen"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.console.print(f"❌ Balance Error: {e}")
            return 0.0
    
    def safety_check(self) -> bool:
        """🛡️ Grundlegende Sicherheitschecks"""
        balance = self.get_balance()
        
        # Balance Check
        if self.mode == "live_demo" and balance < 50:
            self.console.print("⚠️ [red]Balance zu niedrig für Live Demo (<$50)[/red]")
            return False
        
        # Daily Trade Limit
        if self.trades_today >= self.params.max_daily_trades:
            self.console.print("⚠️ [yellow]Daily Trade Limit erreicht[/yellow]")
            return False
        
        return True
    
    async def execute_safe_trade(self, symbol: str, signal: Dict) -> bool:
        """🔒 Sicherer Trade mit allen Checks"""
        try:
            # Safety Check
            if not self.safety_check():
                return False
            
            # Position Size berechnen
            balance = self.get_balance()
            position_usd = calculate_optimal_position_size(
                balance, signal['confidence'], signal['price_change_pct'], symbol, self.params
            )
            
            # Minimum Position Check
            min_pos = 10.0 if self.mode == "live_demo" else 5.0
            if position_usd < min_pos:
                self.console.print(f"⚠️ Position zu klein: ${position_usd:.2f}")
                return False
            
            # Live Trading Confirmation
            if self.mode == "live_demo":
                self.console.print(f"\\n🔥 [bold red]LIVE TRADE CONFIRMATION:[/bold red]")
                self.console.print(f"💰 {symbol}: ${position_usd:.2f}")
                self.console.print(f"🎯 Confidence: {signal['confidence']:.1%}")
                
                confirm = input("Bestätige LIVE TRADE (ja/nein): ").lower()
                if confirm != "ja":
                    self.console.print("❌ Trade abgebrochen")
                    return False
            
            # Order ausführen
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            quantity = position_usd / current_price
            
            if quantity >= 0.001:  # Binance Minimum
                order = self.client.order_market(
                    symbol=symbol,
                    side=SIDE_BUY,
                    quantity=quantity
                )
                
                self.console.print(f"✅ [green]Trade erfolgreich: {order['orderId']}[/green]")
                self.trades_today += 1
                return True
            else:
                self.console.print("⚠️ Quantity zu klein")
                return False
                
        except Exception as e:
            self.console.print(f"❌ Trade Error: {e}")
            return False
    
    async def run_simple_trading(self, duration_minutes: int = 30):
        """🚀 Einfacher Trading Loop"""
        self.console.print(Panel.fit(f"🚀 Simple Live Trading - {self.mode.upper()}", style="bold magenta"))
        
        # Start Info
        self.console.print(f"💰 Start Balance: ${self.start_balance:,.2f}")
        self.console.print(f"⏰ Trading Duration: {duration_minutes} Minuten")
        self.console.print(f"🎯 Parameters: {self.params.max_risk_per_trade:.1%} Risk, {self.params.min_ai_confidence:.1%} AI Confidence")
        
        if self.mode == "live_demo":
            self.console.print("\\n⚠️ [bold red]LIVE TRADING MIT ECHTEM GELD AKTIV![/bold red]")
        
        cycles = 0
        max_cycles = duration_minutes // 5  # Alle 5 Minuten ein Zyklus
        
        try:
            while cycles < max_cycles:
                cycles += 1
                self.console.print(f"\\n🔄 [cyan]Zyklus {cycles}/{max_cycles}[/cyan]")
                
                # AI Signale generieren
                self.console.print("🧠 Generiere AI-Signale...")
                signals = await self.signal_generator.generate_signals()
                
                if signals:
                    # Beste Signale filtern
                    good_signals = [
                        s for s in signals 
                        if s['confidence'] >= self.params.min_ai_confidence
                        and abs(s['price_change_pct']) >= self.params.min_price_change_threshold
                        and s['signal'] in ['STRONG_BUY', 'BUY']
                    ]
                    
                    if good_signals:
                        best = good_signals[0]
                        symbol = best['symbol'].replace('COIN', '') + 'USDT'
                        
                        if symbol in config.TRADING_SYMBOLS:
                            self.console.print(f"🎯 [green]Signal: {symbol} {best['signal']} ({best['confidence']:.1%})[/green]")
                            
                            success = await self.execute_safe_trade(symbol, best)
                            if success:
                                self.console.print("✅ Trade erfolgreich!")
                                await asyncio.sleep(300)  # 5 Min Pause nach Trade
                    else:
                        self.console.print("⏸️ Keine starken Signale")
                else:
                    self.console.print("❌ Keine AI-Signale")
                
                # Warten bis nächster Zyklus
                if cycles < max_cycles:
                    self.console.print("⏳ Warte 5 Minuten...")
                    await asyncio.sleep(300)  # 5 Minuten
        
        except KeyboardInterrupt:
            self.console.print("\\n🛑 [yellow]Trading gestoppt[/yellow]")
        
        # Final Stats
        final_balance = self.get_balance()
        pnl = final_balance - self.start_balance
        
        self.console.print("\\n📊 [green]Trading Session beendet[/green]")
        self.console.print(f"💰 End Balance: ${final_balance:,.2f}")
        self.console.print(f"📈 P&L: ${pnl:+,.2f} ({(pnl/self.start_balance)*100:+.2f}%)")
        self.console.print(f"📊 Trades: {self.trades_today}")


def main():
    """🎯 Hauptfunktion"""
    console = Console()
    
    console.print(Panel.fit("🚀 Simple Live Trading Setup", style="bold magenta"))
    console.print("\\n🔧 Wähle Modus:")
    console.print("1. Testnet (Sicher, Fake-Geld)")
    console.print("2. Live Demo (Echtes Geld, kleine Beträge)")
    
    choice = input("\\nWähle (1-2): ").strip()
    
    if choice == "2":
        console.print("\\n⚠️ [bold red]WARNUNG: Live Trading mit echtem Geld![/bold red]")
        confirm = input("Fortfahren? (ja/nein): ").lower()
        if confirm != "ja":
            console.print("❌ Abgebrochen")
            return
        mode = "live_demo"
    else:
        mode = "testnet"
    
    duration = int(input("\\nTrading-Dauer (Minuten, default 30): ") or "30")
    
    # Trading starten
    bot = SimpleLiveTrading(mode)
    asyncio.run(bot.run_simple_trading(duration))


if __name__ == "__main__":
    main()
