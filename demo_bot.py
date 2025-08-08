#!/usr/bin/env python3
"""
🚀 Demo Trading Bot - Ohne API-Keys
Autor: mad4cyber
Version: 1.0 - Demo Edition

🎯 FEATURES:
- Keine API-Keys benötigt
- Simulation Trading Signale
- Live Crypto-Daten
- Portfolio-Simulation
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from pycoingecko import CoinGeckoAPI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

class DemoTradingBot:
    """🚀 Demo Trading Bot ohne API-Keys"""
    
    def __init__(self, balance: float = 10000):
        self.console = Console()
        self.cg = CoinGeckoAPI()
        self.balance = balance
        self.start_balance = balance
        self.positions = {}
        self.trade_history = []
        
        # Demo Trading Pairs
        self.symbols = ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana']
        
    def get_crypto_data(self):
        """Hole aktuelle Crypto-Daten"""
        try:
            data = self.cg.get_price(
                ids=self.symbols,
                vs_currencies='usd',
                include_24hr_change=True,
                include_24hr_vol=True
            )
            return data
        except Exception as e:
            self.console.print(f"❌ API-Fehler: {e}", style="red")
            return None
    
    def generate_signal(self, symbol: str, data: dict) -> str:
        """Generiere Trading-Signal basierend auf Preisänderung"""
        change_24h = data.get(f'{symbol}_24h_change', 0)
        
        if change_24h > 5:
            return "STRONG_BUY"
        elif change_24h > 2:
            return "BUY"
        elif change_24h < -5:
            return "STRONG_SELL"
        elif change_24h < -2:
            return "SELL"
        else:
            return "HOLD"
    
    def simulate_trade(self, symbol: str, signal: str, price: float):
        """Simuliere Trade-Ausführung"""
        trade_amount = self.balance * 0.1  # 10% des Balances
        
        if signal in ["BUY", "STRONG_BUY"] and self.balance > trade_amount:
            # Kaufe
            coins = trade_amount / price
            self.positions[symbol] = self.positions.get(symbol, 0) + coins
            self.balance -= trade_amount
            
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'action': signal,
                'price': price,
                'amount': coins,
                'value': trade_amount
            })
            
        elif signal in ["SELL", "STRONG_SELL"] and symbol in self.positions and self.positions[symbol] > 0:
            # Verkaufe
            coins = self.positions[symbol]
            trade_value = coins * price
            self.balance += trade_value
            self.positions[symbol] = 0
            
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'action': signal,
                'price': price,
                'amount': coins,
                'value': trade_value
            })
    
    def create_dashboard(self, crypto_data: dict) -> Table:
        """Erstelle Trading-Dashboard"""
        table = Table(title="🚀 Demo Trading Bot Dashboard", box=box.ROUNDED)
        table.add_column("Coin", style="cyan")
        table.add_column("Price", justify="right", style="green")
        table.add_column("24h Change", justify="right")
        table.add_column("Signal", justify="center")
        table.add_column("Position", justify="right", style="yellow")
        
        for symbol in self.symbols:
            if symbol in crypto_data:
                data = crypto_data[symbol]
                price = data['usd']
                change_24h = data.get(f'{symbol}_24h_change', 0)
                signal = self.generate_signal(symbol, crypto_data[symbol])
                position = self.positions.get(symbol, 0)
                
                # Farbe für Preisänderung
                change_color = "green" if change_24h >= 0 else "red"
                
                # Signal Emoji
                signal_emoji = {
                    "STRONG_BUY": "🟢 STRONG BUY",
                    "BUY": "🔵 BUY",
                    "HOLD": "⚪ HOLD",
                    "SELL": "🟠 SELL",
                    "STRONG_SELL": "🔴 STRONG SELL"
                }.get(signal, "⚪ HOLD")
                
                table.add_row(
                    symbol.upper(),
                    f"${price:.4f}",
                    f"[{change_color}]{change_24h:+.2f}%[/{change_color}]",
                    signal_emoji,
                    f"{position:.4f}" if position > 0 else "0"
                )
                
                # Simuliere Trade
                self.simulate_trade(symbol, signal, price)
        
        return table
    
    def create_portfolio_panel(self, crypto_data: dict) -> Panel:
        """Erstelle Portfolio-Panel"""
        portfolio_value = self.balance
        
        # Berechne Portfolio-Wert
        for symbol, coins in self.positions.items():
            if coins > 0 and symbol in crypto_data:
                price = crypto_data[symbol]['usd']
                portfolio_value += coins * price
        
        profit_loss = portfolio_value - self.start_balance
        profit_pct = (profit_loss / self.start_balance) * 100
        
        color = "green" if profit_loss >= 0 else "red"
        
        portfolio_info = f"""
💰 Portfolio-Status:
├─ Startkapital: ${self.start_balance:,.2f}
├─ Cash Balance: ${self.balance:,.2f}
├─ Portfolio Wert: ${portfolio_value:,.2f}
└─ [{color}]P&L: ${profit_loss:+,.2f} ({profit_pct:+.2f}%)[/{color}]

🔄 Aktive Positionen: {sum(1 for p in self.positions.values() if p > 0)}
📊 Trades heute: {len(self.trade_history)}
        """
        
        return Panel(portfolio_info, title="💼 Portfolio", box=box.ROUNDED)
    
    async def run_demo_bot(self):
        """Starte Demo Bot mit Live-Updates"""
        with Live(auto_refresh=False) as live:
            while True:
                try:
                    # Hole aktuelle Daten
                    crypto_data = self.get_crypto_data()
                    if not crypto_data:
                        await asyncio.sleep(10)
                        continue
                    
                    # Erstelle Dashboard
                    dashboard = self.create_dashboard(crypto_data)
                    portfolio = self.create_portfolio_panel(crypto_data)
                    
                    # Layout erstellen
                    layout = f"{dashboard}\n\n{portfolio}"
                    
                    live.update(layout, refresh=True)
                    
                    # Warte 10 Sekunden
                    await asyncio.sleep(10)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"❌ Fehler: {e}", style="red")
                    await asyncio.sleep(5)

async def main():
    """🚀 Hauptfunktion"""
    console = Console()
    
    console.print(Panel.fit("🚀 Demo Trading Bot", style="bold blue"))
    console.print("💰 Startkapital: $10,000")
    console.print("📊 Coins: BTC, ETH, BNB, XRP, SOL")
    console.print("⚠️  SIMULATION - Kein echtes Geld!")
    console.print("🔄 Updates alle 10 Sekunden")
    console.print("\nDrücke CTRL+C zum Beenden\n")
    
    # Automatischer Start für Demo
    await asyncio.sleep(1)
    
    bot = DemoTradingBot(balance=10000)
    
    try:
        await bot.run_demo_bot()
    except KeyboardInterrupt:
        console.print("\n👋 Demo Bot beendet!")
        
        # Finale Statistiken
        final_balance = bot.balance
        for symbol, coins in bot.positions.items():
            if coins > 0:
                # Geschätzter aktueller Preis (vereinfacht)
                final_balance += coins * 50000  # Dummy-Preis
        
        profit = final_balance - bot.start_balance
        console.print(f"\n📊 Finale Statistiken:")
        console.print(f"💰 Endkapital: ${final_balance:,.2f}")
        console.print(f"📈 Gewinn/Verlust: ${profit:+,.2f}")
        console.print(f"📊 Trades: {len(bot.trade_history)}")

if __name__ == "__main__":
    asyncio.run(main())