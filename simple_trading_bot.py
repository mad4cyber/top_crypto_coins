#!/usr/bin/env python3
"""
🤖 Einfacher AI Trading Bot - Fehlerfrei
Autor: mad4cyber
Version: 4.2 - Simplified Edition
"""

import asyncio
import time
from datetime import datetime
from pycoingecko import CoinGeckoAPI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

class SimpleTradeSignal:
    def __init__(self, symbol: str, action: str, price: float, reason: str):
        self.symbol = symbol
        self.action = action  # BUY, SELL, HOLD
        self.price = price
        self.reason = reason
        self.timestamp = datetime.now()

class SimpleTradingBot:
    """🤖 Einfacher Trading Bot ohne komplexe AI"""
    
    def __init__(self, balance: float = 10000):
        self.console = Console()
        self.cg = CoinGeckoAPI()
        self.balance = balance
        self.initial_balance = balance
        self.positions = {}
        self.trades = []
        self.watchlist = ['bitcoin', 'ethereum', 'binancecoin']
        
    def get_simple_signal(self, coin_id: str) -> SimpleTradeSignal:
        """📊 Einfache Trading-Signale basierend auf 24h-Änderung"""
        try:
            data = self.cg.get_coins_markets(
                vs_currency='eur',
                ids=coin_id,
                order='market_cap',
                per_page=1,
                page=1,
                sparkline=False,
                price_change_percentage='24h'
            )
            
            if not data:
                return SimpleTradeSignal(coin_id.upper(), "HOLD", 0, "Keine Daten")
                
            coin = data[0]
            symbol = coin['symbol'].upper()
            price = coin['current_price']
            change_24h = coin.get('price_change_percentage_24h', 0) or 0
            
            # Einfache Trading-Logik
            if change_24h > 5:  # Starker Anstieg
                return SimpleTradeSignal(symbol, "BUY", price, f"📈 Starker Anstieg: +{change_24h:.1f}%")
            elif change_24h < -5:  # Starker Rückgang
                return SimpleTradeSignal(symbol, "SELL", price, f"📉 Starker Rückgang: {change_24h:.1f}%")
            else:
                return SimpleTradeSignal(symbol, "HOLD", price, f"⚖️ Neutral: {change_24h:.1f}%")
                
        except Exception as e:
            return SimpleTradeSignal(coin_id.upper(), "HOLD", 0, f"Fehler: {e}")
    
    def execute_simple_trade(self, signal: SimpleTradeSignal) -> bool:
        """💱 Einfache Trade-Ausführung"""
        try:
            if signal.action == "BUY" and self.balance > signal.price:
                # Kaufe für 10% des verfügbaren Guthabens
                investment = min(self.balance * 0.1, self.balance)
                amount = investment / signal.price
                
                if signal.symbol in self.positions:
                    self.positions[signal.symbol] += amount
                else:
                    self.positions[signal.symbol] = amount
                    
                self.balance -= investment
                
                trade = {
                    'time': signal.timestamp,
                    'action': 'BUY',
                    'symbol': signal.symbol,
                    'amount': amount,
                    'price': signal.price,
                    'cost': investment
                }
                self.trades.append(trade)
                return True
                
            elif signal.action == "SELL" and signal.symbol in self.positions and self.positions[signal.symbol] > 0:
                # Verkaufe die gesamte Position
                amount = self.positions[signal.symbol]
                revenue = amount * signal.price
                
                self.positions[signal.symbol] = 0
                self.balance += revenue
                
                trade = {
                    'time': signal.timestamp,
                    'action': 'SELL',
                    'symbol': signal.symbol,
                    'amount': amount,
                    'price': signal.price,
                    'revenue': revenue
                }
                self.trades.append(trade)
                return True
                
            return False
            
        except Exception as e:
            self.console.print(f"❌ Trade-Fehler: {e}")
            return False
    
    def create_status_display(self, signals: list) -> None:
        """📊 Status-Anzeige erstellen"""
        # Trading Signals
        signals_table = Table(title="🧠 Trading Signals", box=box.ROUNDED)
        signals_table.add_column("Symbol", style="cyan")
        signals_table.add_column("Aktion", style="white")
        signals_table.add_column("Preis", style="yellow")
        signals_table.add_column("Grund", style="dim")
        
        for signal in signals:
            action_color = "green" if signal.action == "BUY" else "red" if signal.action == "SELL" else "yellow"
            action_icon = "🚀" if signal.action == "BUY" else "📉" if signal.action == "SELL" else "⏸️"
            
            signals_table.add_row(
                signal.symbol,
                f"[{action_color}]{action_icon} {signal.action}[/{action_color}]",
                f"€{signal.price:.2f}" if signal.price > 0 else "N/A",
                signal.reason
            )
        
        self.console.print(signals_table)
        
        # Portfolio Status
        total_portfolio_value = self.balance
        for symbol, amount in self.positions.items():
            if amount > 0:
                # Aktuellen Preis schätzen (vereinfacht)
                for signal in signals:
                    if signal.symbol == symbol:
                        total_portfolio_value += amount * signal.price
                        break
        
        portfolio_table = Table(title="💼 Portfolio", box=box.SIMPLE)
        portfolio_table.add_column("Metrik", style="cyan")
        portfolio_table.add_column("Wert", style="white")
        
        portfolio_table.add_row("💰 Cash", f"€{self.balance:.2f}")
        portfolio_table.add_row("📊 Portfolio-Wert", f"€{total_portfolio_value:.2f}")
        
        profit_loss = total_portfolio_value - self.initial_balance
        profit_loss_pct = (profit_loss / self.initial_balance) * 100
        
        color = "green" if profit_loss >= 0 else "red"
        icon = "📈" if profit_loss >= 0 else "📉"
        
        portfolio_table.add_row("📈 Gewinn/Verlust", f"[{color}]{icon} €{profit_loss:+.2f} ({profit_loss_pct:+.1f}%)[/{color}]")
        
        self.console.print(portfolio_table)
        
        # Aktive Positionen
        if any(amount > 0 for amount in self.positions.values()):
            positions_table = Table(title="📊 Positionen", box=box.SIMPLE)
            positions_table.add_column("Symbol", style="cyan")
            positions_table.add_column("Menge", style="white")
            positions_table.add_column("Aktueller Wert", style="green")
            
            for symbol, amount in self.positions.items():
                if amount > 0:
                    current_price = 0
                    for signal in signals:
                        if signal.symbol == symbol:
                            current_price = signal.price
                            break
                    
                    value = amount * current_price if current_price > 0 else 0
                    positions_table.add_row(
                        symbol,
                        f"{amount:.4f}",
                        f"€{value:.2f}"
                    )
            
            self.console.print(positions_table)
    
    async def run_simple_bot(self):
        """🔄 Einfacher Bot-Loop"""
        self.console.print(Panel("🤖 Simple Trading Bot gestartet", style="bold green"))
        
        try:
            cycle = 0
            while True:
                cycle += 1
                self.console.clear()
                
                # Header
                self.console.print(Panel(f"🤖 Simple AI Trading Bot - Zyklus {cycle}", style="bold magenta"))
                
                # Signale generieren
                signals = []
                for coin_id in self.watchlist:
                    signal = self.get_simple_signal(coin_id)
                    signals.append(signal)
                    
                    # Trade ausführen wenn Signal stark genug
                    if signal.action in ["BUY", "SELL"]:
                        executed = self.execute_simple_trade(signal)
                        if executed:
                            action_msg = f"✅ {signal.action}: {signal.symbol} @ €{signal.price:.2f}"
                            self.console.print(f"[green]{action_msg}[/green]")
                
                # Status anzeigen
                self.create_status_display(signals)
                
                # Footer
                next_update = datetime.now().strftime("%H:%M:%S")
                footer_text = f"🕐 Letztes Update: {next_update} | 🔄 Nächstes Update in 30s | Drücke CTRL+C zum Beenden"
                self.console.print(Panel(footer_text, style="dim"))
                
                # 30 Sekunden warten (in 1s Schritten für bessere Cancellation)
                for i in range(30):
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            self.console.print("\n🛑 [yellow]Bot wird beendet...[/yellow]")
        except Exception as e:
            self.console.print(f"\n❌ [red]Fehler: {e}[/red]")
        finally:
            # Finale Statistiken
            self.console.print("\n" + "="*50)
            self.console.print("📊 [green]Finale Trading-Statistiken:[/green]")
            self.console.print(f"💰 Endkapital: €{self.balance:.2f}")
            self.console.print(f"📈 Ausgeführte Trades: {len(self.trades)}")
            
            if self.trades:
                profit_loss = self.balance - self.initial_balance
                profit_loss_pct = (profit_loss / self.initial_balance) * 100
                color = "green" if profit_loss >= 0 else "red"
                self.console.print(f"📊 [{color}]Gesamtergebnis: €{profit_loss:+.2f} ({profit_loss_pct:+.1f}%)[/{color}]")

async def main():
    """🚀 Hauptfunktion"""
    bot = SimpleTradingBot(balance=10000)
    
    print("🤖 Simple AI Trading Bot")
    print("💰 Startkapital: €10,000")
    print("📊 Coins: Bitcoin, Ethereum, Binance Coin")
    print("⚠️  SIMULATION - Kein echtes Geld!")
    print("🔄 Updates alle 30 Sekunden")
    print("\nDrücke CTRL+C zum Beenden\n")
    
    input("Enter drücken zum Starten...")
    
    try:
        await bot.run_simple_bot()
    except KeyboardInterrupt:
        print("\n👋 Bot beendet!")

if __name__ == "__main__":
    asyncio.run(main())
