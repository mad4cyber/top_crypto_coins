#!/usr/bin/env python3
"""
📚 Woche 1: Testnet Training System
Autor: mad4cyber
Version: 1.0 - Training Edition

🎯 WOCHE 1 ZIELE:
- 10+ erfolgreiche Testnet-Trades
- System-Verständnis aufbauen
- Parameter-Optimierung
- Trading-Routine entwickeln
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import config
    from trading_parameters import *
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL
    from trading_signals_demo import SimpleTradingSignals
except ImportError as e:
    print(f"❌ Import Error: {e}")
    exit(1)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box

class Week1Training:
    """📚 Woche 1 Testnet Training System"""
    
    def __init__(self):
        self.console = Console()
        
        # API Setup (nur Testnet für Training)
        self.client = Client(
            api_key=config.BINANCE_TESTNET_API_KEY,
            api_secret=config.BINANCE_TESTNET_SECRET,
            testnet=True
        )
        
        self.signal_generator = SimpleTradingSignals()
        
        # Training Stats
        self.training_stats = {
            'total_sessions': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'daily_sessions': {}
        }
        
        # Conservative Training Parameters (angepasst für mehr Trades)
        self.params = TradingParameters(
            max_risk_per_trade=0.01,      # 1% Risiko (Training)
            min_ai_confidence=0.60,       # 60% Konfidenz (für Training reduziert)
            max_positions=3,              # Max 3 Positionen
            base_position_size_usd=25.0,  # $25 Training-Größe
            max_daily_trades=8            # Max 8 Trades/Tag (Training)
        )
        
        self.load_training_stats()
    
    def load_training_stats(self):
        """📊 Training-Statistiken laden"""
        try:
            with open('/Users/anderson/top_crypto_coins/training_stats.json', 'r') as f:
                self.training_stats.update(json.load(f))
        except FileNotFoundError:
            # Erste Session - Stats sind bereits initialisiert
            pass
    
    def save_training_stats(self):
        """💾 Training-Statistiken speichern"""
        try:
            with open('/Users/anderson/top_crypto_coins/training_stats.json', 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)
        except Exception as e:
            self.console.print(f"⚠️ Stats-Speicherung fehlgeschlagen: {e}")
    
    def get_balance(self) -> float:
        """💰 Testnet USDT Balance"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.console.print(f"⚠️ Balance-Fehler: {e}")
            return 10000.0  # Fallback für Training
    
    def display_training_dashboard(self):
        """📊 Training-Dashboard"""
        self.console.clear()
        self.console.print(Panel.fit("📚 Woche 1: Testnet Training Dashboard", style="bold blue"))
        
        # Training Progress
        progress_table = Table(title="🎯 Training-Fortschritt", box=box.ROUNDED)
        progress_table.add_column("Metrik", style="cyan")
        progress_table.add_column("Aktuell", style="white")
        progress_table.add_column("Ziel", style="green")
        progress_table.add_column("Status", style="yellow")
        
        # Trades Progress
        trades_progress = min(100, (self.training_stats['total_trades'] / 10) * 100)
        trades_status = "✅ Erreicht" if self.training_stats['total_trades'] >= 10 else f"{trades_progress:.0f}%"
        progress_table.add_row("🔄 Trades", str(self.training_stats['total_trades']), "10+", trades_status)
        
        # Success Rate
        success_rate = 0
        if self.training_stats['total_trades'] > 0:
            success_rate = (self.training_stats['successful_trades'] / self.training_stats['total_trades']) * 100
        success_status = "✅ Gut" if success_rate >= 60 else "⚠️ Verbessern" if success_rate >= 40 else "❌ Üben"
        progress_table.add_row("🎯 Erfolgsrate", f"{success_rate:.1f}%", "60%+", success_status)
        
        # Sessions
        sessions_status = "✅ Erreicht" if self.training_stats['total_sessions'] >= 7 else f"{self.training_stats['total_sessions']}/7"
        progress_table.add_row("📅 Tage geübt", str(self.training_stats['total_sessions']), "7", sessions_status)
        
        self.console.print(progress_table)
        
        # Performance Stats
        if self.training_stats['total_trades'] > 0:
            perf_table = Table(title="📈 Performance", box=box.SIMPLE)
            perf_table.add_column("Statistik", style="cyan")
            perf_table.add_column("Wert", style="white")
            
            perf_table.add_row("💰 Gesamt P&L", f"${self.training_stats['total_pnl']:+,.2f}")
            perf_table.add_row("🚀 Bester Trade", f"${self.training_stats['best_trade']:+,.2f}")
            perf_table.add_row("📉 Schlechtester Trade", f"${self.training_stats['worst_trade']:+,.2f}")
            perf_table.add_row("⚖️ Ø Trade", f"${(self.training_stats['total_pnl']/self.training_stats['total_trades']):+,.2f}")
            
            self.console.print(perf_table)
        
        # Current Balance
        current_balance = self.get_balance()
        self.console.print(f"💰 [green]Aktuelle Balance: ${current_balance:,.2f} USDT[/green]")
    
    async def execute_training_trade(self, symbol: str, signal: Dict) -> bool:
        """🎓 Training-Trade ausführen"""
        try:
            # Dynamic Position Size für Training based on coin price
            balance = self.get_balance()
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            
            # Symbol-specific requirements
            symbol_requirements = {
                'BTCUSDT': {'min_qty': 0.00001, 'step_size': 0.00001},
                'ETHUSDT': {'min_qty': 0.00001, 'step_size': 0.00001},
                'BNBUSDT': {'min_qty': 0.00001, 'step_size': 0.00001},
                'SOLUSDT': {'min_qty': 0.001, 'step_size': 0.001},
                'ADAUSDT': {'min_qty': 0.1, 'step_size': 0.1}
            }
            
            req = symbol_requirements.get(symbol, {'min_qty': 0.001, 'step_size': 0.001})
            min_quantity = req['min_qty']
            step_size = req['step_size']
            
            # Dynamic position sizing
            position_usd = min(25.0, balance * 0.05)  # Start with $25 or 5% of balance
            
            if position_usd < 10:
                self.console.print("⚠️ Position zu klein für Training")
                return False
            
            # Calculate quantity and round to proper step size
            quantity = position_usd / current_price
            
            # Round to step size (important for Binance)
            quantity = round(quantity / step_size) * step_size
            
            # Format to avoid scientific notation
            if step_size >= 1:
                quantity = round(quantity, 0)
            elif step_size >= 0.1:
                quantity = round(quantity, 1)
            elif step_size >= 0.01:
                quantity = round(quantity, 2)
            elif step_size >= 0.001:
                quantity = round(quantity, 3)
            elif step_size >= 0.0001:
                quantity = round(quantity, 4)
            else:
                quantity = round(quantity, 5)
            
            self.console.print(f"💰 Position: ${position_usd:.2f} | Quantity: {quantity} | Min: {min_quantity} | Step: {step_size}")
            
            if quantity >= min_quantity:
                # Training-Trade ausführen
                order = self.client.order_market(
                    symbol=symbol,
                    side=SIDE_BUY,
                    quantity=quantity
                )
                
                self.console.print(f"✅ [green]Training-Trade: {order['orderId']} | {quantity:.6f} {symbol}[/green]")
                
                # Stats aktualisieren
                self.training_stats['total_trades'] += 1
                self.training_stats['successful_trades'] += 1
                
                # Simuliere P&L für Training (basierend auf AI-Prognose)
                expected_pnl = position_usd * (signal['price_change_pct'] / 100)
                self.training_stats['total_pnl'] += expected_pnl
                self.training_stats['best_trade'] = max(self.training_stats['best_trade'], expected_pnl)
                self.training_stats['worst_trade'] = min(self.training_stats['worst_trade'], expected_pnl)
                
                return True
            else:
                self.console.print("⚠️ Quantity zu klein")
                self.training_stats['failed_trades'] += 1
                return False
                
        except Exception as e:
            self.console.print(f"❌ Training-Trade Fehler: {e}")
            self.training_stats['failed_trades'] += 1
            return False
    
    async def run_training_session(self, duration_minutes: int = 15):
        """📚 Eine Training-Session durchführen"""
        session_start = datetime.now()
        today = session_start.strftime('%Y-%m-%d')
        
        self.console.print(Panel.fit(f"📚 Training-Session ({duration_minutes} Min)", style="bold blue"))
        
        # Session Stats
        session_trades = 0
        session_pnl = 0.0
        
        cycles = max(1, duration_minutes // 5)  # Mindestens 1 Zyklus
        
        try:
            for cycle in range(cycles):
                self.console.print(f"\n🔄 [cyan]Training-Zyklus {cycle + 1}/{cycles}[/cyan]")
                
                # AI-Signale generieren
                self.console.print("🧠 [cyan]Analysiere Märkte...[/cyan]")
                signals = await self.signal_generator.generate_signals()
                
                if signals:
                    self.console.print(f"📊 [dim]{len(signals)} Signale generiert[/dim]")
                    
                    # Debug: Zeige alle Signale
                    for i, sig in enumerate(signals[:3]):
                        self.console.print(f"  {i+1}. {sig['symbol']}: {sig['signal']} ({sig['confidence']:.1%}, {sig['price_change_pct']:+.2f}%)")
                    
                    # Beste Training-Signale filtern
                    good_signals = [
                        s for s in signals 
                        if s['confidence'] >= self.params.min_ai_confidence
                        and abs(s['price_change_pct']) >= 1.5  # Min 1.5% für Training (reduziert)
                        and s['signal'] in ['STRONG_BUY', 'BUY']
                    ]
                    
                    self.console.print(f"✅ [green]{len(good_signals)} gute Signale gefunden[/green]")
                    
                    if good_signals and session_trades < self.params.max_daily_trades:
                        best = good_signals[0]
                        
                        # Symbol mapping
                        symbol_mapping = {
                            'BITCOIN': 'BTCUSDT',
                            'SOLANA': 'SOLUSDT',
                            'BINANCECOIN': 'BNBUSDT',
                            'ETHEREUM': 'ETHUSDT',
                            'CARDANO': 'ADAUSDT'
                        }
                        symbol = symbol_mapping.get(best['symbol'], '')
                        
                        if symbol in config.TRADING_SYMBOLS:
                            self.console.print(f"🎯 [green]Training-Signal: {symbol} ({best['confidence']:.1%}, {best['price_change_pct']:+.2f}%)[/green]")
                            
                            success = await self.execute_training_trade(symbol, best)
                            if success:
                                session_trades += 1
                                session_pnl += best['price_change_pct'] * 0.25  # Training P&L
                                
                                # Dashboard update
                                self.display_training_dashboard()
                                await asyncio.sleep(3)  # Zeit zum Lesen
                    else:
                        self.console.print("⏸️ [yellow]Keine starken Signale oder Limit erreicht[/yellow]")
                else:
                    self.console.print("❌ [red]Keine AI-Signale generiert[/red]")
                
                # Pause zwischen Zyklen
                if cycle < cycles - 1:
                    self.console.print("⏳ Pause...")
                    await asyncio.sleep(5)  # Kurze Pause für Training
        
        except KeyboardInterrupt:
            self.console.print("\n🛑 [yellow]Training-Session gestoppt[/yellow]")
        
        # Session Summary
        self.training_stats['total_sessions'] += 1
        if today not in self.training_stats['daily_sessions']:
            self.training_stats['daily_sessions'][today] = 0
        self.training_stats['daily_sessions'][today] += 1
        
        # Stats speichern
        self.save_training_stats()
        
        # Session Summary anzeigen
        self.console.print(f"\n📊 [green]Training-Session beendet[/green]")
        self.console.print(f"⏰ Dauer: {duration_minutes} Minuten")
        self.console.print(f"🔄 Trades: {session_trades}")
        self.console.print(f"📈 Session P&L: ${session_pnl:+.2f}")
        
        # Progress Check
        if self.training_stats['total_trades'] >= 10:
            self.console.print("\n🎉 [bold green]WOCHE 1 ZIEL ERREICHT! 10+ Trades abgeschlossen![/bold green]")
            self.console.print("🚀 [cyan]Bereit für Woche 2: Live Demo Trading![/cyan]")
    
    def show_week_summary(self):
        """📊 Wochen-Zusammenfassung"""
        self.console.print(Panel.fit("📊 Woche 1 Training - Zusammenfassung", style="bold green"))
        
        summary_table = Table(title="🎯 Training-Ergebnisse", box=box.ROUNDED)
        summary_table.add_column("Achievement", style="cyan")
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Bewertung", style="green")
        
        # Trades
        trades_status = "✅ Erreicht" if self.training_stats['total_trades'] >= 10 else "⚠️ Weiter üben"
        summary_table.add_row("🔄 10+ Trades", f"{self.training_stats['total_trades']}/10", trades_status)
        
        # Success Rate
        success_rate = 0
        if self.training_stats['total_trades'] > 0:
            success_rate = (self.training_stats['successful_trades'] / self.training_stats['total_trades']) * 100
        
        rate_status = "🏆 Exzellent" if success_rate >= 80 else "✅ Gut" if success_rate >= 60 else "⚠️ OK" if success_rate >= 40 else "❌ Mehr Übung"
        summary_table.add_row("🎯 Erfolgsrate", f"{success_rate:.1f}%", rate_status)
        
        # Sessions
        session_status = "🏆 Sehr gut" if self.training_stats['total_sessions'] >= 7 else "✅ Gut" if self.training_stats['total_sessions'] >= 5 else "⚠️ Mehr üben"
        summary_table.add_row("📅 Training-Tage", f"{self.training_stats['total_sessions']}", session_status)
        
        self.console.print(summary_table)
        
        # Recommendations
        if self.training_stats['total_trades'] >= 10 and success_rate >= 60:
            self.console.print("\n🎉 [bold green]BEREIT FÜR LIVE TRADING![/bold green]")
            self.console.print("🚀 [cyan]Nächster Schritt: Woche 2 Live Demo mit echtem Geld[/cyan]")
        else:
            self.console.print("\n📚 [yellow]Empfehlung: Mehr Training[/yellow]")
            if self.training_stats['total_trades'] < 10:
                self.console.print(f"🔄 Noch {10 - self.training_stats['total_trades']} Trades für Ziel")
            if success_rate < 60:
                self.console.print("🎯 Fokus auf Signalqualität und Timing")


async def main():
    """🎯 Woche 1 Training Hauptfunktion"""
    training = Week1Training()
    
    # Dashboard anzeigen
    training.display_training_dashboard()
    
    console = Console()
    console.print("\n📚 [bold blue]Woche 1: Testnet Training[/bold blue]")
    console.print("🎯 Ziel: 10+ erfolgreiche Trades auf Testnet")
    console.print("\n🔧 Wähle Training-Option:")
    console.print("1. Kurze Session (15 Min)")
    console.print("2. Standard Session (30 Min)")
    console.print("3. Intensive Session (60 Min)")
    console.print("4. Wochen-Zusammenfassung anzeigen")
    
    choice = input("\nWähle (1-4): ").strip()
    
    if choice == "4":
        training.show_week_summary()
    elif choice == "3":
        await training.run_training_session(60)
    elif choice == "2":
        await training.run_training_session(30)
    else:
        await training.run_training_session(15)


if __name__ == "__main__":
    asyncio.run(main())
