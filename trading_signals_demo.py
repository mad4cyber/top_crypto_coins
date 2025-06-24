#!/usr/bin/env python3
"""
🎯 Trading Signal Generator Demo
Autor: mad4cyber
Version: 1.0 - Demo Edition

🚀 FEATURES:
- AI-Powered Trading Signale
- Technische Analyse
- Risk Management
- Portfolio-Simulation
- Kein Live Display (Demo-Version)
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

# Importiere unsere Module
from ai_predictor import CryptoAIPredictor

class SimpleTradingSignals:
    """🎯 Einfache Trading Signal Generierung"""
    
    def __init__(self):
        self.console = Console()
        self.ai_predictor = CryptoAIPredictor()
        
        # Trading Parameter
        self.min_confidence = 0.75
        self.symbols = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']
        
    async def generate_signals(self) -> List[Dict]:
        """🧠 Trading-Signale generieren"""
        signals = []
        
        self.console.print("🔍 [cyan]Analysiere Kryptowährungen...[/cyan]")
        
        for i, symbol in enumerate(self.symbols, 1):
            try:
                self.console.print(f"[cyan]Analysiere {symbol}... ({i}/{len(self.symbols)})[/cyan]")
                # AI-Prognose abrufen
                result = self.ai_predictor.predict_future_prices(symbol)
                
                if 'error' not in result:
                    current_price = result['current_price']
                    predicted_price = result['predicted_price']
                    price_change_pct = result['price_change_pct']
                    confidence = result['confidence']
                    
                    # Trading-Signal bestimmen
                    if confidence >= self.min_confidence:
                        if price_change_pct > 3.0:  # > 3% erwarteter Anstieg
                            signal_type = "STRONG_BUY"
                            signal_strength = "🔥🔥🔥"
                        elif price_change_pct > 1.5:  # > 1.5% erwarteter Anstieg
                            signal_type = "BUY"
                            signal_strength = "🔥🔥"
                        elif price_change_pct < -3.0:  # > 3% erwarteter Rückgang
                            signal_type = "STRONG_SELL"
                            signal_strength = "💥💥💥"
                        elif price_change_pct < -1.5:  # > 1.5% erwarteter Rückgang
                            signal_type = "SELL"
                            signal_strength = "💥💥"
                        else:
                            signal_type = "HOLD"
                            signal_strength = "⏸️"
                    else:
                        signal_type = "HOLD"
                        signal_strength = "💧"
                    
                    # Risk Management
                    entry_price = current_price
                    stop_loss = entry_price * 0.95 if signal_type in ["BUY", "STRONG_BUY"] else entry_price * 1.05
                    take_profit = entry_price * 1.10 if signal_type in ["BUY", "STRONG_BUY"] else entry_price * 0.90
                    risk_reward = 2.0  # 1:2 Risk/Reward
                    
                    signal = {
                        'symbol': symbol.upper(),
                        'signal': signal_type,
                        'strength': signal_strength,
                        'confidence': confidence,
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'price_change_pct': price_change_pct,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward': risk_reward,
                        'timestamp': datetime.now()
                    }
                    
                    signals.append(signal)
                    
                # Kurze Pause zwischen Anfragen
                await asyncio.sleep(0.5)
                    
            except Exception as e:
                self.console.print(f"❌ [red]Fehler bei {symbol}: {e}[/red]")
        
        # Nach Konfidenz sortieren
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def display_signals(self, signals: List[Dict]):
        """📊 Trading-Signale anzeigen"""
        self.console.print(Panel.fit("🎯 AI Trading Signals Dashboard", style="bold magenta"))
        
        # Haupt-Signale Tabelle
        table = Table(title="🎯 Trading Signals (Sortiert nach Konfidenz)", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Signal", style="white", width=15)
        table.add_column("Stärke", style="yellow", width=8)
        table.add_column("Konfidenz", style="green", width=10)
        table.add_column("Aktuell", style="white", width=12)
        table.add_column("Prognose", style="white", width=12)
        table.add_column("Änderung", style="white", width=10)
        table.add_column("Stop Loss", style="red", width=12)
        table.add_column("Take Profit", style="green", width=12)
        
        for signal in signals:
            # Signal-Styling
            if signal['signal'] in ["BUY", "STRONG_BUY"]:
                signal_color = "green"
                signal_icon = "🚀" if signal['signal'] == "STRONG_BUY" else "📈"
            elif signal['signal'] in ["SELL", "STRONG_SELL"]:
                signal_color = "red"
                signal_icon = "💥" if signal['signal'] == "STRONG_SELL" else "📉"
            else:
                signal_color = "yellow"
                signal_icon = "⏸️"
            
            # Änderung-Styling
            change_color = "green" if signal['price_change_pct'] > 0 else "red"
            
            table.add_row(
                signal['symbol'],
                f"[{signal_color}]{signal_icon} {signal['signal']}[/{signal_color}]",
                signal['strength'],
                f"{signal['confidence']:.1%}",
                f"€{signal['current_price']:,.2f}",
                f"€{signal['predicted_price']:,.2f}",
                f"[{change_color}]{signal['price_change_pct']:+.2f}%[/{change_color}]",
                f"€{signal['stop_loss']:,.2f}",
                f"€{signal['take_profit']:,.2f}"
            )
        
        self.console.print(table)
        
        # Top Signal Highlight
        if signals:
            top_signal = signals[0]
            self.console.print(f"\n🏆 [bold]TOP SIGNAL: {top_signal['symbol']}[/bold]")
            self.console.print(f"📊 Signal: {top_signal['signal']}")
            self.console.print(f"🎯 Konfidenz: {top_signal['confidence']:.1%}")
            self.console.print(f"💰 Entry: €{top_signal['entry_price']:,.2f}")
            self.console.print(f"🛑 Stop Loss: €{top_signal['stop_loss']:,.2f}")
            self.console.print(f"🎯 Take Profit: €{top_signal['take_profit']:,.2f}")
            self.console.print(f"⚖️ Risk/Reward: 1:{top_signal['risk_reward']:.1f}")
    
    def simulate_portfolio(self, signals: List[Dict], initial_balance: float = 10000) -> Dict:
        """💰 Portfolio-Simulation"""
        balance = initial_balance
        positions = []
        
        # Nur starke Signale handeln
        strong_signals = [s for s in signals if s['confidence'] >= self.min_confidence and s['signal'] in ["STRONG_BUY", "BUY"]]
        
        for signal in strong_signals[:3]:  # Max 3 Positionen
            # 20% des Portfolios pro Position
            position_value = balance * 0.20
            position_size = position_value / signal['entry_price']
            
            positions.append({
                'symbol': signal['symbol'],
                'size': position_size,
                'entry_price': signal['entry_price'],
                'current_price': signal['predicted_price'],  # Simuliere Prognose als Realität
                'value': position_size * signal['predicted_price'],
                'pnl': (signal['predicted_price'] - signal['entry_price']) * position_size
            })
            
            balance -= position_value
        
        total_value = balance + sum(pos['value'] for pos in positions)
        total_pnl = sum(pos['pnl'] for pos in positions)
        total_return = ((total_value - initial_balance) / initial_balance) * 100
        
        return {
            'initial_balance': initial_balance,
            'cash_balance': balance,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'positions': positions
        }
    
    def display_portfolio_simulation(self, portfolio: Dict):
        """📊 Portfolio-Simulation anzeigen"""
        self.console.print(Panel.fit("💼 Portfolio Simulation", style="bold blue"))
        
        # Portfolio Statistiken
        stats_table = Table(title="📊 Portfolio Statistiken", box=box.SIMPLE)
        stats_table.add_column("Metrik", style="cyan")
        stats_table.add_column("Wert", style="white")
        
        stats_table.add_row("💰 Startkapital", f"€{portfolio['initial_balance']:,.2f}")
        stats_table.add_row("💵 Cash", f"€{portfolio['cash_balance']:,.2f}")
        stats_table.add_row("📊 Gesamtwert", f"€{portfolio['total_value']:,.2f}")
        
        pnl_color = "green" if portfolio['total_pnl'] >= 0 else "red"
        pnl_icon = "📈" if portfolio['total_pnl'] >= 0 else "📉"
        stats_table.add_row(
            "💹 P&L", 
            f"[{pnl_color}]{pnl_icon} €{portfolio['total_pnl']:+,.2f}[/{pnl_color}]"
        )
        
        return_color = "green" if portfolio['total_return'] >= 0 else "red"
        stats_table.add_row(
            "📈 Rendite", 
            f"[{return_color}]{portfolio['total_return']:+.2f}%[/{return_color}]"
        )
        
        self.console.print(stats_table)
        
        # Positionen
        if portfolio['positions']:
            positions_table = Table(title="📋 Positionen", box=box.SIMPLE)
            positions_table.add_column("Symbol", style="cyan")
            positions_table.add_column("Größe", style="yellow")
            positions_table.add_column("Entry", style="white")
            positions_table.add_column("Aktuell", style="white")
            positions_table.add_column("Wert", style="green")
            positions_table.add_column("P&L", style="white")
            
            for pos in portfolio['positions']:
                pnl_color = "green" if pos['pnl'] >= 0 else "red"
                positions_table.add_row(
                    pos['symbol'],
                    f"{pos['size']:.4f}",
                    f"€{pos['entry_price']:,.2f}",
                    f"€{pos['current_price']:,.2f}",
                    f"€{pos['value']:,.2f}",
                    f"[{pnl_color}]€{pos['pnl']:+,.2f}[/{pnl_color}]"
                )
            
            self.console.print(positions_table)


async def main():
    """🎯 Hauptfunktion"""
    console = Console()
    
    console.print(Panel.fit("🎯 AI Trading Signal Generator Demo", style="bold magenta"))
    console.print("⚠️  [yellow]Dies ist eine Demo-Version für Lernzwecke![/yellow]")
    console.print("💡 [dim]Echtes Trading nur nach gründlichen Tests![/dim]\n")
    
    # Signal Generator initialisieren
    signal_generator = SimpleTradingSignals()
    
    console.print("🚀 [green]Starte Trading Signal Analyse...[/green]")
    console.print("⏰ Das kann einige Minuten dauern...\n")
    
    try:
        # Signale generieren
        signals = await signal_generator.generate_signals()
        
        if signals:
            # Signale anzeigen
            signal_generator.display_signals(signals)
            
            # Portfolio-Simulation
            console.print("\n🔮 [cyan]Führe Portfolio-Simulation durch...[/cyan]")
            portfolio = signal_generator.simulate_portfolio(signals)
            signal_generator.display_portfolio_simulation(portfolio)
            
            console.print(f"\n✅ [green]{len(signals)} Trading-Signale generiert![/green]")
            console.print("🎯 [cyan]Signale sind nach AI-Konfidenz sortiert[/cyan]")
            
            # Zusammenfassung
            strong_signals = [s for s in signals if s['signal'] in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']]
            console.print(f"💪 [bold]{len(strong_signals)} starke Handelssignale gefunden![/bold]")
            
            # Trading-Empfehlungen
            console.print("\n📋 [bold]Trading-Empfehlungen:[/bold]")
            for i, signal in enumerate(strong_signals[:3], 1):
                console.print(f"{i}. {signal['symbol']}: {signal['signal']} (Konfidenz: {signal['confidence']:.1%})")
        
        else:
            console.print("❌ [red]Keine Signale generiert. Prüfe Internetverbindung.[/red]")
            
    except Exception as e:
        console.print(f"❌ [red]Fehler: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())
