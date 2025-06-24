#!/usr/bin/env python3
"""
🤖 Binance Trading Bot Integration mit Advanced Signals
Autor: mad4cyber
Version: 3.0 - Production Ready Edition

🚀 FEATURES:
- Vollständige Binance API Integration
- AI-Powered Signal Generation
- Automatische Order-Platzierung
- Risk Management
- Portfolio-Tracking
- Real-Time Monitoring
- Paper Trading Mode
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

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
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress
from rich import box

from advanced_trading_signals import AdvancedTradingSignals, TradingSignal, SignalType, SignalStrength, TimeFrame

class TradingMode(Enum):
    """Trading Modi"""
    PAPER = "paper"  # Simulation
    LIVE = "live"    # Echtes Trading
    TESTNET = "testnet"  # Binance Testnet

@dataclass
class Trade:
    """Trade-Objekt"""
    id: str
    symbol: str
    side: str  # BUY/SELL
    amount: float
    price: float
    order_id: Optional[str] = None
    status: str = "PENDING"
    timestamp: datetime = None
    profit_loss: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Position-Objekt"""
    symbol: str
    amount: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()

class BinanceTradingBot:
    """🤖 Erweiterte Binance Trading Bot Integration"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 mode: TradingMode = TradingMode.PAPER, initial_balance: float = 10000):
        self.console = Console()
        self.mode = mode
        
        # Binance Client Setup
        self.client = None
        if BINANCE_AVAILABLE and api_key and api_secret:
            try:
                if mode == TradingMode.TESTNET:
                    self.client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
                elif mode == TradingMode.LIVE:
                    self.client = Client(api_key=api_key, api_secret=api_secret)
                    self.console.print("⚠️ [red]LIVE TRADING MODUS AKTIVIERT![/red]")
                
                if self.client:
                    # API-Verbindung testen
                    account_info = self.client.get_account()
                    self.console.print(f"✅ [green]Binance API verbunden ({mode.value.upper()})[/green]")
            except Exception as e:
                self.console.print(f"❌ [red]Binance API Fehler: {e}[/red]")
                self.console.print("🔄 [yellow]Wechsle zu Paper Trading Modus...[/yellow]")
                self.mode = TradingMode.PAPER
        else:
            self.mode = TradingMode.PAPER
        
        # Signal Generator
        self.signal_generator = AdvancedTradingSignals(
            api_key=api_key, 
            api_secret=api_secret,
            testnet=(mode == TradingMode.TESTNET)
        )
        
        # Portfolio Management
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        
        # Trading Settings
        self.max_positions = 5
        self.max_risk_per_trade = 0.02  # 2%
        self.min_signal_confidence = 0.10
        self.trading_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'
        ]
        
        # Performance Tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('BinanceTradingBot')
    
    def get_account_balance(self) -> float:
        """💰 Aktuelle Kontoguthaben abrufen"""
        if self.mode == TradingMode.PAPER:
            return self.balance
        
        try:
            account = self.client.get_account()
            usdt_balance = 0.0
            
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            return usdt_balance
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {e}")
            return 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """💲 Aktueller Preis eines Symbols"""
        try:
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            else:
                # Fallback ohne API
                return 50000.0  # Mock-Preis
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Preises für {symbol}: {e}")
            return 0.0
    
    def get_symbol_filters(self, symbol: str) -> dict:
        """📏 Binance Symbol-Filter abrufen"""
        try:
            if self.client:
                exchange_info = self.client.get_exchange_info()
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        filters = {}
                        for filter_info in symbol_info['filters']:
                            if filter_info['filterType'] == 'LOT_SIZE':
                                filters['min_qty'] = float(filter_info['minQty'])
                                filters['max_qty'] = float(filter_info['maxQty'])
                                filters['step_size'] = float(filter_info['stepSize'])
                            elif filter_info['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']:
                                filters['min_notional'] = float(filter_info['minNotional'])
                        return filters
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Symbol-Filter: {e}")
        
        # Fallback-Werte
        return {
            'min_qty': 0.001,
            'max_qty': 1000000,
            'step_size': 0.001,
            'min_notional': 5.0
        }
    
    def round_to_step_size(self, quantity: float, step_size: float) -> float:
        """🔧 Quantity auf Step Size runden (mit Floating-Point-Korrektur)"""
        if step_size == 0:
            return quantity
        
        # Anzahl der Steps berechnen
        steps = quantity / step_size
        # Auf ganze Steps runden (nicht nur abrunden)
        rounded_steps = round(steps)
        # Zurück zur Quantity
        result = rounded_steps * step_size
        
        # Präzision basierend auf Step Size bestimmen
        step_str = f'{step_size:.10f}'.rstrip('0').rstrip('.')
        decimal_places = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        # Auf entsprechende Dezimalstellen runden
        return round(result, decimal_places)
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """📊 Optimale Positionsgröße berechnen (Binance-konform)"""
        available_balance = self.get_account_balance()
        
        # Symbol-Filter abrufen
        filters = self.get_symbol_filters(signal.symbol)
        
        # Risk-basierte Positionsgröße
        risk_amount = available_balance * self.max_risk_per_trade
        
        # Stop-Loss Distanz
        if signal.stop_loss:
            stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            position_value = risk_amount / stop_distance
        else:
            position_value = risk_amount / 0.05  # Default 5% Stop-Loss
        
        # Konfidenz-Anpassung
        confidence_multiplier = signal.confidence / self.min_signal_confidence
        position_value *= confidence_multiplier
        
        # Maximum 20% des Portfolios pro Position
        max_position_value = available_balance * 0.20
        position_value = min(position_value, max_position_value)
        
        # Position in Coin-Anzahl umrechnen
        raw_position_size = position_value / signal.entry_price
        
        # Auf Binance Step Size runden
        position_size = self.round_to_step_size(raw_position_size, filters['step_size'])
        
        # Minimum Quantity prüfen
        if position_size < filters['min_qty']:
            position_size = filters['min_qty']
        
        # Minimum Notional prüfen
        notional_value = position_size * signal.entry_price
        if notional_value < filters['min_notional']:
            # Mindest-Positionsgröße für Notional berechnen
            min_position_for_notional = filters['min_notional'] / signal.entry_price
            position_size = self.round_to_step_size(min_position_for_notional, filters['step_size'])
            # Sicherstellen, dass es nach dem Runden noch ausreicht
            if position_size * signal.entry_price < filters['min_notional']:
                position_size += filters['step_size']
        
        # Maximum Quantity prüfen
        if position_size > filters['max_qty']:
            position_size = self.round_to_step_size(filters['max_qty'], filters['step_size'])
        
        self.logger.info(f"📏 Position Size für {signal.symbol}: {position_size:.8f} (Notional: ${position_size * signal.entry_price:.2f})")
        
        return position_size
    
    def format_quantity_for_binance(self, quantity: float, step_size: float) -> str:
        """🔧 Quantity für Binance API formatieren"""
        # Bestimme Dezimalstellen basierend auf Step Size
        step_str = f'{step_size:.10f}'.rstrip('0').rstrip('.')
        decimal_places = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        # Formatiere mit exakter Präzision
        formatted = f'{quantity:.{decimal_places}f}'
        return formatted
    
    async def place_order(self, signal: TradingSignal) -> Optional[Trade]:
        """📈 Order platzieren"""
        try:
            position_size = self.calculate_position_size(signal)
            
            if position_size <= 0:
                self.logger.warning(f"Positionsgröße zu klein für {signal.symbol}")
                return None
            
            # Symbol-Filter für Formatierung
            filters = self.get_symbol_filters(signal.symbol)
            formatted_quantity = self.format_quantity_for_binance(position_size, filters['step_size'])
            
            # Order-Details
            side = SIDE_BUY if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else SIDE_SELL
            order_type = ORDER_TYPE_MARKET
            
            trade = Trade(
                id=f"{signal.symbol}_{int(time.time())}",
                symbol=signal.symbol,
                side=side,
                amount=position_size,
                price=signal.entry_price
            )
            if self.mode == TradingMode.PAPER:
                # Paper Trading - Simulation
                trade.status = "FILLED"
                trade.order_id = f"PAPER_{trade.id}"
                
                # Portfolio aktualisieren
                if side == SIDE_BUY:
                    self.balance -= position_size * signal.entry_price
                    
                    if signal.symbol in self.positions:
                        # Position erweitern
                        existing = self.positions[signal.symbol]
                        total_amount = existing.amount + position_size
                        total_cost = (existing.amount * existing.avg_price) + (position_size * signal.entry_price)
                        new_avg_price = total_cost / total_amount
                        
                        self.positions[signal.symbol] = Position(
                            symbol=signal.symbol,
                            amount=total_amount,
                            avg_price=new_avg_price,
                            current_price=signal.entry_price
                        )
                    else:
                        # Neue Position
                        self.positions[signal.symbol] = Position(
                            symbol=signal.symbol,
                            amount=position_size,
                            avg_price=signal.entry_price,
                            current_price=signal.entry_price
                        )
                
                elif side == SIDE_SELL and signal.symbol in self.positions:
                    # Position verkaufen
                    position = self.positions[signal.symbol]
                    sell_amount = min(position_size, position.amount)
                    revenue = sell_amount * signal.entry_price
                    
                    self.balance += revenue
                    
                    # Profit/Loss berechnen
                    cost_basis = sell_amount * position.avg_price
                    profit_loss = revenue - cost_basis
                    trade.profit_loss = profit_loss
                    
                    # Position aktualisieren
                    position.amount -= sell_amount
                    if position.amount <= 0:
                        del self.positions[signal.symbol]
                
            else:
                # Live/Testnet Trading
                try:
                    # Market Order platzieren mit formatierter Quantity
                    order = self.client.order_market(
                        symbol=signal.symbol,
                        side=side,
                        quantity=formatted_quantity
                    )
                    
                    trade.order_id = order['orderId']
                    trade.status = order['status']
                    
                    self.logger.info(f"✅ Order platziert: {order['orderId']}")
                    
                except BinanceAPIException as e:
                    self.logger.error(f"Binance API Fehler: {e}")
                    return None
            
            # Trade zur Historie hinzufügen
            self.trades.append(trade)
            self.total_trades += 1
            
            # Performance Tracking
            if trade.profit_loss:
                if trade.profit_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            self.logger.info(f"🎯 {side} {position_size:.6f} {signal.symbol} @ ${signal.entry_price:.4f}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Fehler beim Platzieren der Order: {e}")
            return None
    
    def update_positions(self):
        """🔄 Positionen aktualisieren"""
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price > 0:
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.amount
    
    def check_stop_loss_take_profit(self):
        """🛑 Stop-Loss und Take-Profit prüfen"""
        # Vereinfachte Implementierung - kann erweitert werden
        pass
    
    def calculate_portfolio_stats(self) -> Dict[str, float]:
        """📊 Portfolio-Statistiken berechnen"""
        self.update_positions()
        
        total_value = self.balance
        total_unrealized_pnl = 0.0
        
        for position in self.positions.values():
            total_value += position.amount * position.current_price
            total_unrealized_pnl += position.unrealized_pnl
        
        total_realized_pnl = sum(trade.profit_loss for trade in self.trades if trade.profit_loss)
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        return {
            'total_value': total_value,
            'cash_balance': self.balance,
            'positions_value': total_value - self.balance,
            'total_pnl': total_pnl,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_return_pct': ((total_value - self.initial_balance) / self.initial_balance) * 100,
            'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100
        }
    
    def create_dashboard(self, recent_signals: List[TradingSignal]) -> Layout:
        """📊 Live-Dashboard erstellen"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )
        
        layout["body"].split_row(
            Layout(name="signals"),
            Layout(name="portfolio"),
            Layout(name="trades")
        )
        
        # Header
        header = Panel(
            f"🤖 BINANCE TRADING BOT v3.0 - {self.mode.value.upper()} MODE",
            style="bold magenta"
        )
        layout["header"].update(header)
        
        # Signals
        signals_table = Table(title="🎯 Recent Signals", box=box.SIMPLE)
        signals_table.add_column("Symbol", style="cyan")
        signals_table.add_column("Signal", style="white")
        signals_table.add_column("Confidence", style="green")
        signals_table.add_column("Price", style="yellow")
        
        for signal in recent_signals[-5:]:
            color = "green" if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else "red"
            icon = "🚀" if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else "📉"
            
            signals_table.add_row(
                signal.symbol,
                f"[{color}]{icon} {signal.signal.value}[/{color}]",
                f"{signal.confidence:.1%}",
                f"${signal.entry_price:.4f}"
            )
        
        layout["signals"].update(Panel(signals_table, title="Trading Signals"))
        
        # Portfolio
        stats = self.calculate_portfolio_stats()
        portfolio_table = Table(title="💼 Portfolio", box=box.SIMPLE)
        portfolio_table.add_column("Metric", style="cyan")
        portfolio_table.add_column("Value", style="white")
        
        portfolio_table.add_row("💰 Total Value", f"${stats['total_value']:,.2f}")
        portfolio_table.add_row("💵 Cash", f"${stats['cash_balance']:,.2f}")
        portfolio_table.add_row("📊 Positions", f"${stats['positions_value']:,.2f}")
        
        pnl_color = "green" if stats['total_pnl'] >= 0 else "red"
        pnl_icon = "📈" if stats['total_pnl'] >= 0 else "📉"
        portfolio_table.add_row(
            "💹 P&L", 
            f"[{pnl_color}]{pnl_icon} ${stats['total_pnl']:+,.2f}[/{pnl_color}]"
        )
        
        portfolio_table.add_row("📈 Return", f"{stats['total_return_pct']:+.2f}%")
        portfolio_table.add_row("🎯 Win Rate", f"{stats['win_rate']:.1f}%")
        
        layout["portfolio"].update(Panel(portfolio_table, title="Portfolio Stats"))
        
        # Recent Trades
        trades_table = Table(title="📋 Recent Trades", box=box.SIMPLE)
        trades_table.add_column("Symbol", style="cyan")
        trades_table.add_column("Side", style="white")
        trades_table.add_column("Amount", style="yellow")
        trades_table.add_column("Price", style="green")
        trades_table.add_column("P&L", style="white")
        
        for trade in self.trades[-5:]:
            side_color = "green" if trade.side == "BUY" else "red"
            pnl_text = f"${trade.profit_loss:+.2f}" if trade.profit_loss else "-"
            pnl_color = "green" if (trade.profit_loss or 0) >= 0 else "red"
            
            trades_table.add_row(
                trade.symbol,
                f"[{side_color}]{trade.side}[/{side_color}]",
                f"{trade.amount:.6f}",
                f"${trade.price:.4f}",
                f"[{pnl_color}]{pnl_text}[/{pnl_color}]"
            )
        
        layout["trades"].update(Panel(trades_table, title="Trade History"))
        
        # Footer
        runtime = datetime.now() - self.start_time
        footer = Panel(
            f"🕐 Runtime: {runtime} | 📊 Trades: {self.total_trades} | 🔄 Next scan in 60s | Mode: {self.mode.value.upper()}",
            style="dim"
        )
        layout["footer"].update(footer)
        
        return layout
    
    async def trading_loop(self):
        """🔄 Haupt-Trading-Loop"""
        signals_history = []
        
        self.console.print(f"🚀 [green]Trading Bot gestartet im {self.mode.value.upper()} Modus![/green]")
        self.console.print(f"💰 Startkapital: ${self.initial_balance:,.2f}")
        self.console.print(f"🎯 Min. Konfidenz: {self.min_signal_confidence:.0%}")
        self.console.print(f"📊 Überwachte Symbole: {', '.join(self.trading_symbols)}")
        
        try:
            with Live(auto_refresh=False) as live:
                while True:
                    try:
                        # Signale für alle Symbole generieren
                        current_signals = await self.signal_generator.scan_multiple_symbols(
                            self.trading_symbols, 
                            TimeFrame.H1
                        )
                        
                        # Hochwertige Signale filtern
                        high_confidence_signals = [
                            signal for signal in current_signals 
                            if signal.confidence >= self.min_signal_confidence
                            and signal.signal != SignalType.HOLD
                        ]
                        
                        # Trading-Entscheidungen treffen
                        for signal in high_confidence_signals:
                            # Prüfen ob bereits Position vorhanden
                            if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                                if len(self.positions) < self.max_positions:
                                    trade = await self.place_order(signal)
                                    if trade:
                                        self.logger.info(f"✅ Neue Position eröffnet: {signal.symbol}")
                            
                            elif signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                                if signal.symbol in self.positions:
                                    trade = await self.place_order(signal)
                                    if trade:
                                        self.logger.info(f"✅ Position geschlossen: {signal.symbol}")
                        
                        # Stop-Loss/Take-Profit prüfen
                        self.check_stop_loss_take_profit()
                        
                        # Signale zur Historie hinzufügen
                        signals_history.extend(current_signals)
                        if len(signals_history) > 50:
                            signals_history = signals_history[-50:]  # Nur letzte 50 behalten
                        
                        # Dashboard aktualisieren
                        dashboard = self.create_dashboard(signals_history)
                        live.update(dashboard)
                        live.refresh()
                        
                        # 60 Sekunden warten
                        for i in range(60):
                            await asyncio.sleep(1)
                        
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        self.logger.error(f"Fehler im Trading-Loop: {e}")
                        await asyncio.sleep(10)
                        
        except KeyboardInterrupt:
            self.console.print("\n🛑 [yellow]Trading Bot wird beendet...[/yellow]")
        finally:
            # Finale Statistiken
            stats = self.calculate_portfolio_stats()
            self.console.print("\n📊 [green]Finale Trading-Statistiken:[/green]")
            self.console.print(f"💰 Endwert: ${stats['total_value']:,.2f}")
            self.console.print(f"📈 Gesamtrendite: {stats['total_return_pct']:+.2f}%")
            self.console.print(f"📊 Trades: {self.total_trades}")
            self.console.print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")


async def main():
    """🎯 Hauptfunktion"""
    console = Console()
    
    console.print(Panel.fit("🤖 Binance Trading Bot v3.0", style="bold magenta"))
    console.print("⚠️  [red]WARNUNG: Echtes Trading kann zu Verlusten führen![/red]")
    console.print("💡 [yellow]Teste immer erst mit Paper Trading oder Testnet![/yellow]\n")
    
    # Trading-Modus wählen
    console.print("🔧 Wähle Trading-Modus:")
    console.print("1. Paper Trading (Simulation)")
    console.print("2. Testnet (Binance Testnet)")
    console.print("3. Live Trading (ECHTES GELD!)")
    
    choice = input("\nWähle (1-3): ").strip()
    
    if choice == "1":
        mode = TradingMode.PAPER
        api_key = None
        api_secret = None
    elif choice == "2":
        mode = TradingMode.TESTNET
        # Versuche API-Keys aus config.py zu laden
        try:
            import config
            if hasattr(config, 'BINANCE_TESTNET_API_KEY') and config.BINANCE_TESTNET_API_KEY != "dein_testnet_api_key_hier":
                api_key = config.BINANCE_TESTNET_API_KEY
                api_secret = config.BINANCE_TESTNET_SECRET
                console.print("\n✅ [green]Testnet API-Keys aus config.py geladen![/green]")
            else:
                console.print("\n🔑 Binance Testnet API-Keys benötigt:")
                api_key = input("API Key: ").strip() or None
                api_secret = input("API Secret: ").strip() or None
        except ImportError:
            console.print("\n🔑 Binance Testnet API-Keys benötigt:")
            api_key = input("API Key: ").strip() or None
            api_secret = input("API Secret: ").strip() or None
    elif choice == "3":
        mode = TradingMode.LIVE
        console.print("\n⚠️ [red]ACHTUNG: LIVE TRADING MODUS![/red]")
        console.print("🔑 Binance Live API-Keys benötigt:")
        api_key = input("API Key: ").strip() or None
        api_secret = input("API Secret: ").strip() or None
        
        confirm = input("\n⚠️ Bestätige LIVE TRADING (ja/nein): ").strip().lower()
        if confirm != "ja":
            console.print("❌ Live Trading abgebrochen.")
            return
    else:
        console.print("❌ Ungültige Auswahl. Verwende Paper Trading.")
        mode = TradingMode.PAPER
        api_key = None
        api_secret = None
    
    # Trading Bot initialisieren
    bot = BinanceTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        mode=mode,
        initial_balance=10000
    )
    
    console.print(f"\n🚀 Starte Trading Bot im {mode.value.upper()} Modus...")
    console.print("🔄 Drücke CTRL+C zum Beenden\n")
    
    try:
        await bot.trading_loop()
    except KeyboardInterrupt:
        console.print("\n👋 Trading Bot beendet.")


if __name__ == "__main__":
    if not BINANCE_AVAILABLE:
        print("📦 Installiere Binance Library:")
        print("pip install python-binance")
    else:
        asyncio.run(main())
