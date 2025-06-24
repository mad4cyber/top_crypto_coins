#!/usr/bin/env python3
"""
🤖 AI-Powered Trading Bot für Kryptowährungen
Autor: mad4cyber
Version: 4.1 - Trading Edition
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box

from ai_predictor import CryptoAIPredictor

class TradingStrategy(Enum):
    """Trading-Strategien"""
    AI_PREDICTION = "ai_prediction"
    MOMENTUM = "momentum" 
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"

class OrderType(Enum):
    """Order-Typen"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class TradingSignal:
    """Trading-Signal Klasse"""
    def __init__(self, symbol: str, signal: OrderType, confidence: float, 
                 price: float, reason: str, timestamp: datetime = None):
        self.symbol = symbol
        self.signal = signal
        self.confidence = confidence
        self.price = price
        self.reason = reason
        self.timestamp = timestamp or datetime.now()

class CryptoTradingBot:
    """🤖 AI-Powered Kryptowährungs Trading Bot"""
    
    def __init__(self, initial_balance: float = 10000):
        self.console = Console()
        self.ai_predictor = CryptoAIPredictor()
        
        # Portfolio
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # {symbol: amount}
        self.trade_history = []
        
        # Trading-Einstellungen
        self.risk_per_trade = 0.02  # 2% Risiko pro Trade
        self.min_confidence = 0.7   # 70% Mindest-Konfidenz
        self.stop_loss = 0.05       # 5% Stop-Loss
        self.take_profit = 0.10     # 10% Take-Profit
        
        # Überwachte Coins
        self.watchlist = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradingBot')
        
    def analyze_market_conditions(self) -> Dict[str, any]:
        """📊 Marktbedingungen analysieren"""
        try:
            # Hier würde normalerweise eine umfassende Marktanalyse stattfinden
            # Fear & Greed Index, VIX, etc.
            
            # Vereinfachte Simulation
            market_conditions = {
                'trend': 'bullish',  # bullish, bearish, sideways
                'volatility': 'medium',  # low, medium, high
                'volume': 'high',
                'fear_greed_index': 65,  # 0-100
                'sentiment': 'optimistic'
            }
            
            return market_conditions
            
        except Exception as e:
            self.logger.error(f"Fehler bei Marktanalyse: {e}")
            return {}
    
    async def generate_ai_signals(self, symbol: str) -> Optional[TradingSignal]:
        """🧠 AI-basierte Trading-Signale generieren"""
        try:
            # AI-Prognose abrufen (ohne Rich Progress für Trading Bot)
            # Erstelle temporäre AI-Instanz ohne Progress Bar
            temp_predictor = CryptoAIPredictor()
            temp_predictor.console = None  # Deaktiviere Rich Console
            
            # Vereinfachte Prognose ohne Progress Bar
            df = temp_predictor.fetch_historical_data(symbol, days=365)
            if df.empty:
                return None
                
            df_features = temp_predictor.create_features(df)
            ml_results = temp_predictor.train_ml_models(df_features, target_days=1)
            
            if not ml_results:
                return None
                
            # Aktuelle Features für Prognose
            current_features = df_features.iloc[-1][ml_results['features']].values.reshape(1, -1)
            current_features_scaled = ml_results['scaler'].transform(current_features)
            
            # Prognosen von verschiedenen Modellen
            predictions = {}
            for model_name, model in ml_results['models'].items():
                pred = model.predict(current_features_scaled)[0]
                predictions[model_name] = pred
            
            # Ensemble-Prognose (Durchschnitt)
            ensemble_prediction = sum(predictions.values()) / len(predictions)
            
            current_price = df['price'].iloc[-1]
            price_change_pct = ((ensemble_prediction - current_price) / current_price) * 100
            confidence = max(ml_results['performance'][model]['r2'] for model in ml_results['performance'])
            
            prediction = {
                'current_price': current_price,
                'predicted_price': ensemble_prediction,
                'price_change_pct': price_change_pct,
                'confidence': confidence
            }
            
            if 'error' in prediction:
                return None
            
            current_price = prediction['current_price']
            predicted_price = prediction['predicted_price']
            price_change_pct = prediction['price_change_pct']
            confidence = prediction['confidence']
            
            # Trading-Signal basierend auf AI-Prognose
            if confidence < self.min_confidence:
                return TradingSignal(
                    symbol=symbol.upper(),
                    signal=OrderType.HOLD,
                    confidence=confidence,
                    price=current_price,
                    reason=f"Niedrige AI-Konfidenz: {confidence:.1%}"
                )
            
            # Buy-Signal
            if price_change_pct > 2.0:  # > 2% erwarteter Anstieg
                return TradingSignal(
                    symbol=symbol.upper(),
                    signal=OrderType.BUY,
                    confidence=confidence,
                    price=current_price,
                    reason=f"AI prognostiziert +{price_change_pct:.1f}% (Konfidenz: {confidence:.1%})"
                )
            
            # Sell-Signal  
            elif price_change_pct < -2.0:  # > 2% erwarteter Rückgang
                return TradingSignal(
                    symbol=symbol.upper(),
                    signal=OrderType.SELL,
                    confidence=confidence,
                    price=current_price,
                    reason=f"AI prognostiziert {price_change_pct:.1f}% (Konfidenz: {confidence:.1%})"
                )
            
            # Hold-Signal
            else:
                return TradingSignal(
                    symbol=symbol.upper(),
                    signal=OrderType.HOLD,
                    confidence=confidence,
                    price=current_price,
                    reason=f"Neutrale AI-Prognose: {price_change_pct:.1f}%"
                )
                
        except Exception as e:
            self.logger.error(f"Fehler bei AI-Signal-Generierung für {symbol}: {e}")
            return None
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """💰 Positionsgröße berechnen"""
        # Kelly-Kriterium mit Risikomanagement
        max_risk_amount = self.balance * self.risk_per_trade
        
        # Anpassung basierend auf Konfidenz
        confidence_multiplier = min(signal.confidence / self.min_confidence, 2.0)
        position_value = max_risk_amount * confidence_multiplier
        
        # Maximale Position: 20% des Portfolios
        max_position = self.balance * 0.20
        position_value = min(position_value, max_position)
        
        # In Coin-Anzahl umrechnen
        position_size = position_value / signal.price
        
        return position_size
    
    async def execute_trade(self, signal: TradingSignal) -> bool:
        """💱 Trade ausführen (Simulation)"""
        try:
            symbol = signal.symbol
            
            if signal.signal == OrderType.BUY:
                # Kaufen
                position_size = self.calculate_position_size(signal)
                cost = position_size * signal.price
                
                if cost > self.balance:
                    self.logger.warning(f"Nicht genug Balance für {symbol} Trade")
                    return False
                
                # Position eröffnen/erweitern
                if symbol in self.positions:
                    self.positions[symbol] += position_size
                else:
                    self.positions[symbol] = position_size
                
                self.balance -= cost
                
                # Trade-Historie
                trade = {
                    'timestamp': signal.timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'amount': position_size,
                    'price': signal.price,
                    'cost': cost,
                    'reason': signal.reason,
                    'confidence': signal.confidence
                }
                self.trade_history.append(trade)
                
                self.logger.info(f"✅ GEKAUFT: {position_size:.4f} {symbol} @ €{signal.price:.2f}")
                return True
                
            elif signal.signal == OrderType.SELL:
                # Verkaufen
                if symbol not in self.positions or self.positions[symbol] <= 0:
                    self.logger.warning(f"Keine Position in {symbol} zum Verkaufen")
                    return False
                
                position_size = self.positions[symbol]
                revenue = position_size * signal.price
                
                # Position schließen
                self.positions[symbol] = 0
                self.balance += revenue
                
                # Trade-Historie
                trade = {
                    'timestamp': signal.timestamp,
                    'symbol': symbol,
                    'action': 'SELL',
                    'amount': position_size,
                    'price': signal.price,
                    'revenue': revenue,
                    'reason': signal.reason,
                    'confidence': signal.confidence
                }
                self.trade_history.append(trade)
                
                self.logger.info(f"✅ VERKAUFT: {position_size:.4f} {symbol} @ €{signal.price:.2f}")
                return True
                
            return True  # HOLD
            
        except Exception as e:
            self.logger.error(f"Fehler bei Trade-Ausführung: {e}")
            return False
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """💰 Portfolio-Wert berechnen"""
        portfolio_value = self.balance  # Cash
        
        for symbol, amount in self.positions.items():
            if amount > 0 and symbol in current_prices:
                portfolio_value += amount * current_prices[symbol]
        
        return {
            'total_value': portfolio_value,
            'cash': self.balance,
            'total_return': ((portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'positions_value': portfolio_value - self.balance
        }
    
    def create_dashboard(self, signals: List[TradingSignal], portfolio_stats: Dict) -> Layout:
        """📊 Trading-Dashboard erstellen"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="signals"),
            Layout(name="portfolio")
        )
        
        # Header
        header = Panel(
            "🤖 AI TRADING BOT v4.1 - Live Dashboard",
            style="bold magenta"
        )
        layout["header"].update(header)
        
        # Signals
        signals_table = Table(title="🧠 AI Trading Signals", box=box.ROUNDED)
        signals_table.add_column("Symbol", style="cyan")
        signals_table.add_column("Signal", style="white")
        signals_table.add_column("Konfidenz", style="green")
        signals_table.add_column("Preis", style="yellow")
        signals_table.add_column("Grund", style="dim")
        
        for signal in signals[-10:]:  # Letzte 10 Signale
            signal_color = "green" if signal.signal == OrderType.BUY else "red" if signal.signal == OrderType.SELL else "yellow"
            signal_icon = "🚀" if signal.signal == OrderType.BUY else "📉" if signal.signal == OrderType.SELL else "⏸️"
            
            signals_table.add_row(
                signal.symbol,
                f"[{signal_color}]{signal_icon} {signal.signal.value.upper()}[/{signal_color}]",
                f"{signal.confidence:.1%}",
                f"€{signal.price:.2f}",
                signal.reason[:30] + "..." if len(signal.reason) > 30 else signal.reason
            )
        
        layout["signals"].update(Panel(signals_table, title="Trading Signals"))
        
        # Portfolio
        portfolio_table = Table(title="💼 Portfolio Status", box=box.SIMPLE)
        portfolio_table.add_column("Metrik", style="cyan")
        portfolio_table.add_column("Wert", style="white")
        
        portfolio_table.add_row("💰 Gesamtwert", f"€{portfolio_stats['total_value']:,.2f}")
        portfolio_table.add_row("💵 Cash", f"€{portfolio_stats['cash']:,.2f}")
        portfolio_table.add_row("📊 Positionen", f"€{portfolio_stats['positions_value']:,.2f}")
        
        return_color = "green" if portfolio_stats['total_return'] >= 0 else "red"
        return_icon = "📈" if portfolio_stats['total_return'] >= 0 else "📉"
        portfolio_table.add_row(
            "📈 Gesamtrendite", 
            f"[{return_color}]{return_icon} {portfolio_stats['total_return']:+.2f}%[/{return_color}]"
        )
        
        layout["portfolio"].update(Panel(portfolio_table, title="Portfolio"))
        
        # Footer
        footer = Panel(
            f"🕐 Letztes Update: {datetime.now().strftime('%H:%M:%S')} | 🔄 Nächste Analyse in 60s",
            style="dim"
        )
        layout["footer"].update(footer)
        
        return layout
    
    async def run_trading_loop(self):
        """🔄 Haupt-Trading-Loop"""
        signals_history = []
        
        try:
            with Live(auto_refresh=False) as live:
                while True:
                    try:
                        # Marktbedingungen analysieren
                        market_conditions = self.analyze_market_conditions()
                        
                        # AI-Signale für alle Watchlist-Coins generieren
                        current_signals = []
                        current_prices = {}
                        
                        for symbol in self.watchlist:
                            signal = await self.generate_ai_signals(symbol)
                            if signal:
                                current_signals.append(signal)
                                current_prices[signal.symbol] = signal.price
                                
                                # Trade ausführen falls Signal stark genug
                                if signal.confidence >= self.min_confidence and signal.signal != OrderType.HOLD:
                                    await self.execute_trade(signal)
                        
                        # Portfolio-Statistiken berechnen
                        portfolio_stats = self.calculate_portfolio_value(current_prices)
                        
                        # Signale zur Historie hinzufügen
                        signals_history.extend(current_signals)
                        
                        # Dashboard aktualisieren
                        dashboard = self.create_dashboard(signals_history, portfolio_stats)
                        live.update(dashboard)
                        live.refresh()
                        
                        # 60 Sekunden warten mit besserer Cancellation
                        for i in range(60):
                            await asyncio.sleep(1)
                        
                    except asyncio.CancelledError:
                        self.console.print("\n🛑 [yellow]Trading Bot wird beendet...[/yellow]")
                        break
                    except Exception as e:
                        self.logger.error(f"Fehler im Trading-Loop: {e}")
                        # Kürzere Wartezeit bei Fehlern
                        for i in range(10):
                            await asyncio.sleep(1)
                            
        except KeyboardInterrupt:
            self.console.print("\n🛑 [red]Trading Bot durch Benutzer gestoppt[/red]")
        except Exception as e:
            self.console.print(f"\n❌ [red]Kritischer Fehler: {e}[/red]")
        finally:
            # Cleanup
            self.console.print("\n📊 [green]Finale Trading-Statistiken:[/green]")
            self.console.print(f"💰 Endkapital: €{self.balance:,.2f}")
            self.console.print(f"📈 Trades ausgeführt: {len(self.trade_history)}")
            if self.trade_history:
                profit_loss = self.balance - self.initial_balance
                profit_loss_pct = (profit_loss / self.initial_balance) * 100
                color = "green" if profit_loss >= 0 else "red"
                self.console.print(f"📊 [{color}]Gesamtgewinn/-verlust: €{profit_loss:+,.2f} ({profit_loss_pct:+.2f}%)[/{color}]")


async def main():
    """🤖 Trading Bot Hauptfunktion"""
    bot = CryptoTradingBot(initial_balance=10000)
    
    print("🚀 Starte AI Trading Bot...")
    print("💰 Startkapital: €10,000")
    print("📊 Überwachte Coins:", ", ".join(bot.watchlist))
    print("🎯 Mindest-Konfidenz:", f"{bot.min_confidence:.0%}")
    print("\n⚠️  ACHTUNG: Dies ist eine SIMULATION für Lernzwecke!")
    print("💡 Verwende niemals echtes Geld ohne gründliche Tests!")
    print("\n🔄 Drücke CTRL+C zum Beenden\n")
    
    input("Drücke Enter um zu starten...")
    
    try:
        await bot.run_trading_loop()
    except KeyboardInterrupt:
        print("\n👋 Trading Bot beendet.")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")


if __name__ == "__main__":
    asyncio.run(main())
