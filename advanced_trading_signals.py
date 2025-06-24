#!/usr/bin/env python3
"""
🎯 Advanced Trading Signal Generator mit Binance Integration
Autor: mad4cyber
Version: 2.0 - Enhanced Signal Edition

🚀 FEATURES:
- Erweiterte technische Signale (RSI, MACD, Bollinger Bands)
- AI-Powered Prognosen
- Multi-Timeframe Analyse 
- Binance API Integration
- Risk Management
- Real-Time Alerts
- Portfolio-optimierte Signale
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Trading Libraries
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# ML Libraries
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress
from rich import box

from ai_predictor import CryptoAIPredictor

class SignalStrength(Enum):
    """Signal-Stärke"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class SignalType(Enum):
    """Signal-Typen"""
    BUY = "BUY"
    SELL = "SELL"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    HOLD = "HOLD"

class TimeFrame(Enum):
    """Zeiträume"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class TradingSignal:
    """Enhanced Trading Signal"""
    symbol: str
    signal: SignalType
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timeframe: TimeFrame = TimeFrame.H1
    indicators: Dict[str, float] = None
    ai_prediction: Dict[str, float] = None
    reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.indicators is None:
            self.indicators = {}

class AdvancedTradingSignals:
    """🎯 Erweiterte Trading-Signal Generierung"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.console = Console()
        self.ai_predictor = CryptoAIPredictor()
        
        # Binance Client
        self.binance_client = None
        if BINANCE_AVAILABLE and api_key and api_secret:
            try:
                self.binance_client = Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet
                )
                self.console.print("✅ [green]Binance API verbunden![/green]")
            except Exception as e:
                self.console.print(f"❌ [red]Binance API Fehler: {e}[/red]")
        
        # Signal-Parameter
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_threshold = 0.8
        self.volume_threshold = 1.5
        self.min_confidence = 0.75
        
        # Risk Management
        self.default_stop_loss = 0.05  # 5%
        self.default_take_profit = 0.10  # 10%
        self.max_risk_per_trade = 0.02  # 2%
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradingSignals')
    
    def get_binance_klines(self, symbol: str, timeframe: TimeFrame, limit: int = 500) -> pd.DataFrame:
        """📊 Binance Kerzendaten abrufen"""
        if not self.binance_client:
            return pd.DataFrame()
        
        try:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=timeframe.value,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Datentypen konvertieren
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Binance-Daten für {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """📈 Erweiterte technische Indikatoren berechnen"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price Change
        df['price_change_pct'] = df['close'].pct_change()
        
        return df
    
    def generate_technical_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """🔧 Technische Trading-Signale generieren"""
        if df.empty or len(df) < 50:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        # RSI Signale
        if latest['rsi'] < self.rsi_oversold:
            signals['rsi'] = {'signal': 'BUY', 'strength': 0.8, 'reason': f"RSI überverkauft ({latest['rsi']:.1f})"}
        elif latest['rsi'] > self.rsi_overbought:
            signals['rsi'] = {'signal': 'SELL', 'strength': 0.8, 'reason': f"RSI überkauft ({latest['rsi']:.1f})"}
        else:
            signals['rsi'] = {'signal': 'NEUTRAL', 'strength': 0.3, 'reason': f"RSI neutral ({latest['rsi']:.1f})"}
        
        # MACD Signale
        if latest['macd'] > latest['macd_signal'] and df.iloc[-2]['macd'] <= df.iloc[-2]['macd_signal']:
            signals['macd'] = {'signal': 'BUY', 'strength': 0.7, 'reason': "MACD bullisches Crossover"}
        elif latest['macd'] < latest['macd_signal'] and df.iloc[-2]['macd'] >= df.iloc[-2]['macd_signal']:
            signals['macd'] = {'signal': 'SELL', 'strength': 0.7, 'reason': "MACD bärisches Crossover"}
        else:
            signals['macd'] = {'signal': 'NEUTRAL', 'strength': 0.2, 'reason': "MACD kein Crossover"}
        
        # Bollinger Bands Signale
        if latest['bb_position'] < 0.2:
            signals['bollinger'] = {'signal': 'BUY', 'strength': 0.6, 'reason': "Preis nahe unterer Bollinger Band"}
        elif latest['bb_position'] > 0.8:
            signals['bollinger'] = {'signal': 'SELL', 'strength': 0.6, 'reason': "Preis nahe oberer Bollinger Band"}
        else:
            signals['bollinger'] = {'signal': 'NEUTRAL', 'strength': 0.2, 'reason': "Preis in Bollinger Band Mitte"}
        
        # Moving Average Signale
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            signals['ma'] = {'signal': 'BUY', 'strength': 0.5, 'reason': "Bullischer MA Trend"}
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            signals['ma'] = {'signal': 'SELL', 'strength': 0.5, 'reason': "Bärischer MA Trend"}
        else:
            signals['ma'] = {'signal': 'NEUTRAL', 'strength': 0.2, 'reason': "Neutraler MA Trend"}
        
        # Stochastic Signale
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals['stochastic'] = {'signal': 'BUY', 'strength': 0.6, 'reason': "Stochastic überverkauft"}
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals['stochastic'] = {'signal': 'SELL', 'strength': 0.6, 'reason': "Stochastic überkauft"}
        else:
            signals['stochastic'] = {'signal': 'NEUTRAL', 'strength': 0.2, 'reason': "Stochastic neutral"}
        
        # Volume Signale
        if latest['volume_ratio'] > self.volume_threshold:
            signals['volume'] = {'signal': 'CONFIRM', 'strength': 0.4, 'reason': f"Hohes Volumen ({latest['volume_ratio']:.1f}x)"}
        else:
            signals['volume'] = {'signal': 'WEAK', 'strength': 0.1, 'reason': f"Niedriges Volumen ({latest['volume_ratio']:.1f}x)"}
        
        return signals
    
    async def get_ai_prediction(self, symbol: str) -> Dict[str, any]:
        """🧠 AI-Prognose abrufen"""
        try:
            # CoinGecko Symbol konvertieren
            coin_id = symbol.lower().replace('usdt', '').replace('btc', 'bitcoin').replace('eth', 'ethereum')
            if coin_id == 'bnb':
                coin_id = 'binancecoin'
            
            result = self.ai_predictor.predict_future_prices(coin_id)
            
            if 'error' not in result:
                return {
                    'current_price': result['current_price'],
                    'predicted_price': result['predicted_price'],
                    'price_change_pct': result['price_change_pct'],
                    'confidence': result['confidence']
                }
        except Exception as e:
            self.logger.error(f"AI-Prognose Fehler für {symbol}: {e}")
        
        return {}
    
    def calculate_risk_reward(self, entry_price: float, signal_type: SignalType) -> Tuple[float, float, float]:
        """💰 Stop-Loss, Take-Profit und Risk-Reward-Ratio berechnen"""
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = entry_price * (1 - self.default_stop_loss)
            take_profit = entry_price * (1 + self.default_take_profit)
        else:  # SELL signals
            stop_loss = entry_price * (1 + self.default_stop_loss)
            take_profit = entry_price * (1 - self.default_take_profit)
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return stop_loss, take_profit, risk_reward_ratio
    
    def combine_signals(self, technical_signals: Dict, ai_prediction: Dict) -> TradingSignal:
        """🎯 Signale intelligent kombinieren"""
        # Gewichtung der Signale
        weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bollinger': 0.15,
            'ma': 0.15,
            'stochastic': 0.1,
            'volume': 0.05,
            'ai': 0.1
        }
        
        buy_score = 0
        sell_score = 0
        
        # Technische Signale bewerten
        for indicator, signal_data in technical_signals.items():
            if indicator == 'volume':
                continue  # Volume bestätigt nur
                
            weight = weights.get(indicator, 0)
            strength = signal_data.get('strength', 0)
            
            if signal_data.get('signal') == 'BUY':
                buy_score += weight * strength
            elif signal_data.get('signal') == 'SELL':
                sell_score += weight * strength
        
        # AI-Prognose einbeziehen
        if ai_prediction:
            ai_weight = weights['ai']
            confidence = ai_prediction.get('confidence', 0)
            price_change = ai_prediction.get('price_change_pct', 0)
            
            if price_change > 2.0:  # > 2% erwarteter Anstieg
                buy_score += ai_weight * confidence
            elif price_change < -2.0:  # > 2% erwarteter Rückgang
                sell_score += ai_weight * confidence
        
        # Volume-Bestätigung
        volume_signal = technical_signals.get('volume', {})
        if volume_signal.get('signal') == 'CONFIRM':
            volume_multiplier = 1.2  # 20% Boost bei hohem Volumen
            buy_score *= volume_multiplier
            sell_score *= volume_multiplier
        
        # Signal-Typ bestimmen
        total_confidence = max(buy_score, sell_score)
        
        if buy_score > sell_score:
            if buy_score > 0.8:
                signal_type = SignalType.STRONG_BUY
                strength = SignalStrength.VERY_STRONG
            elif buy_score > 0.6:
                signal_type = SignalType.BUY
                strength = SignalStrength.STRONG
            elif buy_score > 0.1:  # Sehr niedrig für Testing
                signal_type = SignalType.BUY
                strength = SignalStrength.MODERATE
            else:
                signal_type = SignalType.HOLD
                strength = SignalStrength.WEAK
        elif sell_score > buy_score:
            if sell_score > 0.8:
                signal_type = SignalType.STRONG_SELL
                strength = SignalStrength.VERY_STRONG
            elif sell_score > 0.6:
                signal_type = SignalType.SELL
                strength = SignalStrength.STRONG
            elif sell_score > 0.1:  # Sehr niedrig für Testing
                signal_type = SignalType.SELL
                strength = SignalStrength.MODERATE
            else:
                signal_type = SignalType.HOLD
                strength = SignalStrength.WEAK
        else:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK
        
        return signal_type, strength, total_confidence
    
    async def generate_trading_signal(self, symbol: str, timeframe: TimeFrame = TimeFrame.H1) -> TradingSignal:
        """🎯 Hauptfunktion: Komplettes Trading-Signal generieren"""
        try:
            # Marktdaten abrufen
            df = self.get_binance_klines(symbol, timeframe, limit=500)
            if df.empty:
                self.logger.warning(f"Keine Daten für {symbol} verfügbar")
                return None
            
            # Technische Indikatoren berechnen
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Technische Signale generieren
            technical_signals = self.generate_technical_signals(df_with_indicators)
            
            # AI-Prognose abrufen
            ai_prediction = await self.get_ai_prediction(symbol)
            
            # Signale kombinieren
            signal_type, strength, confidence = self.combine_signals(technical_signals, ai_prediction)
            
            # Aktueller Preis
            current_price = df_with_indicators.iloc[-1]['close']
            
            # Risk Management
            stop_loss, take_profit, risk_reward_ratio = self.calculate_risk_reward(current_price, signal_type)
            
            # Grund zusammenstellen
            reasons = []
            for indicator, signal_data in technical_signals.items():
                if signal_data.get('strength', 0) > 0.5:
                    reasons.append(signal_data.get('reason', ''))
            
            if ai_prediction and ai_prediction.get('confidence', 0) > 0.7:
                change = ai_prediction.get('price_change_pct', 0)
                reasons.append(f"AI: {change:+.1f}% ({ai_prediction.get('confidence', 0):.1%})")
            
            reason = " | ".join(reasons[:3])  # Top 3 Gründe
            
            # Trading Signal erstellen
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                timeframe=timeframe,
                indicators=technical_signals,
                ai_prediction=ai_prediction,
                reason=reason
            )
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Fehler bei Signal-Generierung für {symbol}: {e}")
            return None
    
    async def scan_multiple_symbols(self, symbols: List[str], timeframe: TimeFrame = TimeFrame.H1) -> List[TradingSignal]:
        """🔍 Multiple Symbole gleichzeitig scannen"""
        signals = []
        
        self.logger.info(f"🔍 Scanne {len(symbols)} Symbole für Trading-Signale...")
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"📊 Analysiere {symbol} ({i}/{len(symbols)})...")
            signal = await self.generate_trading_signal(symbol, timeframe)
            if signal:
                signals.append(signal)
                self.logger.info(f"✅ Signal gefunden für {symbol}: {signal.signal.value} (Konfidenz: {signal.confidence:.1%})")
        
        # Nach Konfidenz sortieren
        signals.sort(key=lambda x: x.confidence, reverse=True)
        self.logger.info(f"✅ {len(signals)} Trading-Signale generiert!")
        return signals
    
    def display_signals(self, signals: List[TradingSignal]):
        """📊 Trading-Signale anzeigen"""
        self.console.print(Panel.fit("🎯 Advanced Trading Signals", style="bold magenta"))
        
        # Haupt-Signale Tabelle
        table = Table(title="🎯 Trading Signals Dashboard", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Signal", style="white")
        table.add_column("Stärke", style="yellow")
        table.add_column("Konfidenz", style="green")
        table.add_column("Preis", style="white")
        table.add_column("Stop Loss", style="red")
        table.add_column("Take Profit", style="green")
        table.add_column("R/R", style="yellow")
        table.add_column("Grund", style="dim")
        
        for signal in signals[:10]:  # Top 10 Signale
            # Signal-Styling
            if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                signal_color = "green"
                signal_icon = "🚀" if signal.signal == SignalType.STRONG_BUY else "📈"
            elif signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                signal_color = "red"
                signal_icon = "💥" if signal.signal == SignalType.STRONG_SELL else "📉"
            else:
                signal_color = "yellow"
                signal_icon = "⏸️"
            
            # Stärke-Styling
            strength_icons = {
                SignalStrength.VERY_STRONG: "🔥🔥🔥",
                SignalStrength.STRONG: "🔥🔥",
                SignalStrength.MODERATE: "🔥",
                SignalStrength.WEAK: "💧"
            }
            
            table.add_row(
                signal.symbol,
                f"[{signal_color}]{signal_icon} {signal.signal.value}[/{signal_color}]",
                strength_icons.get(signal.strength, ""),
                f"{signal.confidence:.1%}",
                f"${signal.entry_price:.4f}",
                f"${signal.stop_loss:.4f}" if signal.stop_loss else "-",
                f"${signal.take_profit:.4f}" if signal.take_profit else "-",
                f"{signal.risk_reward_ratio:.1f}" if signal.risk_reward_ratio else "-",
                signal.reason[:40] + "..." if len(signal.reason) > 40 else signal.reason
            )
        
        self.console.print(table)


async def main():
    """🎯 Hauptfunktion für Trading Signal Generator"""
    console = Console()
    
    console.print(Panel.fit("🎯 Advanced Trading Signal Generator", style="bold magenta"))
    console.print("⚠️  [yellow]WICHTIG: Verwende Testnet/Paper Trading für Tests![/yellow]")
    console.print("💡 [dim]Echte API-Keys nur nach gründlichen Tests verwenden![/dim]\n")
    
    # API-Keys (für Testnet)
    api_key = None  # Hier deine Binance Testnet API Key
    api_secret = None  # Hier dein Binance Testnet Secret
    
    # Signal Generator initialisieren
    signal_generator = AdvancedTradingSignals(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )
    
    # Top Krypto-Symbole für Binance
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'
    ]
    
    console.print(f"🔍 Scanne {len(symbols)} Symbole für Trading-Signale...")
    console.print("⏰ Das kann einige Minuten dauern...\n")
    
    # Signale generieren
    signals = await signal_generator.scan_multiple_symbols(symbols, TimeFrame.H1)
    
    # Ergebnisse anzeigen
    if signals:
        signal_generator.display_signals(signals)
        
        console.print(f"\n✅ [green]{len(signals)} Trading-Signale generiert![/green]")
        console.print("🎯 [cyan]Signale sind nach Konfidenz sortiert[/cyan]")
        
        # Top Signal Details
        if signals:
            top_signal = signals[0]
            console.print(f"\n🏆 [bold]Top Signal: {top_signal.symbol}[/bold]")
            console.print(f"📊 Signal: {top_signal.signal.value}")
            console.print(f"🎯 Konfidenz: {top_signal.confidence:.1%}")
            console.print(f"💰 Entry: ${top_signal.entry_price:.4f}")
            if top_signal.stop_loss:
                console.print(f"🛑 Stop Loss: ${top_signal.stop_loss:.4f}")
            if top_signal.take_profit:
                console.print(f"🎯 Take Profit: ${top_signal.take_profit:.4f}")
            console.print(f"⚖️ Risk/Reward: {top_signal.risk_reward_ratio:.1f}")
    else:
        console.print("❌ [red]Keine Signale generiert. Prüfe API-Verbindung.[/red]")


if __name__ == "__main__":
    if not BINANCE_AVAILABLE:
        print("📦 Installiere Binance Library:")
        print("pip install python-binance")
    else:
        asyncio.run(main())
