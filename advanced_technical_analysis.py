#!/usr/bin/env python3
"""
📊 Advanced Technical Analysis Suite
Autor: mad4cyber
Version: 1.0 - Technical Analysis Edition

🚀 FEATURES:
- Advanced Technical Indicators
- Pattern Recognition & Detection
- Multi-Timeframe Analysis
- Signal Generation & Scoring
- Trend Analysis & Forecasting
- Support/Resistance Detection
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import requests
import warnings
warnings.filterwarnings('ignore')

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"

class PatternType(Enum):
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    CUP_AND_HANDLE = "cup_and_handle"

class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class TechnicalIndicator:
    """📊 Technical Indicator Result"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral'
    strength: SignalStrength
    timeframe: str
    timestamp: str
    interpretation: str
    confidence: float

@dataclass
class PatternDetection:
    """🔍 Pattern Detection Result"""
    pattern_type: PatternType
    confidence: float
    start_date: str
    end_date: str
    support_level: float
    resistance_level: float
    breakout_target: float
    risk_reward_ratio: float
    description: str
    signal: str
    timeframe: str

@dataclass
class SupportResistance:
    """📏 Support/Resistance Level"""
    level: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1
    touches: int
    last_touch: str
    timeframe: str
    confidence: float

@dataclass
class TrendAnalysis:
    """📈 Trend Analysis Result"""
    timeframe: str
    direction: TrendDirection
    strength: float
    duration_days: int
    start_price: float
    current_price: float
    change_percent: float
    confidence: float
    support_levels: List[float]
    resistance_levels: List[float]
    trend_line_slope: float

@dataclass
class TechnicalSignal:
    """🎯 Comprehensive Technical Signal"""
    timestamp: str
    coin_id: str
    coin_symbol: str
    timeframe: str
    overall_signal: str  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    signal_strength: float
    confidence: float
    indicators: List[TechnicalIndicator]
    patterns: List[PatternDetection]
    trend_analysis: TrendAnalysis
    support_resistance: List[SupportResistance]
    price_targets: Dict[str, float]
    risk_assessment: str
    recommended_action: str

class AdvancedTechnicalAnalysis:
    """📊 Advanced Technical Analysis System"""
    
    def __init__(self, data_file: str = "technical_analysis_data.json"):
        self.data_file = data_file
        self.analysis_results: Dict[str, List[TechnicalSignal]] = {}
        self.indicator_history: Dict[str, List[TechnicalIndicator]] = {}
        self.pattern_history: List[PatternDetection] = []
        self.support_resistance_levels: Dict[str, List[SupportResistance]] = {}
        
        # Configuration
        self.analysis_active = False
        self.analysis_thread = None
        self.timeframes = ['1h', '4h', '1d', '1w']
        self.analysis_interval = 300  # 5 minutes
        
        # Load existing data
        self.load_data()
    
    def get_price_data(self, coin_id: str, timeframe: str = '1d', days: int = 100) -> pd.DataFrame:
        """📈 Get historical price data"""
        try:
            # Simulate price data for demo (would use real API in production)
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Create realistic OHLCV data
            np.random.seed(42)  # For reproducible demo data
            
            base_price = np.random.uniform(100, 50000)  # Base price
            prices = []
            volume = []
            
            for i in range(days):
                # Add trend and volatility
                trend = np.sin(i * 0.1) * 0.02  # Long term trend
                volatility = np.random.normal(0, 0.03)  # Daily volatility
                
                if i == 0:
                    open_price = base_price
                else:
                    open_price = prices[-1]['close']
                
                change = open_price * (trend + volatility)
                high = open_price + abs(change) + np.random.uniform(0, open_price * 0.02)
                low = open_price - abs(change) - np.random.uniform(0, open_price * 0.02)
                close = open_price + change
                vol = np.random.uniform(1000000, 10000000)
                
                prices.append({
                    'open': max(0.01, open_price),
                    'high': max(0.01, high),
                    'low': max(0.01, low),
                    'close': max(0.01, close),
                })
                volume.append(vol)
            
            df = pd.DataFrame(prices, index=dates)
            df['volume'] = volume
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting price data for {coin_id}: {e}")
            return pd.DataFrame()
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """📊 Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """📈 Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """⚡ Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """📊 MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """📊 Bollinger Bands"""
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """🎯 Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """📏 Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """💪 Average Directional Index"""
        tr = self.calculate_atr(high, low, close, 1)
        
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        dm_plus[(dm_plus - dm_minus) < 0] = 0
        dm_minus[(dm_minus - dm_plus) < 0] = 0
        
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / tr.rolling(window=period).mean())
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """🎭 Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """🌀 Commodity Channel Index"""
        tp = (high + low + close) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return cci
    
    def analyze_indicators(self, coin_id: str, timeframe: str) -> List[TechnicalIndicator]:
        """🔍 Comprehensive Indicator Analysis"""
        indicators = []
        
        try:
            df = self.get_price_data(coin_id, timeframe)
            if df.empty:
                return indicators
            
            current_price = df['close'].iloc[-1]
            
            # RSI Analysis
            rsi = self.calculate_rsi(df['close'])
            rsi_value = rsi.iloc[-1]
            
            if rsi_value > 70:
                rsi_signal = 'sell'
                rsi_strength = SignalStrength.STRONG
                rsi_interpretation = f"Overbought (RSI: {rsi_value:.1f}) - Potential sell signal"
            elif rsi_value < 30:
                rsi_signal = 'buy'
                rsi_strength = SignalStrength.STRONG
                rsi_interpretation = f"Oversold (RSI: {rsi_value:.1f}) - Potential buy signal"
            else:
                rsi_signal = 'neutral'
                rsi_strength = SignalStrength.WEAK
                rsi_interpretation = f"Neutral zone (RSI: {rsi_value:.1f})"
            
            indicators.append(TechnicalIndicator(
                name="RSI",
                value=rsi_value,
                signal=rsi_signal,
                strength=rsi_strength,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                interpretation=rsi_interpretation,
                confidence=0.7
            ))
            
            # MACD Analysis
            macd_data = self.calculate_macd(df['close'])
            macd_value = macd_data['macd'].iloc[-1]
            signal_value = macd_data['signal'].iloc[-1]
            histogram = macd_data['histogram'].iloc[-1]
            
            if macd_value > signal_value and histogram > 0:
                macd_signal = 'buy'
                macd_strength = SignalStrength.MODERATE
                macd_interpretation = "MACD bullish crossover - Buy signal"
            elif macd_value < signal_value and histogram < 0:
                macd_signal = 'sell'
                macd_strength = SignalStrength.MODERATE
                macd_interpretation = "MACD bearish crossover - Sell signal"
            else:
                macd_signal = 'neutral'
                macd_strength = SignalStrength.WEAK
                macd_interpretation = "MACD in neutral position"
            
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=macd_value,
                signal=macd_signal,
                strength=macd_strength,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                interpretation=macd_interpretation,
                confidence=0.6
            ))
            
            # Bollinger Bands Analysis
            bb_data = self.calculate_bollinger_bands(df['close'])
            upper_band = bb_data['upper'].iloc[-1]
            lower_band = bb_data['lower'].iloc[-1]
            middle_band = bb_data['middle'].iloc[-1]
            
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
            
            if bb_position > 0.8:
                bb_signal = 'sell'
                bb_strength = SignalStrength.MODERATE
                bb_interpretation = "Price near upper Bollinger Band - Potential reversal"
            elif bb_position < 0.2:
                bb_signal = 'buy'
                bb_strength = SignalStrength.MODERATE
                bb_interpretation = "Price near lower Bollinger Band - Potential bounce"
            else:
                bb_signal = 'neutral'
                bb_strength = SignalStrength.WEAK
                bb_interpretation = "Price within normal Bollinger Band range"
            
            indicators.append(TechnicalIndicator(
                name="Bollinger_Bands",
                value=bb_position,
                signal=bb_signal,
                strength=bb_strength,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                interpretation=bb_interpretation,
                confidence=0.5
            ))
            
            # Stochastic Analysis
            stoch_data = self.calculate_stochastic(df['high'], df['low'], df['close'])
            stoch_k = stoch_data['k'].iloc[-1]
            stoch_d = stoch_data['d'].iloc[-1]
            
            if stoch_k > 80 and stoch_d > 80:
                stoch_signal = 'sell'
                stoch_strength = SignalStrength.MODERATE
                stoch_interpretation = "Stochastic overbought - Sell signal"
            elif stoch_k < 20 and stoch_d < 20:
                stoch_signal = 'buy'
                stoch_strength = SignalStrength.MODERATE
                stoch_interpretation = "Stochastic oversold - Buy signal"
            else:
                stoch_signal = 'neutral'
                stoch_strength = SignalStrength.WEAK
                stoch_interpretation = "Stochastic in neutral zone"
            
            indicators.append(TechnicalIndicator(
                name="Stochastic",
                value=stoch_k,
                signal=stoch_signal,
                strength=stoch_strength,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                interpretation=stoch_interpretation,
                confidence=0.6
            ))
            
            # ADX Analysis (Trend Strength)
            adx_data = self.calculate_adx(df['high'], df['low'], df['close'])
            adx_value = adx_data['adx'].iloc[-1]
            di_plus = adx_data['di_plus'].iloc[-1]
            di_minus = adx_data['di_minus'].iloc[-1]
            
            if adx_value > 40:
                if di_plus > di_minus:
                    adx_signal = 'buy'
                    adx_interpretation = f"Strong bullish trend (ADX: {adx_value:.1f})"
                else:
                    adx_signal = 'sell'
                    adx_interpretation = f"Strong bearish trend (ADX: {adx_value:.1f})"
                adx_strength = SignalStrength.STRONG
            elif adx_value > 25:
                adx_signal = 'neutral'
                adx_strength = SignalStrength.MODERATE
                adx_interpretation = f"Moderate trend strength (ADX: {adx_value:.1f})"
            else:
                adx_signal = 'neutral'
                adx_strength = SignalStrength.WEAK
                adx_interpretation = f"Weak or no trend (ADX: {adx_value:.1f})"
            
            indicators.append(TechnicalIndicator(
                name="ADX",
                value=adx_value,
                signal=adx_signal,
                strength=adx_strength,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat(),
                interpretation=adx_interpretation,
                confidence=0.8
            ))
            
        except Exception as e:
            print(f"❌ Error analyzing indicators for {coin_id}: {e}")
        
        return indicators
    
    def detect_patterns(self, coin_id: str, timeframe: str) -> List[PatternDetection]:
        """🔍 Advanced Pattern Detection"""
        patterns = []
        
        try:
            df = self.get_price_data(coin_id, timeframe, days=50)
            if len(df) < 20:
                return patterns
            
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Double Top Detection
            pattern = self._detect_double_top(highs, closes)
            if pattern:
                patterns.append(pattern)
            
            # Double Bottom Detection
            pattern = self._detect_double_bottom(lows, closes)
            if pattern:
                patterns.append(pattern)
            
            # Head and Shoulders Detection
            pattern = self._detect_head_and_shoulders(highs, closes)
            if pattern:
                patterns.append(pattern)
            
            # Triangle Patterns
            ascending_triangle = self._detect_ascending_triangle(highs, lows)
            if ascending_triangle:
                patterns.append(ascending_triangle)
            
            descending_triangle = self._detect_descending_triangle(highs, lows)
            if descending_triangle:
                patterns.append(descending_triangle)
            
        except Exception as e:
            print(f"❌ Error detecting patterns for {coin_id}: {e}")
        
        return patterns
    
    def _detect_double_top(self, highs: np.ndarray, closes: np.ndarray) -> Optional[PatternDetection]:
        """🔍 Detect Double Top Pattern"""
        if len(highs) < 20:
            return None
        
        try:
            # Find peaks
            peaks = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 2:
                return None
            
            # Look for two peaks at similar levels
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1_idx, peak1_price = peaks[i]
                    peak2_idx, peak2_price = peaks[j]
                    
                    # Check if peaks are at similar levels (within 5%)
                    if abs(peak1_price - peak2_price) / peak1_price < 0.05 and peak2_idx - peak1_idx > 5:
                        # Find valley between peaks
                        valley_start = peak1_idx
                        valley_end = peak2_idx
                        valley_low = min(highs[valley_start:valley_end])
                        
                        resistance_level = (peak1_price + peak2_price) / 2
                        support_level = valley_low
                        current_price = closes[-1]
                        
                        # Calculate breakout target (pattern height)
                        pattern_height = resistance_level - support_level
                        breakout_target = support_level - pattern_height
                        
                        confidence = 0.7 if abs(peak1_price - peak2_price) / peak1_price < 0.02 else 0.5
                        
                        return PatternDetection(
                            pattern_type=PatternType.DOUBLE_TOP,
                            confidence=confidence,
                            start_date=datetime.now().isoformat(),
                            end_date=datetime.now().isoformat(),
                            support_level=support_level,
                            resistance_level=resistance_level,
                            breakout_target=breakout_target,
                            risk_reward_ratio=(current_price - breakout_target) / (resistance_level - current_price),
                            description=f"Double Top pattern detected at {resistance_level:.2f}",
                            signal="sell" if current_price > support_level else "neutral",
                            timeframe="1d"
                        )
            
        except Exception as e:
            print(f"❌ Error in double top detection: {e}")
        
        return None
    
    def _detect_double_bottom(self, lows: np.ndarray, closes: np.ndarray) -> Optional[PatternDetection]:
        """🔍 Detect Double Bottom Pattern"""
        if len(lows) < 20:
            return None
        
        try:
            # Find valleys
            valleys = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                    valleys.append((i, lows[i]))
            
            if len(valleys) < 2:
                return None
            
            # Look for two valleys at similar levels
            for i in range(len(valleys) - 1):
                for j in range(i + 1, len(valleys)):
                    valley1_idx, valley1_price = valleys[i]
                    valley2_idx, valley2_price = valleys[j]
                    
                    # Check if valleys are at similar levels (within 5%)
                    if abs(valley1_price - valley2_price) / valley1_price < 0.05 and valley2_idx - valley1_idx > 5:
                        # Find peak between valleys
                        peak_start = valley1_idx
                        peak_end = valley2_idx
                        peak_high = max(lows[peak_start:peak_end])
                        
                        support_level = (valley1_price + valley2_price) / 2
                        resistance_level = peak_high
                        current_price = closes[-1]
                        
                        # Calculate breakout target
                        pattern_height = resistance_level - support_level
                        breakout_target = resistance_level + pattern_height
                        
                        confidence = 0.7 if abs(valley1_price - valley2_price) / valley1_price < 0.02 else 0.5
                        
                        return PatternDetection(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            confidence=confidence,
                            start_date=datetime.now().isoformat(),
                            end_date=datetime.now().isoformat(),
                            support_level=support_level,
                            resistance_level=resistance_level,
                            breakout_target=breakout_target,
                            risk_reward_ratio=(breakout_target - current_price) / (current_price - support_level),
                            description=f"Double Bottom pattern detected at {support_level:.2f}",
                            signal="buy" if current_price < resistance_level else "neutral",
                            timeframe="1d"
                        )
            
        except Exception as e:
            print(f"❌ Error in double bottom detection: {e}")
        
        return None
    
    def _detect_head_and_shoulders(self, highs: np.ndarray, closes: np.ndarray) -> Optional[PatternDetection]:
        """🔍 Detect Head and Shoulders Pattern"""
        if len(highs) < 25:
            return None
        
        try:
            # Find peaks
            peaks = []
            for i in range(3, len(highs) - 3):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and 
                    highs[i] > highs[i-2] and highs[i] > highs[i+2] and
                    highs[i] > highs[i-3] and highs[i] > highs[i+3]):
                    peaks.append((i, highs[i]))
            
            if len(peaks) < 3:
                return None
            
            # Look for H&S pattern (3 peaks with middle one highest)
            for i in range(len(peaks) - 2):
                left_shoulder_idx, left_shoulder = peaks[i]
                head_idx, head = peaks[i + 1]
                right_shoulder_idx, right_shoulder = peaks[i + 2]
                
                # Check if head is higher than both shoulders
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):  # Shoulders at similar level
                    
                    # Calculate neckline (support level between valleys)
                    left_valley = min(highs[left_shoulder_idx:head_idx])
                    right_valley = min(highs[head_idx:right_shoulder_idx])
                    neckline = (left_valley + right_valley) / 2
                    
                    current_price = closes[-1]
                    pattern_height = head - neckline
                    breakout_target = neckline - pattern_height
                    
                    confidence = 0.8 if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05 else 0.6
                    
                    return PatternDetection(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        confidence=confidence,
                        start_date=datetime.now().isoformat(),
                        end_date=datetime.now().isoformat(),
                        support_level=neckline,
                        resistance_level=head,
                        breakout_target=breakout_target,
                        risk_reward_ratio=(current_price - breakout_target) / (head - current_price),
                        description=f"Head and Shoulders pattern with neckline at {neckline:.2f}",
                        signal="sell" if current_price > neckline else "neutral",
                        timeframe="1d"
                    )
            
        except Exception as e:
            print(f"❌ Error in head and shoulders detection: {e}")
        
        return None
    
    def _detect_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> Optional[PatternDetection]:
        """🔍 Detect Ascending Triangle Pattern"""
        if len(highs) < 15:
            return None
        
        try:
            # Check for horizontal resistance and ascending support
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Find resistance level (similar highs)
            max_high = max(recent_highs)
            resistance_touches = sum(1 for h in recent_highs if abs(h - max_high) / max_high < 0.02)
            
            if resistance_touches < 2:
                return None
            
            # Check for ascending lows (support trend)
            first_half_lows = recent_lows[:7]
            second_half_lows = recent_lows[8:]
            
            if len(first_half_lows) > 0 and len(second_half_lows) > 0:
                avg_early_lows = np.mean(first_half_lows)
                avg_recent_lows = np.mean(second_half_lows)
                
                if avg_recent_lows > avg_early_lows:  # Ascending support
                    pattern_height = max_high - avg_early_lows
                    breakout_target = max_high + pattern_height
                    
                    return PatternDetection(
                        pattern_type=PatternType.TRIANGLE_ASCENDING,
                        confidence=0.6,
                        start_date=datetime.now().isoformat(),
                        end_date=datetime.now().isoformat(),
                        support_level=avg_recent_lows,
                        resistance_level=max_high,
                        breakout_target=breakout_target,
                        risk_reward_ratio=2.0,
                        description=f"Ascending Triangle with resistance at {max_high:.2f}",
                        signal="buy",
                        timeframe="1d"
                    )
            
        except Exception as e:
            print(f"❌ Error in ascending triangle detection: {e}")
        
        return None
    
    def _detect_descending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> Optional[PatternDetection]:
        """🔍 Detect Descending Triangle Pattern"""
        if len(lows) < 15:
            return None
        
        try:
            # Check for horizontal support and descending resistance
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Find support level (similar lows)
            min_low = min(recent_lows)
            support_touches = sum(1 for l in recent_lows if abs(l - min_low) / min_low < 0.02)
            
            if support_touches < 2:
                return None
            
            # Check for descending highs (resistance trend)
            first_half_highs = recent_highs[:7]
            second_half_highs = recent_highs[8:]
            
            if len(first_half_highs) > 0 and len(second_half_highs) > 0:
                avg_early_highs = np.mean(first_half_highs)
                avg_recent_highs = np.mean(second_half_highs)
                
                if avg_recent_highs < avg_early_highs:  # Descending resistance
                    pattern_height = avg_early_highs - min_low
                    breakout_target = min_low - pattern_height
                    
                    return PatternDetection(
                        pattern_type=PatternType.TRIANGLE_DESCENDING,
                        confidence=0.6,
                        start_date=datetime.now().isoformat(),
                        end_date=datetime.now().isoformat(),
                        support_level=min_low,
                        resistance_level=avg_recent_highs,
                        breakout_target=breakout_target,
                        risk_reward_ratio=2.0,
                        description=f"Descending Triangle with support at {min_low:.2f}",
                        signal="sell",
                        timeframe="1d"
                    )
            
        except Exception as e:
            print(f"❌ Error in descending triangle detection: {e}")
        
        return None
    
    def analyze_trend(self, coin_id: str, timeframe: str) -> TrendAnalysis:
        """📈 Comprehensive Trend Analysis"""
        try:
            df = self.get_price_data(coin_id, timeframe, days=50)
            if df.empty:
                return TrendAnalysis(
                    timeframe=timeframe,
                    direction=TrendDirection.UNDEFINED,
                    strength=0,
                    duration_days=0,
                    start_price=0,
                    current_price=0,
                    change_percent=0,
                    confidence=0,
                    support_levels=[],
                    resistance_levels=[],
                    trend_line_slope=0
                )
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            current_price = closes[-1]
            start_price = closes[0]
            
            # Calculate overall trend
            change_percent = ((current_price - start_price) / start_price) * 100
            
            # Determine trend direction
            if change_percent > 10:
                direction = TrendDirection.BULLISH
            elif change_percent < -10:
                direction = TrendDirection.BEARISH
            elif abs(change_percent) < 5:
                direction = TrendDirection.SIDEWAYS
            else:
                direction = TrendDirection.UNDEFINED
            
            # Calculate trend strength using linear regression
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            trend_line_slope = slope
            
            # Calculate R-squared for trend strength
            predicted = slope * x + intercept
            ss_res = np.sum((closes - predicted) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            strength = min(abs(r_squared), 1.0)
            
            # Find support and resistance levels
            support_levels = self._find_support_levels(lows)
            resistance_levels = self._find_resistance_levels(highs)
            
            return TrendAnalysis(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                duration_days=len(closes),
                start_price=start_price,
                current_price=current_price,
                change_percent=change_percent,
                confidence=min(strength + 0.2, 1.0),
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_line_slope=trend_line_slope
            )
            
        except Exception as e:
            print(f"❌ Error analyzing trend for {coin_id}: {e}")
            return TrendAnalysis(
                timeframe=timeframe,
                direction=TrendDirection.UNDEFINED,
                strength=0,
                duration_days=0,
                start_price=0,
                current_price=0,
                change_percent=0,
                confidence=0,
                support_levels=[],
                resistance_levels=[],
                trend_line_slope=0
            )
    
    def _find_support_levels(self, lows: np.ndarray, tolerance: float = 0.02) -> List[float]:
        """📏 Find Support Levels"""
        if len(lows) < 10:
            return []
        
        # Find local minima
        support_candidates = []
        for i in range(2, len(lows) - 2):
            if lows[i] <= lows[i-1] and lows[i] <= lows[i+1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+2]:
                support_candidates.append(lows[i])
        
        # Group similar levels
        support_levels = []
        for candidate in support_candidates:
            # Check if this level is already represented
            is_duplicate = any(abs(candidate - level) / level < tolerance for level in support_levels)
            if not is_duplicate:
                support_levels.append(candidate)
        
        return sorted(support_levels)[-5:]  # Return top 5 support levels
    
    def _find_resistance_levels(self, highs: np.ndarray, tolerance: float = 0.02) -> List[float]:
        """📏 Find Resistance Levels"""
        if len(highs) < 10:
            return []
        
        # Find local maxima
        resistance_candidates = []
        for i in range(2, len(highs) - 2):
            if highs[i] >= highs[i-1] and highs[i] >= highs[i+1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+2]:
                resistance_candidates.append(highs[i])
        
        # Group similar levels
        resistance_levels = []
        for candidate in resistance_candidates:
            # Check if this level is already represented
            is_duplicate = any(abs(candidate - level) / level < tolerance for level in resistance_levels)
            if not is_duplicate:
                resistance_levels.append(candidate)
        
        return sorted(resistance_levels, reverse=True)[:5]  # Return top 5 resistance levels
    
    def generate_comprehensive_signal(self, coin_id: str, coin_symbol: str, timeframe: str = '1d') -> TechnicalSignal:
        """🎯 Generate Comprehensive Technical Signal"""
        try:
            # Analyze all components
            indicators = self.analyze_indicators(coin_id, timeframe)
            patterns = self.detect_patterns(coin_id, timeframe)
            trend_analysis = self.analyze_trend(coin_id, timeframe)
            
            # Calculate overall signal strength
            buy_signals = sum(1 for ind in indicators if ind.signal == 'buy')
            sell_signals = sum(1 for ind in indicators if ind.signal == 'sell')
            neutral_signals = sum(1 for ind in indicators if ind.signal == 'neutral')
            
            total_signals = len(indicators)
            if total_signals == 0:
                signal_strength = 0
                overall_signal = 'neutral'
            else:
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals
                
                if buy_ratio >= 0.6:
                    overall_signal = 'strong_buy' if buy_ratio >= 0.8 else 'buy'
                    signal_strength = buy_ratio
                elif sell_ratio >= 0.6:
                    overall_signal = 'strong_sell' if sell_ratio >= 0.8 else 'sell'
                    signal_strength = sell_ratio
                else:
                    overall_signal = 'neutral'
                    signal_strength = max(buy_ratio, sell_ratio)
            
            # Adjust signal based on trend
            if trend_analysis.direction == TrendDirection.BULLISH and trend_analysis.strength > 0.6:
                if overall_signal in ['sell', 'strong_sell']:
                    overall_signal = 'neutral'  # Trend overrides weak counter-signals
                elif overall_signal == 'buy':
                    overall_signal = 'strong_buy'
            elif trend_analysis.direction == TrendDirection.BEARISH and trend_analysis.strength > 0.6:
                if overall_signal in ['buy', 'strong_buy']:
                    overall_signal = 'neutral'  # Trend overrides weak counter-signals
                elif overall_signal == 'sell':
                    overall_signal = 'strong_sell'
            
            # Calculate confidence
            indicator_confidence = np.mean([ind.confidence for ind in indicators]) if indicators else 0.5
            trend_confidence = trend_analysis.confidence
            pattern_confidence = np.mean([p.confidence for p in patterns]) if patterns else 0.5
            
            overall_confidence = (indicator_confidence * 0.5 + trend_confidence * 0.3 + pattern_confidence * 0.2)
            
            # Calculate price targets
            current_price = trend_analysis.current_price
            price_targets = self._calculate_price_targets(current_price, trend_analysis, patterns)
            
            # Risk assessment
            risk_assessment = self._assess_risk(trend_analysis, indicators, patterns)
            
            # Recommended action
            recommended_action = self._get_recommended_action(overall_signal, signal_strength, overall_confidence)
            
            # Get support/resistance levels
            support_resistance = self._get_support_resistance_levels(coin_id, timeframe)
            
            signal = TechnicalSignal(
                timestamp=datetime.now().isoformat(),
                coin_id=coin_id,
                coin_symbol=coin_symbol,
                timeframe=timeframe,
                overall_signal=overall_signal,
                signal_strength=signal_strength,
                confidence=overall_confidence,
                indicators=indicators,
                patterns=patterns,
                trend_analysis=trend_analysis,
                support_resistance=support_resistance,
                price_targets=price_targets,
                risk_assessment=risk_assessment,
                recommended_action=recommended_action
            )
            
            # Store in history
            if coin_id not in self.analysis_results:
                self.analysis_results[coin_id] = []
            self.analysis_results[coin_id].append(signal)
            
            # Keep only recent results
            self.analysis_results[coin_id] = self.analysis_results[coin_id][-50:]
            
            return signal
            
        except Exception as e:
            print(f"❌ Error generating technical signal for {coin_id}: {e}")
            # Return empty signal on error
            return TechnicalSignal(
                timestamp=datetime.now().isoformat(),
                coin_id=coin_id,
                coin_symbol=coin_symbol,
                timeframe=timeframe,
                overall_signal='neutral',
                signal_strength=0,
                confidence=0,
                indicators=[],
                patterns=[],
                trend_analysis=TrendAnalysis(
                    timeframe=timeframe,
                    direction=TrendDirection.UNDEFINED,
                    strength=0,
                    duration_days=0,
                    start_price=0,
                    current_price=0,
                    change_percent=0,
                    confidence=0,
                    support_levels=[],
                    resistance_levels=[],
                    trend_line_slope=0
                ),
                support_resistance=[],
                price_targets={},
                risk_assessment='unknown',
                recommended_action='hold'
            )
    
    def _calculate_price_targets(self, current_price: float, trend_analysis: TrendAnalysis, patterns: List[PatternDetection]) -> Dict[str, float]:
        """🎯 Calculate Price Targets"""
        targets = {}
        
        # Trend-based targets
        if trend_analysis.direction == TrendDirection.BULLISH:
            targets['short_term_bull'] = current_price * 1.05
            targets['medium_term_bull'] = current_price * 1.15
            targets['long_term_bull'] = current_price * 1.30
        elif trend_analysis.direction == TrendDirection.BEARISH:
            targets['short_term_bear'] = current_price * 0.95
            targets['medium_term_bear'] = current_price * 0.85
            targets['long_term_bear'] = current_price * 0.70
        
        # Pattern-based targets
        for pattern in patterns:
            if pattern.breakout_target > 0:
                direction = "bull" if pattern.signal == "buy" else "bear"
                targets[f'pattern_{pattern.pattern_type.value}_{direction}'] = pattern.breakout_target
        
        # Support/Resistance targets
        if trend_analysis.resistance_levels:
            targets['next_resistance'] = min([r for r in trend_analysis.resistance_levels if r > current_price], default=current_price * 1.1)
        
        if trend_analysis.support_levels:
            targets['next_support'] = max([s for s in trend_analysis.support_levels if s < current_price], default=current_price * 0.9)
        
        return targets
    
    def _assess_risk(self, trend_analysis: TrendAnalysis, indicators: List[TechnicalIndicator], patterns: List[PatternDetection]) -> str:
        """⚠️ Assess Risk Level"""
        risk_factors = 0
        
        # Trend risk
        if trend_analysis.direction == TrendDirection.UNDEFINED:
            risk_factors += 1
        
        if trend_analysis.strength < 0.3:
            risk_factors += 1
        
        # Indicator conflicts
        buy_signals = sum(1 for ind in indicators if ind.signal == 'buy')
        sell_signals = sum(1 for ind in indicators if ind.signal == 'sell')
        
        if abs(buy_signals - sell_signals) <= 1 and len(indicators) > 3:
            risk_factors += 1  # Conflicting signals
        
        # Pattern risk
        bearish_patterns = sum(1 for p in patterns if p.signal == 'sell')
        if bearish_patterns > 0:
            risk_factors += 1
        
        # Return risk assessment
        if risk_factors >= 3:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        elif risk_factors >= 1:
            return 'low'
        else:
            return 'very_low'
    
    def _get_recommended_action(self, overall_signal: str, signal_strength: float, confidence: float) -> str:
        """💡 Get Recommended Action"""
        if confidence < 0.4:
            return 'wait_for_confirmation'
        
        if overall_signal == 'strong_buy' and signal_strength > 0.7:
            return 'aggressive_buy'
        elif overall_signal == 'buy' and signal_strength > 0.5:
            return 'moderate_buy'
        elif overall_signal == 'strong_sell' and signal_strength > 0.7:
            return 'aggressive_sell'
        elif overall_signal == 'sell' and signal_strength > 0.5:
            return 'moderate_sell'
        else:
            return 'hold'
    
    def _get_support_resistance_levels(self, coin_id: str, timeframe: str) -> List[SupportResistance]:
        """📏 Get Support/Resistance Levels"""
        levels = []
        
        try:
            df = self.get_price_data(coin_id, timeframe, days=30)
            if df.empty:
                return levels
            
            highs = df['high'].values
            lows = df['low'].values
            current_price = df['close'].iloc[-1]
            
            # Find support levels
            support_prices = self._find_support_levels(lows)
            for support in support_prices:
                if support < current_price:
                    levels.append(SupportResistance(
                        level=support,
                        level_type='support',
                        strength=0.7,
                        touches=2,
                        last_touch=datetime.now().isoformat(),
                        timeframe=timeframe,
                        confidence=0.6
                    ))
            
            # Find resistance levels
            resistance_prices = self._find_resistance_levels(highs)
            for resistance in resistance_prices:
                if resistance > current_price:
                    levels.append(SupportResistance(
                        level=resistance,
                        level_type='resistance',
                        strength=0.7,
                        touches=2,
                        last_touch=datetime.now().isoformat(),
                        timeframe=timeframe,
                        confidence=0.6
                    ))
            
        except Exception as e:
            print(f"❌ Error getting support/resistance levels: {e}")
        
        return levels
    
    def get_analysis_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """📊 Get Analysis Summary"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        total_analyses = 0
        strong_signals = 0
        bullish_signals = 0
        bearish_signals = 0
        pattern_detections = 0
        
        for coin_results in self.analysis_results.values():
            for signal in coin_results:
                signal_time = datetime.fromisoformat(signal.timestamp.replace('Z', ''))
                if signal_time >= cutoff_date:
                    total_analyses += 1
                    
                    if signal.overall_signal in ['strong_buy', 'strong_sell']:
                        strong_signals += 1
                    
                    if signal.overall_signal in ['buy', 'strong_buy']:
                        bullish_signals += 1
                    elif signal.overall_signal in ['sell', 'strong_sell']:
                        bearish_signals += 1
                    
                    pattern_detections += len(signal.patterns)
        
        return {
            'period_days': days_back,
            'total_analyses': total_analyses,
            'strong_signals': strong_signals,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'neutral_signals': total_analyses - bullish_signals - bearish_signals,
            'pattern_detections': pattern_detections,
            'analysis_active': self.analysis_active,
            'last_update': datetime.now().isoformat()
        }
    
    def start_continuous_analysis(self, coin_list: List[str] = None):
        """🔄 Start Continuous Analysis"""
        if self.analysis_active:
            print("⚠️ Analysis already running")
            return
        
        if not coin_list:
            coin_list = ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana']
        
        self.analysis_active = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(coin_list,),
            daemon=True
        )
        self.analysis_thread.start()
        print("🔄 Continuous technical analysis started")
    
    def stop_continuous_analysis(self):
        """⏹️ Stop Continuous Analysis"""
        self.analysis_active = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        print("⏹️ Continuous technical analysis stopped")
    
    def _analysis_worker(self, coin_list: List[str]):
        """🔄 Analysis Worker Thread"""
        while self.analysis_active:
            try:
                for coin_id in coin_list:
                    if not self.analysis_active:
                        break
                    
                    coin_symbol = self._get_coin_symbol(coin_id)
                    signal = self.generate_comprehensive_signal(coin_id, coin_symbol)
                    
                    # Print significant signals
                    if signal.overall_signal in ['strong_buy', 'strong_sell']:
                        print(f"🎯 {signal.overall_signal.upper()}: {coin_symbol} - Confidence: {signal.confidence:.1%}")
                
                # Save data after each cycle
                self.save_data()
                
                # Wait before next analysis cycle
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                print(f"❌ Error in analysis worker: {e}")
                time.sleep(60)
    
    def _get_coin_symbol(self, coin_id: str) -> str:
        """🔄 Convert coin ID to symbol"""
        symbol_mapping = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'binancecoin': 'BNB',
            'ripple': 'XRP',
            'solana': 'SOL',
            'cardano': 'ADA',
            'dogecoin': 'DOGE'
        }
        return symbol_mapping.get(coin_id, coin_id.upper())
    
    def load_data(self):
        """📁 Load analysis data"""
        try:
            import os
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load analysis results
                for coin_id, results in data.get('analysis_results', {}).items():
                    self.analysis_results[coin_id] = []
                    for result_data in results:
                        # Reconstruct TechnicalSignal objects (simplified)
                        signal = TechnicalSignal(**result_data)
                        self.analysis_results[coin_id].append(signal)
                        
        except Exception as e:
            print(f"❌ Error loading analysis data: {e}")
    
    def save_data(self):
        """💾 Save analysis data"""
        try:
            # Convert data to serializable format
            data = {
                'analysis_results': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Save recent analysis results
            for coin_id, results in self.analysis_results.items():
                recent_results = results[-10:]  # Keep last 10
                data['analysis_results'][coin_id] = [asdict(signal) for signal in recent_results]
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving analysis data: {e}")

# Test der Advanced Technical Analysis
def main():
    """📊 Test Advanced Technical Analysis"""
    analyzer = AdvancedTechnicalAnalysis()
    
    print("📊 Advanced Technical Analysis - Demo")
    print("=" * 50)
    
    # Test single coin analysis
    print("\n🔍 Analyzing Bitcoin...")
    signal = analyzer.generate_comprehensive_signal('bitcoin', 'BTC')
    
    print(f"   Overall Signal: {signal.overall_signal.upper()}")
    print(f"   Signal Strength: {signal.signal_strength:.1%}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Trend: {signal.trend_analysis.direction.value} ({signal.trend_analysis.strength:.1%})")
    print(f"   Risk Assessment: {signal.risk_assessment}")
    print(f"   Recommended Action: {signal.recommended_action}")
    
    print(f"\n📊 Technical Indicators ({len(signal.indicators)}):")
    for ind in signal.indicators[:5]:  # Show first 5
        print(f"   • {ind.name}: {ind.signal} ({ind.value:.2f}) - {ind.interpretation}")
    
    if signal.patterns:
        print(f"\n🔍 Pattern Detections ({len(signal.patterns)}):")
        for pattern in signal.patterns:
            print(f"   • {pattern.pattern_type.value}: {pattern.confidence:.1%} confidence - {pattern.description}")
    
    if signal.price_targets:
        print(f"\n🎯 Price Targets:")
        for target_name, target_price in signal.price_targets.items():
            print(f"   • {target_name}: ${target_price:.2f}")
    
    # Test multi-coin analysis
    print("\n🔄 Running multi-coin analysis...")
    coins = ['bitcoin', 'ethereum', 'binancecoin']
    
    for coin_id in coins:
        coin_symbol = analyzer._get_coin_symbol(coin_id)
        signal = analyzer.generate_comprehensive_signal(coin_id, coin_symbol)
        print(f"   {coin_symbol}: {signal.overall_signal} (confidence: {signal.confidence:.1%})")
    
    # Get summary
    print("\n📊 Analysis Summary:")
    summary = analyzer.get_analysis_summary()
    print(f"   Total Analyses: {summary['total_analyses']}")
    print(f"   Strong Signals: {summary['strong_signals']}")
    print(f"   Bullish Signals: {summary['bullish_signals']}")
    print(f"   Bearish Signals: {summary['bearish_signals']}")
    print(f"   Pattern Detections: {summary['pattern_detections']}")
    
    # Save data
    analyzer.save_data()

if __name__ == "__main__":
    main()