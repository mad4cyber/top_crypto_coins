#!/usr/bin/env python3
"""
⚙️ Optimierte Trading-Parameter für Live Trading
Autor: mad4cyber
Version: 1.0 - Production Ready

🎯 FEATURES:
- Conservative Live Trading Settings
- Risk Management Profiles
- Asset-spezifische Parameter
- Performance-optimierte Einstellungen
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class RiskProfile(Enum):
    """Risk Management Profile"""
    CONSERVATIVE = "conservative"  # Sehr sicher, kleine Gewinne
    MODERATE = "moderate"         # Ausgewogen
    AGGRESSIVE = "aggressive"     # Höheres Risiko, höhere Gewinne

class TradingMode(Enum):
    """Trading Modi"""
    TESTNET = "testnet"
    LIVE_DEMO = "live_demo"      # Live mit Mini-Beträgen
    LIVE_SMALL = "live_small"    # Live mit kleinen Beträgen  
    LIVE_NORMAL = "live_normal"  # Normales Live Trading

@dataclass
class TradingParameters:
    """🎯 Optimierte Trading-Parameter"""
    
    # === RISK MANAGEMENT ===
    max_risk_per_trade: float = 0.01        # 1% Risiko pro Trade (sehr konservativ)
    max_portfolio_risk: float = 0.05        # 5% Gesamtrisiko
    max_positions: int = 3                  # Maximal 3 gleichzeitige Positionen
    max_daily_trades: int = 10              # Max 10 Trades pro Tag
    
    # === AI-SIGNAL PARAMETER ===
    min_ai_confidence: float = 0.85         # 85% Mindest-AI-Konfidenz (sehr hoch)
    min_price_change_threshold: float = 2.0 # Mindestens 2% erwartete Änderung
    signal_timeout_hours: int = 4           # Signal verfällt nach 4 Stunden
    
    # === STOP-LOSS / TAKE-PROFIT ===
    default_stop_loss_pct: float = 0.03     # 3% Stop-Loss (enger)
    default_take_profit_pct: float = 0.06   # 6% Take-Profit (2:1 Ratio)
    trailing_stop_activation: float = 0.02  # Trailing Stop bei 2% Gewinn
    trailing_stop_distance: float = 0.01    # 1% Trailing Distance
    
    # === POSITION SIZING ===
    base_position_size_usd: float = 50.0    # $50 Basis-Positionsgröße
    max_position_size_usd: float = 200.0    # $200 Maximum pro Position
    confidence_multiplier: float = 1.5      # Max 1.5x bei hoher Konfidenz
    
    # === ASSET-SPEZIFISCHE PARAMETER ===
    asset_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.asset_multipliers is None:
            # Conservative Multipliers für verschiedene Assets
            self.asset_multipliers = {
                'BTCUSDT': 1.0,    # Bitcoin: Standard
                'ETHUSDT': 0.9,    # Ethereum: Etwas konservativer  
                'BNBUSDT': 0.8,    # BNB: Konservativer
                'ADAUSDT': 0.7,    # Altcoins: Sehr konservativ
                'SOLUSDT': 0.7,    # Altcoins: Sehr konservativ
            }

# === VORDEFINIERTE PARAMETER-SETS ===

CONSERVATIVE_PARAMS = TradingParameters(
    max_risk_per_trade=0.005,        # 0.5% Risiko pro Trade
    min_ai_confidence=0.90,          # 90% Mindest-Konfidenz
    min_price_change_threshold=3.0,  # 3% Mindest-Änderung
    default_stop_loss_pct=0.02,      # 2% Stop-Loss
    default_take_profit_pct=0.04,    # 4% Take-Profit
    base_position_size_usd=25.0,     # $25 Basis-Größe
    max_positions=2                  # Max 2 Positionen
)

MODERATE_PARAMS = TradingParameters(
    max_risk_per_trade=0.01,         # 1% Risiko pro Trade
    min_ai_confidence=0.85,          # 85% Mindest-Konfidenz
    min_price_change_threshold=2.0,  # 2% Mindest-Änderung
    default_stop_loss_pct=0.03,      # 3% Stop-Loss
    default_take_profit_pct=0.06,    # 6% Take-Profit
    base_position_size_usd=50.0,     # $50 Basis-Größe
    max_positions=3                  # Max 3 Positionen
)

AGGRESSIVE_PARAMS = TradingParameters(
    max_risk_per_trade=0.02,         # 2% Risiko pro Trade
    min_ai_confidence=0.80,          # 80% Mindest-Konfidenz
    min_price_change_threshold=1.5,  # 1.5% Mindest-Änderung
    default_stop_loss_pct=0.04,      # 4% Stop-Loss
    default_take_profit_pct=0.08,    # 8% Take-Profit
    base_position_size_usd=100.0,    # $100 Basis-Größe
    max_positions=5                  # Max 5 Positionen
)

# === LIVE TRADING PARAMETER FÜR VERSCHIEDENE MODI ===

LIVE_DEMO_PARAMS = TradingParameters(
    max_risk_per_trade=0.002,        # 0.2% Risiko (sehr klein)
    min_ai_confidence=0.95,          # 95% Konfidenz (sehr hoch)
    base_position_size_usd=10.0,     # $10 für Demo
    max_position_size_usd=25.0,      # $25 Maximum
    max_positions=2,                 # Nur 2 Positionen
    max_daily_trades=5               # Max 5 Trades/Tag
)

LIVE_SMALL_PARAMS = TradingParameters(
    max_risk_per_trade=0.005,        # 0.5% Risiko
    min_ai_confidence=0.90,          # 90% Konfidenz
    base_position_size_usd=25.0,     # $25 Basis
    max_position_size_usd=100.0,     # $100 Maximum
    max_positions=3,                 # 3 Positionen
    max_daily_trades=8               # Max 8 Trades/Tag
)

LIVE_NORMAL_PARAMS = MODERATE_PARAMS  # Standard-Parameter

def get_trading_parameters(mode: TradingMode, risk_profile: RiskProfile) -> TradingParameters:
    """🎯 Optimale Parameter für Modus und Risk Profile"""
    
    if mode == TradingMode.TESTNET:
        # Testnet: Kann aggressiver sein
        return AGGRESSIVE_PARAMS
    
    elif mode == TradingMode.LIVE_DEMO:
        # Live Demo: Sehr konservativ
        return LIVE_DEMO_PARAMS
    
    elif mode == TradingMode.LIVE_SMALL:
        # Live Small: Konservativ
        return LIVE_SMALL_PARAMS
    
    elif mode == TradingMode.LIVE_NORMAL:
        # Live Normal: Nach Risk Profile
        if risk_profile == RiskProfile.CONSERVATIVE:
            return CONSERVATIVE_PARAMS
        elif risk_profile == RiskProfile.MODERATE:
            return MODERATE_PARAMS
        else:  # AGGRESSIVE
            return AGGRESSIVE_PARAMS
    
    return MODERATE_PARAMS  # Fallback

def calculate_optimal_position_size(
    balance_usd: float, 
    ai_confidence: float, 
    expected_change_pct: float,
    asset_symbol: str,
    params: TradingParameters
) -> float:
    """💰 Optimale Positionsgröße berechnen"""
    
    # Basis-Positionsgröße
    base_size = params.base_position_size_usd
    
    # Asset-spezifischer Multiplier
    asset_multiplier = params.asset_multipliers.get(asset_symbol, 0.5)
    
    # Konfidenz-Anpassung
    confidence_factor = min(ai_confidence / params.min_ai_confidence, params.confidence_multiplier)
    
    # Expected Change Anpassung
    change_factor = min(abs(expected_change_pct) / params.min_price_change_threshold, 1.5)
    
    # Portfolio-basierte Anpassung
    portfolio_factor = min(balance_usd / 1000, 2.0)  # Skaliert mit Portfolio-Größe
    
    # Finale Positionsgröße
    position_size = (
        base_size * 
        asset_multiplier * 
        confidence_factor * 
        change_factor * 
        portfolio_factor
    )
    
    # Limitierungen anwenden
    position_size = min(position_size, params.max_position_size_usd)
    position_size = min(position_size, balance_usd * params.max_risk_per_trade * 20)  # Max Risk Limit
    
    return position_size

def get_asset_specific_params(symbol: str, base_params: TradingParameters) -> TradingParameters:
    """🎯 Asset-spezifische Parameter-Anpassungen"""
    
    # Kopiere base params
    params = TradingParameters(
        max_risk_per_trade=base_params.max_risk_per_trade,
        max_portfolio_risk=base_params.max_portfolio_risk,
        max_positions=base_params.max_positions,
        max_daily_trades=base_params.max_daily_trades,
        min_ai_confidence=base_params.min_ai_confidence,
        min_price_change_threshold=base_params.min_price_change_threshold,
        signal_timeout_hours=base_params.signal_timeout_hours,
        default_stop_loss_pct=base_params.default_stop_loss_pct,
        default_take_profit_pct=base_params.default_take_profit_pct,
        trailing_stop_activation=base_params.trailing_stop_activation,
        trailing_stop_distance=base_params.trailing_stop_distance,
        base_position_size_usd=base_params.base_position_size_usd,
        max_position_size_usd=base_params.max_position_size_usd,
        confidence_multiplier=base_params.confidence_multiplier,
        asset_multipliers=base_params.asset_multipliers.copy()
    )
    
    # Asset-spezifische Anpassungen
    if symbol == 'BTCUSDT':
        # Bitcoin: Standardparameter
        pass
    
    elif symbol == 'ETHUSDT':
        # Ethereum: Etwas enger Stop-Loss
        params.default_stop_loss_pct *= 0.9
        
    elif symbol in ['ADAUSDT', 'SOLUSDT']:
        # Altcoins: Konservativer
        params.min_ai_confidence += 0.02  # +2% Konfidenz
        params.default_stop_loss_pct *= 0.8  # Engerer Stop-Loss
        params.max_risk_per_trade *= 0.8  # Weniger Risiko
    
    elif symbol == 'BNBUSDT':
        # BNB: Etwas konservativer
        params.min_ai_confidence += 0.01  # +1% Konfidenz
    
    return params

# === LIVE TRADING SAFETY CHECKS ===

def validate_live_trading_safety(
    mode: TradingMode,
    balance_usd: float,
    params: TradingParameters
) -> tuple[bool, List[str]]:
    """🛡️ Sicherheitschecks für Live Trading"""
    
    warnings = []
    is_safe = True
    
    # Balance Checks
    if mode in [TradingMode.LIVE_DEMO, TradingMode.LIVE_SMALL, TradingMode.LIVE_NORMAL]:
        if balance_usd < 100:
            warnings.append("⚠️ Balance unter $100 - sehr riskant für Live Trading")
            is_safe = False
        
        if params.max_risk_per_trade > 0.02:
            warnings.append("⚠️ Risiko pro Trade über 2% - reduziere für Live Trading")
            is_safe = False
        
        if params.min_ai_confidence < 0.85:
            warnings.append("⚠️ AI-Konfidenz unter 85% - erhöhe für Live Trading")
            is_safe = False
        
        if params.max_positions > 5:
            warnings.append("⚠️ Zu viele gleichzeitige Positionen für Live Trading")
            is_safe = False
    
    # Spezielle Live Demo Checks
    if mode == TradingMode.LIVE_DEMO:
        if params.base_position_size_usd > 25:
            warnings.append("⚠️ Position zu groß für Live Demo - reduziere auf <$25")
            is_safe = False
    
    return is_safe, warnings

if __name__ == "__main__":
    # Parameter-Beispiele anzeigen
    print("🎯 Trading Parameter Examples:")
    print("\n1. Conservative Live Demo:")
    demo_params = get_trading_parameters(TradingMode.LIVE_DEMO, RiskProfile.CONSERVATIVE)
    print(f"   Risk per trade: {demo_params.max_risk_per_trade:.1%}")
    print(f"   AI confidence: {demo_params.min_ai_confidence:.1%}")
    print(f"   Position size: ${demo_params.base_position_size_usd}")
    
    print("\n2. Moderate Live Trading:")
    live_params = get_trading_parameters(TradingMode.LIVE_NORMAL, RiskProfile.MODERATE)
    print(f"   Risk per trade: {live_params.max_risk_per_trade:.1%}")
    print(f"   AI confidence: {live_params.min_ai_confidence:.1%}")
    print(f"   Position size: ${live_params.base_position_size_usd}")
    
    print("\n3. Position Size Example (1000 USD Balance, 90% AI, 3% expected):")
    position_size = calculate_optimal_position_size(1000, 0.90, 3.0, 'BTCUSDT', live_params)
    print(f"   Optimal position: ${position_size:.2f}")
