#!/usr/bin/env python3
"""
📊 Dynamic Asset Allocation System
Autor: mad4cyber
Version: 1.0 - Dynamic Allocation Edition

🚀 FEATURES:
- Dynamische Portfolio-Anpassung basierend auf Marktbedingungen
- Sektor-Rotation Algorithmen
- Volatilitäts-basierte Position Sizing  
- Korrelations-basierte Diversifikation
- Automatische Risiko-Anpassung
- Multi-Timeframe Momentum Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from ml_portfolio_optimizer import MLPortfolioOptimizer
from performance_tracker import PerformanceTracker
from ai_predictor import CryptoAIPredictor

@dataclass
class AllocationSignal:
    """📈 Allocation Signal für Dynamic Rebalancing"""
    timestamp: str
    signal_type: str  # 'momentum', 'mean_reversion', 'volatility', 'correlation'
    strength: float  # 0-1
    direction: str   # 'increase', 'decrease', 'neutral'
    affected_assets: List[str]
    reasoning: str
    confidence: float
    timeframe: str  # '1h', '4h', '1d', '1w'

@dataclass  
class MarketCondition:
    """🌍 Market Condition Analysis"""
    timestamp: str
    overall_trend: str  # 'bullish', 'bearish', 'neutral'
    volatility_regime: str  # 'low', 'medium', 'high'
    correlation_level: float  # Average correlation between assets
    momentum_strength: float  # Overall momentum
    fear_greed_level: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    recommended_allocation_style: str  # 'defensive', 'balanced', 'aggressive'

class DynamicAllocator:
    """📊 Dynamic Asset Allocation System"""
    
    def __init__(self, data_file: str = "dynamic_allocation_data.json"):
        self.data_file = data_file
        self.ml_optimizer = MLPortfolioOptimizer()
        self.performance_tracker = PerformanceTracker()
        self.ai_predictor = CryptoAIPredictor()
        
        # Historical data
        self.allocation_signals: List[AllocationSignal] = []
        self.market_conditions: List[MarketCondition] = []
        self.allocation_history: List[Dict] = []
        
        # Configuration
        self.momentum_lookback_days = 7
        self.volatility_lookback_days = 14
        self.correlation_lookback_days = 30
        self.rebalancing_threshold = 0.05  # 5%
        
        # Load existing data
        self.load_allocation_data()
    
    def load_allocation_data(self):
        """📁 Load allocation history data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load allocation signals
                for signal_data in data.get('allocation_signals', []):
                    signal = AllocationSignal(**signal_data)
                    self.allocation_signals.append(signal)
                
                # Load market conditions  
                for condition_data in data.get('market_conditions', []):
                    condition = MarketCondition(**condition_data)
                    self.market_conditions.append(condition)
                
                # Load allocation history
                self.allocation_history = data.get('allocation_history', [])
                    
            except Exception as e:
                print(f"❌ Error loading allocation data: {e}")
    
    def save_allocation_data(self):
        """💾 Save allocation data"""
        try:
            data = {
                'allocation_signals': [asdict(signal) for signal in self.allocation_signals],
                'market_conditions': [asdict(condition) for condition in self.market_conditions],
                'allocation_history': self.allocation_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving allocation data: {e}")
    
    def analyze_market_conditions(self, coin_ids: List[str]) -> MarketCondition:
        """🌍 Analyze current market conditions"""
        print("🌍 Analyzing market conditions for dynamic allocation...")
        
        try:
            # Get AI predictions for trend analysis
            predictions = {}
            price_changes = []
            volatilities = []
            confidences = []
            
            for coin_id in coin_ids:
                try:
                    pred = self.ai_predictor.predict_future_prices(coin_id)
                    if 'error' not in pred:
                        predictions[coin_id] = pred
                        price_changes.append(pred.get('price_change_pct', 0))
                        volatilities.append(abs(pred.get('price_change_pct', 0)))
                        confidences.append(pred.get('confidence', 0))
                except Exception as e:
                    continue
            
            if not predictions:
                # Default neutral condition
                return MarketCondition(
                    timestamp=datetime.now().isoformat(),
                    overall_trend='neutral',
                    volatility_regime='medium',
                    correlation_level=0.5,
                    momentum_strength=0.5,
                    fear_greed_level='neutral',
                    recommended_allocation_style='balanced'
                )
            
            # Overall trend analysis
            avg_change = np.mean(price_changes)
            if avg_change > 3:
                overall_trend = 'bullish'
            elif avg_change < -3:
                overall_trend = 'bearish'
            else:
                overall_trend = 'neutral'
            
            # Volatility regime
            avg_volatility = np.mean(volatilities)
            if avg_volatility > 15:
                volatility_regime = 'high'
            elif avg_volatility < 5:
                volatility_regime = 'low'  
            else:
                volatility_regime = 'medium'
            
            # Correlation analysis (simplified)
            if len(price_changes) > 1:
                # Calculate correlation proxy from price change dispersion
                change_std = np.std(price_changes)
                correlation_level = max(0, min(1, 1 - (change_std / 20)))  # Lower dispersion = higher correlation
            else:
                correlation_level = 0.5
            
            # Momentum strength
            positive_predictions = sum(1 for change in price_changes if change > 0)
            momentum_strength = positive_predictions / len(price_changes) if price_changes else 0.5
            
            # Fear & Greed estimation
            avg_confidence = np.mean(confidences) if confidences else 0.5
            if avg_confidence > 0.8 and avg_change > 5:
                fear_greed_level = 'extreme_greed'
            elif avg_confidence > 0.6 and avg_change > 0:
                fear_greed_level = 'greed'
            elif avg_confidence < 0.3 or avg_change < -10:
                fear_greed_level = 'extreme_fear'
            elif avg_change < -3:
                fear_greed_level = 'fear'
            else:
                fear_greed_level = 'neutral'
            
            # Recommended allocation style
            if volatility_regime == 'high' or fear_greed_level in ['extreme_fear', 'fear']:
                allocation_style = 'defensive'
            elif overall_trend == 'bullish' and volatility_regime == 'low':
                allocation_style = 'aggressive'
            else:
                allocation_style = 'balanced'
            
            condition = MarketCondition(
                timestamp=datetime.now().isoformat(),
                overall_trend=overall_trend,
                volatility_regime=volatility_regime,
                correlation_level=correlation_level,
                momentum_strength=momentum_strength,
                fear_greed_level=fear_greed_level,
                recommended_allocation_style=allocation_style
            )
            
            # Save to history
            self.market_conditions.append(condition)
            if len(self.market_conditions) > 100:
                self.market_conditions = self.market_conditions[-50:]
            
            print(f"📊 Market condition: {overall_trend} trend, {volatility_regime} volatility, {allocation_style} allocation")
            return condition
            
        except Exception as e:
            print(f"❌ Error analyzing market conditions: {e}")
            # Return neutral condition on error
            return MarketCondition(
                timestamp=datetime.now().isoformat(),
                overall_trend='neutral',
                volatility_regime='medium',
                correlation_level=0.5,
                momentum_strength=0.5,
                fear_greed_level='neutral',
                recommended_allocation_style='balanced'
            )
    
    def generate_allocation_signals(self, coin_ids: List[str], market_condition: MarketCondition) -> List[AllocationSignal]:
        """📈 Generate allocation signals based on market conditions"""
        print("📈 Generating dynamic allocation signals...")
        
        signals = []
        
        try:
            # Get AI predictions for all coins
            predictions = {}
            for coin_id in coin_ids:
                try:
                    pred = self.ai_predictor.predict_future_prices(coin_id)
                    if 'error' not in pred:
                        predictions[coin_id] = pred
                except Exception as e:
                    continue
            
            if not predictions:
                return signals
            
            # Momentum-based signals
            momentum_signals = self.generate_momentum_signals(predictions, market_condition)
            signals.extend(momentum_signals)
            
            # Volatility-based signals
            volatility_signals = self.generate_volatility_signals(predictions, market_condition)
            signals.extend(volatility_signals)
            
            # Correlation-based signals
            correlation_signals = self.generate_correlation_signals(predictions, market_condition)
            signals.extend(correlation_signals)
            
            # Risk-based signals
            risk_signals = self.generate_risk_signals(predictions, market_condition)
            signals.extend(risk_signals)
            
            # Save signals
            self.allocation_signals.extend(signals)
            if len(self.allocation_signals) > 200:
                self.allocation_signals = self.allocation_signals[-100:]
            
            print(f"✅ Generated {len(signals)} allocation signals")
            return signals
            
        except Exception as e:
            print(f"❌ Error generating allocation signals: {e}")
            return []
    
    def generate_momentum_signals(self, predictions: Dict, market_condition: MarketCondition) -> List[AllocationSignal]:
        """📈 Generate momentum-based allocation signals"""
        signals = []
        
        # Sort coins by predicted momentum
        coin_momentum = []
        for coin_id, pred in predictions.items():
            momentum_score = pred.get('price_change_pct', 0) * pred.get('confidence', 0)
            coin_momentum.append((coin_id, momentum_score))
        
        coin_momentum.sort(key=lambda x: x[1], reverse=True)
        
        # Generate signals for top and bottom momentum coins
        if len(coin_momentum) >= 2:
            # Strong momentum signal
            top_coin, top_momentum = coin_momentum[0]
            if top_momentum > 3:  # Strong positive momentum
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='momentum',
                    strength=min(1.0, top_momentum / 10),
                    direction='increase',
                    affected_assets=[top_coin],
                    reasoning=f"Strong momentum detected for {top_coin}: {top_momentum:.2f}%",
                    confidence=predictions[top_coin].get('confidence', 0),
                    timeframe='1d'
                ))
            
            # Weak momentum signal
            bottom_coin, bottom_momentum = coin_momentum[-1]
            if bottom_momentum < -3:  # Strong negative momentum
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='momentum',
                    strength=min(1.0, abs(bottom_momentum) / 10),
                    direction='decrease',
                    affected_assets=[bottom_coin],
                    reasoning=f"Weak momentum detected for {bottom_coin}: {bottom_momentum:.2f}%",
                    confidence=predictions[bottom_coin].get('confidence', 0),
                    timeframe='1d'
                ))
        
        return signals
    
    def generate_volatility_signals(self, predictions: Dict, market_condition: MarketCondition) -> List[AllocationSignal]:
        """📊 Generate volatility-based allocation signals"""
        signals = []
        
        # Calculate volatility proxy from predictions
        volatilities = {}
        for coin_id, pred in predictions.items():
            # Use confidence as inverse volatility proxy
            volatility = (1 - pred.get('confidence', 0.5)) * abs(pred.get('price_change_pct', 0))
            volatilities[coin_id] = volatility
        
        if not volatilities:
            return signals
        
        avg_volatility = np.mean(list(volatilities.values()))
        
        # High volatility regime - reduce allocation to volatile assets
        if market_condition.volatility_regime == 'high':
            high_vol_coins = [coin for coin, vol in volatilities.items() if vol > avg_volatility * 1.5]
            if high_vol_coins:
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='volatility',
                    strength=0.7,
                    direction='decrease',
                    affected_assets=high_vol_coins,
                    reasoning=f"High volatility regime - reducing exposure to volatile assets",
                    confidence=0.8,
                    timeframe='1d'
                ))
        
        # Low volatility regime - can increase allocation
        elif market_condition.volatility_regime == 'low':
            stable_coins = [coin for coin, vol in volatilities.items() if vol < avg_volatility * 0.5]
            if stable_coins:
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='volatility',
                    strength=0.6,
                    direction='increase',
                    affected_assets=stable_coins,
                    reasoning=f"Low volatility regime - can increase stable asset allocation",
                    confidence=0.7,
                    timeframe='1d'
                ))
        
        return signals
    
    def generate_correlation_signals(self, predictions: Dict, market_condition: MarketCondition) -> List[AllocationSignal]:
        """🔗 Generate correlation-based allocation signals"""
        signals = []
        
        # When correlation is high, favor diversification
        if market_condition.correlation_level > 0.8:
            # Find assets with different predicted directions
            positive_coins = [coin for coin, pred in predictions.items() if pred.get('price_change_pct', 0) > 0]
            negative_coins = [coin for coin, pred in predictions.items() if pred.get('price_change_pct', 0) < 0]
            
            if positive_coins and negative_coins:
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='correlation',
                    strength=0.6,
                    direction='neutral',
                    affected_assets=positive_coins + negative_coins,
                    reasoning="High correlation detected - favor diversification across directions",
                    confidence=0.7,
                    timeframe='1d'
                ))
        
        return signals
    
    def generate_risk_signals(self, predictions: Dict, market_condition: MarketCondition) -> List[AllocationSignal]:
        """⚠️ Generate risk-based allocation signals"""
        signals = []
        
        # In extreme fear, reduce overall exposure
        if market_condition.fear_greed_level == 'extreme_fear':
            all_coins = list(predictions.keys())
            signals.append(AllocationSignal(
                timestamp=datetime.now().isoformat(),
                signal_type='risk',
                strength=0.9,
                direction='decrease',
                affected_assets=all_coins,
                reasoning="Extreme fear detected - reducing overall market exposure",
                confidence=0.9,
                timeframe='1d'
            ))
        
        # In extreme greed, take some profits
        elif market_condition.fear_greed_level == 'extreme_greed':
            high_gain_coins = [
                coin for coin, pred in predictions.items() 
                if pred.get('price_change_pct', 0) > 5
            ]
            if high_gain_coins:
                signals.append(AllocationSignal(
                    timestamp=datetime.now().isoformat(),
                    signal_type='risk',
                    strength=0.7,
                    direction='decrease',
                    affected_assets=high_gain_coins,
                    reasoning="Extreme greed detected - taking profits on high-gain assets",
                    confidence=0.8,
                    timeframe='1d'
                ))
        
        return signals
    
    def calculate_dynamic_allocation(self, portfolio_name: str, market_condition: MarketCondition, 
                                  signals: List[AllocationSignal]) -> Dict[str, float]:
        """⚖️ Calculate dynamic allocation weights"""
        print("⚖️ Calculating dynamic allocation weights...")
        
        try:
            # Start with ML-optimized weights
            ml_result = self.ml_optimizer.optimize_portfolio(portfolio_name, 'risk_adjusted')
            base_weights = ml_result.recommended_weights.copy()
            
            if not base_weights:
                return {}
            
            # Apply signal-based adjustments
            adjusted_weights = base_weights.copy()
            
            for signal in signals:
                adjustment_factor = signal.strength * signal.confidence
                
                for asset in signal.affected_assets:
                    if asset in adjusted_weights:
                        current_weight = adjusted_weights[asset]
                        
                        if signal.direction == 'increase':
                            # Increase allocation (max 40%)
                            new_weight = min(0.40, current_weight * (1 + adjustment_factor * 0.2))
                        elif signal.direction == 'decrease':
                            # Decrease allocation (min 5%)
                            new_weight = max(0.05, current_weight * (1 - adjustment_factor * 0.3))
                        else:  # neutral
                            new_weight = current_weight
                        
                        adjusted_weights[asset] = new_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            # Apply market condition constraints
            adjusted_weights = self.apply_market_constraints(adjusted_weights, market_condition)
            
            print(f"✅ Dynamic allocation calculated for {len(adjusted_weights)} assets")
            return adjusted_weights
            
        except Exception as e:
            print(f"❌ Error calculating dynamic allocation: {e}")
            return {}
    
    def apply_market_constraints(self, weights: Dict[str, float], 
                                market_condition: MarketCondition) -> Dict[str, float]:
        """📊 Apply market condition constraints to allocation"""
        constrained_weights = weights.copy()
        
        # Defensive allocation in high volatility or extreme fear
        if (market_condition.volatility_regime == 'high' or 
            market_condition.fear_greed_level in ['extreme_fear', 'fear']):
            
            # Reduce maximum position size
            max_position = 0.25  # 25% max in defensive mode
            for asset, weight in constrained_weights.items():
                if weight > max_position:
                    constrained_weights[asset] = max_position
        
        # Aggressive allocation in bull market with low volatility
        elif (market_condition.overall_trend == 'bullish' and 
              market_condition.volatility_regime == 'low'):
            
            # Allow higher concentration in top performers
            max_position = 0.45  # 45% max in aggressive mode
            # This is already handled by the increase signals
        
        # Normalize after constraints
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {k: v / total_weight for k, v in constrained_weights.items()}
        
        return constrained_weights
    
    def execute_dynamic_rebalancing(self, portfolio_name: str) -> Dict:
        """🔄 Execute dynamic portfolio rebalancing"""
        print(f"🔄 Executing dynamic rebalancing for portfolio: {portfolio_name}")
        
        try:
            # Get portfolio coins
            portfolio = self.ml_optimizer.portfolio_manager.portfolios.get(portfolio_name)
            if not portfolio or not portfolio.positions:
                return {'error': 'Portfolio not found or empty'}
            
            coin_ids = [pos.coin_id for pos in portfolio.positions]
            
            # Analyze market conditions
            market_condition = self.analyze_market_conditions(coin_ids)
            
            # Generate allocation signals
            signals = self.generate_allocation_signals(coin_ids, market_condition)
            
            # Calculate dynamic allocation
            target_weights = self.calculate_dynamic_allocation(portfolio_name, market_condition, signals)
            
            if not target_weights:
                return {'error': 'Could not calculate target allocation'}
            
            # Get current weights
            current_weights = self.ml_optimizer.get_current_weights(portfolio_name)
            
            # Check if rebalancing is needed
            needs_rebalancing = False
            weight_changes = {}
            
            for asset, target_weight in target_weights.items():
                current_weight = current_weights.get(asset, 0)
                weight_change = abs(target_weight - current_weight)
                weight_changes[asset] = weight_change
                
                if weight_change > self.rebalancing_threshold:
                    needs_rebalancing = True
            
            # Save to allocation history
            allocation_record = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_name': portfolio_name,
                'market_condition': asdict(market_condition),
                'signals_count': len(signals),
                'target_weights': target_weights,
                'current_weights': current_weights,
                'weight_changes': weight_changes,
                'rebalancing_needed': needs_rebalancing,
                'rebalancing_threshold': self.rebalancing_threshold
            }
            
            self.allocation_history.append(allocation_record)
            if len(self.allocation_history) > 100:
                self.allocation_history = self.allocation_history[-50:]
            
            # Save data
            self.save_allocation_data()
            
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'portfolio_name': portfolio_name,
                'market_condition': asdict(market_condition),
                'allocation_signals': [asdict(signal) for signal in signals],
                'target_allocation': target_weights,
                'current_allocation': current_weights,
                'rebalancing_required': needs_rebalancing,
                'max_weight_change': max(weight_changes.values()) if weight_changes else 0,
                'recommendation': self.get_rebalancing_recommendation(market_condition, signals)
            }
            
            print(f"✅ Dynamic rebalancing analysis completed")
            print(f"   Market condition: {market_condition.overall_trend}")
            print(f"   Allocation style: {market_condition.recommended_allocation_style}")
            print(f"   Rebalancing needed: {needs_rebalancing}")
            
            return result
            
        except Exception as e:
            print(f"❌ Dynamic rebalancing failed: {e}")
            return {'error': str(e)}
    
    def get_rebalancing_recommendation(self, market_condition: MarketCondition, 
                                     signals: List[AllocationSignal]) -> str:
        """💡 Get human-readable rebalancing recommendation"""
        if market_condition.recommended_allocation_style == 'defensive':
            return "Market conditions suggest defensive positioning. Consider reducing risk exposure."
        elif market_condition.recommended_allocation_style == 'aggressive':
            return "Favorable market conditions detected. Consider increasing allocation to high-conviction assets."
        elif len(signals) > 3:
            return "Multiple allocation signals detected. Active rebalancing may be beneficial."
        else:
            return "Market conditions are balanced. Maintain current allocation strategy."
    
    def get_allocation_performance(self, days_back: int = 30) -> Dict:
        """📊 Get dynamic allocation performance metrics"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_allocations = [
            alloc for alloc in self.allocation_history
            if datetime.fromisoformat(alloc['timestamp'].replace('Z', '')) >= cutoff_date
        ]
        
        if not recent_allocations:
            return {'message': 'No recent allocation data'}
        
        # Calculate metrics
        total_rebalances = len(recent_allocations)
        rebalancing_rate = np.mean([alloc['rebalancing_needed'] for alloc in recent_allocations])
        avg_weight_change = np.mean([alloc['max_weight_change'] for alloc in recent_allocations])
        
        # Market condition distribution
        conditions = [alloc['market_condition']['overall_trend'] for alloc in recent_allocations]
        condition_counts = {condition: conditions.count(condition) for condition in set(conditions)}
        
        # Signal analysis
        total_signals = sum(alloc['signals_count'] for alloc in recent_allocations)
        
        return {
            'total_analyses': total_rebalances,
            'rebalancing_rate': rebalancing_rate,
            'avg_weight_change': avg_weight_change,
            'total_signals_generated': total_signals,
            'avg_signals_per_analysis': total_signals / total_rebalances if total_rebalances > 0 else 0,
            'market_condition_distribution': condition_counts,
            'timeframe_days': days_back
        }

# Test der Dynamic Allocation
def main():
    """📊 Test des Dynamic Allocation Systems"""
    allocator = DynamicAllocator()
    
    print("📊 Dynamic Allocator - Demo")
    print("=" * 50)
    
    # Test dynamic rebalancing
    result = allocator.execute_dynamic_rebalancing("Main Portfolio")
    
    if 'error' not in result:
        print(f"\n🔄 Dynamic Rebalancing Results:")
        print(f"Market Condition: {result['market_condition']['overall_trend']}")
        print(f"Allocation Style: {result['market_condition']['recommended_allocation_style']}")
        print(f"Signals Generated: {len(result['allocation_signals'])}")
        print(f"Rebalancing Required: {result['rebalancing_required']}")
        print(f"Max Weight Change: {result['max_weight_change']:.1%}")
        print(f"\nRecommendation: {result['recommendation']}")
        
        if result['target_allocation']:
            print(f"\n⚖️ Target Allocation:")
            for asset, weight in result['target_allocation'].items():
                print(f"  {asset}: {weight:.1%}")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()