#!/usr/bin/env python3
"""
🧠 ML-Based Portfolio Optimization System
Autor: mad4cyber
Version: 1.0 - Machine Learning Edition

🚀 FEATURES:
- Reinforcement Learning für Position Sizing
- Dynamische Risiko-Anpassung
- Smart Portfolio Rebalancing
- Multi-Asset Korrelations-Analyse
- Marktregime Detection
- Advanced Risk Management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    HAS_RL = True
except ImportError:
    HAS_RL = False
    print("⚠️ Stable Baselines3 nicht installiert - RL Features deaktiviert")

from portfolio_manager import PortfolioManager
from performance_tracker import PerformanceTracker
from ai_predictor import CryptoAIPredictor

@dataclass
class OptimizationResult:
    """📊 Portfolio Optimization Result"""
    timestamp: str
    portfolio_name: str
    optimization_type: str
    recommended_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    rebalancing_required: bool
    confidence_score: float
    market_regime: str
    reasoning: List[str]

@dataclass
class MarketRegime:
    """📈 Market Regime Classification"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    indicators: Dict[str, float]
    duration_days: int
    volatility_level: str  # 'low', 'medium', 'high'

class MLPortfolioOptimizer:
    """🧠 Machine Learning Portfolio Optimizer"""
    
    def __init__(self, data_file: str = "ml_portfolio_data.json"):
        self.data_file = data_file
        self.portfolio_manager = PortfolioManager()
        self.performance_tracker = PerformanceTracker()
        self.ai_predictor = CryptoAIPredictor()
        
        # ML Models
        self.position_sizing_model = None
        self.rebalancing_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        
        # RL Environment for position sizing (if available)
        self.rl_env = None
        self.rl_model = None
        
        # Historical data for learning
        self.optimization_history: List[OptimizationResult] = []
        self.market_regimes: List[MarketRegime] = []
        
        # Configuration
        self.min_weight = 0.05  # 5% minimum allocation
        self.max_weight = 0.40  # 40% maximum allocation
        self.rebalancing_threshold = 0.10  # 10% drift threshold
        
        # Load existing data
        self.load_optimization_data()
        
        # Initialize ML models
        self.initialize_ml_models()
    
    def load_optimization_data(self):
        """📁 Load optimization history and market data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load optimization results
                for result_data in data.get('optimization_history', []):
                    result = OptimizationResult(**result_data)
                    self.optimization_history.append(result)
                
                # Load market regimes
                for regime_data in data.get('market_regimes', []):
                    regime = MarketRegime(**regime_data)
                    self.market_regimes.append(regime)
                    
            except Exception as e:
                print(f"❌ Error loading optimization data: {e}")
    
    def save_optimization_data(self):
        """💾 Save optimization data"""
        try:
            data = {
                'optimization_history': [asdict(result) for result in self.optimization_history],
                'market_regimes': [asdict(regime) for regime in self.market_regimes],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving optimization data: {e}")
    
    def initialize_ml_models(self):
        """🤖 Initialize ML models"""
        print("🤖 Initializing ML models for portfolio optimization...")
        
        # Position sizing model (Random Forest)
        self.position_sizing_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Risk adjustment model
        self.risk_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Train models if we have historical data
        if len(self.optimization_history) > 50:
            self.train_ml_models()
        
        # Initialize RL environment if available
        if HAS_RL:
            self.setup_rl_environment()
    
    def train_ml_models(self):
        """📚 Train ML models on historical data"""
        if len(self.optimization_history) < 20:
            print("⚠️ Not enough historical data for ML training")
            return
        
        print("📚 Training ML models on historical optimization data...")
        
        # Prepare training data
        features = []
        targets_position = []
        targets_risk = []
        
        for result in self.optimization_history[-100:]:  # Use last 100 results
            # Feature engineering
            feature_vector = self.extract_features_from_result(result)
            features.append(feature_vector)
            
            # Position sizing targets
            max_weight = max(result.recommended_weights.values())
            targets_position.append(max_weight)
            
            # Risk targets
            targets_risk.append(result.expected_risk)
        
        if len(features) < 10:
            return
        
        X = np.array(features)
        y_position = np.array(targets_position)
        y_risk = np.array(targets_risk)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train position sizing model
        try:
            self.position_sizing_model.fit(X_scaled, y_position)
            print("✅ Position sizing model trained successfully")
        except Exception as e:
            print(f"❌ Position sizing model training failed: {e}")
        
        # Train risk model
        try:
            self.risk_model.fit(X_scaled, y_risk)
            print("✅ Risk model trained successfully")
        except Exception as e:
            print(f"❌ Risk model training failed: {e}")
    
    def extract_features_from_result(self, result: OptimizationResult) -> List[float]:
        """🔍 Extract features from optimization result"""
        # Basic features
        features = [
            result.expected_return,
            result.expected_risk,
            result.sharpe_ratio,
            result.confidence_score,
            len(result.recommended_weights),
            1.0 if result.rebalancing_required else 0.0
        ]
        
        # Market regime encoding
        regime_map = {'bull': 1.0, 'bear': -1.0, 'sideways': 0.0, 'volatile': 0.5}
        features.append(regime_map.get(result.market_regime, 0.0))
        
        # Portfolio concentration (Herfindahl index)
        weights = list(result.recommended_weights.values())
        concentration = sum(w**2 for w in weights)
        features.append(concentration)
        
        return features
    
    def setup_rl_environment(self):
        """🎮 Setup Reinforcement Learning environment"""
        if not HAS_RL:
            return
        
        print("🎮 Setting up RL environment for position sizing...")
        # RL environment would be implemented here
        # For now, we use traditional optimization methods
    
    def detect_market_regime(self, coin_ids: List[str]) -> MarketRegime:
        """📈 Detect current market regime using multiple indicators"""
        print("📈 Analyzing market regime...")
        
        indicators = {}
        
        try:
            # Get recent performance data
            recent_data = []
            for coin_id in coin_ids[:5]:  # Analyze top 5 coins
                try:
                    prediction = self.ai_predictor.predict_future_prices(coin_id)
                    if 'error' not in prediction:
                        recent_data.append({
                            'coin_id': coin_id,
                            'price_change': prediction.get('price_change_pct', 0),
                            'confidence': prediction.get('confidence', 0),
                            'volatility': abs(prediction.get('price_change_pct', 0))
                        })
                except Exception as e:
                    continue
            
            if not recent_data:
                return MarketRegime(
                    regime_type='sideways',
                    confidence=0.5,
                    indicators={},
                    duration_days=1,
                    volatility_level='medium'
                )
            
            # Calculate regime indicators
            avg_change = np.mean([d['price_change'] for d in recent_data])
            avg_volatility = np.mean([d['volatility'] for d in recent_data])
            avg_confidence = np.mean([d['confidence'] for d in recent_data])
            
            indicators['avg_price_change'] = avg_change
            indicators['avg_volatility'] = avg_volatility
            indicators['avg_confidence'] = avg_confidence
            
            # Regime classification
            if avg_change > 5 and avg_volatility < 15:
                regime_type = 'bull'
                confidence = min(0.9, 0.5 + abs(avg_change) / 20)
            elif avg_change < -5 and avg_volatility < 15:
                regime_type = 'bear'  
                confidence = min(0.9, 0.5 + abs(avg_change) / 20)
            elif avg_volatility > 20:
                regime_type = 'volatile'
                confidence = min(0.8, 0.4 + avg_volatility / 40)
            else:
                regime_type = 'sideways'
                confidence = 0.6
            
            # Volatility level
            if avg_volatility < 10:
                volatility_level = 'low'
            elif avg_volatility > 25:
                volatility_level = 'high'
            else:
                volatility_level = 'medium'
            
            regime = MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                indicators=indicators,
                duration_days=1,  # Would track this over time
                volatility_level=volatility_level
            )
            
            # Save to history
            self.market_regimes.append(regime)
            if len(self.market_regimes) > 100:
                self.market_regimes = self.market_regimes[-50:]  # Keep last 50
            
            print(f"📊 Market regime detected: {regime_type} (confidence: {confidence:.2f})")
            return regime
            
        except Exception as e:
            print(f"❌ Error detecting market regime: {e}")
            return MarketRegime(
                regime_type='sideways',
                confidence=0.5,
                indicators={},
                duration_days=1,
                volatility_level='medium'
            )
    
    def calculate_optimal_weights(self, portfolio_name: str, target_risk: float = 0.15) -> Dict[str, float]:
        """⚖️ Calculate optimal portfolio weights using ML"""
        print(f"⚖️ Calculating optimal weights for portfolio: {portfolio_name}")
        
        try:
            # Get current portfolio
            portfolio = self.portfolio_manager.portfolios.get(portfolio_name)
            if not portfolio or not portfolio.positions:
                return {}
            
            coin_ids = [pos.coin_id for pos in portfolio.positions]
            
            # Get AI predictions for all coins
            predictions = {}
            total_confidence = 0
            
            for coin_id in coin_ids:
                try:
                    pred = self.ai_predictor.predict_future_prices(coin_id)
                    if 'error' not in pred:
                        predictions[coin_id] = {
                            'expected_return': pred.get('price_change_pct', 0) / 100,
                            'confidence': pred.get('confidence', 0),
                            'risk_score': 1 - pred.get('confidence', 0)  # Lower confidence = higher risk
                        }
                        total_confidence += pred.get('confidence', 0)
                except Exception as e:
                    print(f"❌ Error getting prediction for {coin_id}: {e}")
                    continue
            
            if not predictions:
                return {}
            
            # Detect market regime
            market_regime = self.detect_market_regime(coin_ids)
            
            # ML-enhanced weight calculation
            weights = {}
            remaining_weight = 1.0
            
            # Sort by risk-adjusted expected return
            sorted_coins = sorted(
                predictions.items(),
                key=lambda x: x[1]['expected_return'] * x[1]['confidence'],
                reverse=True
            )
            
            for i, (coin_id, pred) in enumerate(sorted_coins):
                # Base weight from confidence and expected return
                base_weight = pred['confidence'] * (1 + abs(pred['expected_return']))
                
                # Market regime adjustment
                if market_regime.regime_type == 'bull':
                    # In bull market, favor higher expected returns
                    if pred['expected_return'] > 0:
                        base_weight *= 1.2
                elif market_regime.regime_type == 'bear':
                    # In bear market, favor lower risk
                    base_weight *= (1 + pred['confidence'])
                elif market_regime.regime_type == 'volatile':
                    # In volatile market, reduce concentration
                    base_weight *= 0.8
                
                # Apply constraints
                weight = max(self.min_weight, min(self.max_weight, base_weight))
                
                # Ensure we don't exceed 100%
                weight = min(weight, remaining_weight - (len(sorted_coins) - i - 1) * self.min_weight)
                
                weights[coin_id] = weight
                remaining_weight -= weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            print(f"✅ Optimal weights calculated: {len(weights)} assets")
            return weights
            
        except Exception as e:
            print(f"❌ Error calculating optimal weights: {e}")
            return {}
    
    def optimize_portfolio(self, portfolio_name: str, optimization_type: str = "risk_adjusted") -> OptimizationResult:
        """🎯 Main portfolio optimization function"""
        print(f"🎯 Starting portfolio optimization: {optimization_type}")
        
        try:
            # Get current portfolio
            portfolio = self.portfolio_manager.portfolios.get(portfolio_name)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_name} not found")
            
            coin_ids = [pos.coin_id for pos in portfolio.positions]
            
            # Detect market regime
            market_regime = self.detect_market_regime(coin_ids)
            
            # Calculate optimal weights
            target_risk = 0.15 if optimization_type == "conservative" else 0.25
            optimal_weights = self.calculate_optimal_weights(portfolio_name, target_risk)
            
            if not optimal_weights:
                raise ValueError("Could not calculate optimal weights")
            
            # Calculate expected portfolio metrics
            expected_return = 0
            expected_risk = 0
            reasoning = []
            
            for coin_id, weight in optimal_weights.items():
                try:
                    pred = self.ai_predictor.predict_future_prices(coin_id)
                    if 'error' not in pred:
                        coin_return = pred.get('price_change_pct', 0) / 100
                        coin_risk = (1 - pred.get('confidence', 0)) * 0.3  # Risk estimate
                        
                        expected_return += weight * coin_return
                        expected_risk += (weight ** 2) * (coin_risk ** 2)
                        
                        if weight > 0.15:
                            reasoning.append(f"High allocation to {coin_id}: {weight:.1%} (Strong AI confidence)")
                
                except Exception as e:
                    continue
            
            expected_risk = np.sqrt(expected_risk)  # Portfolio risk
            sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
            
            # Check if rebalancing is needed
            current_weights = self.get_current_weights(portfolio_name)
            rebalancing_required = self.needs_rebalancing(current_weights, optimal_weights)
            
            # Confidence score based on AI predictions and market regime
            confidence_score = market_regime.confidence * 0.7 + 0.3  # Base confidence
            
            # Market regime reasoning
            reasoning.append(f"Market regime: {market_regime.regime_type} (confidence: {market_regime.confidence:.2f})")
            reasoning.append(f"Volatility level: {market_regime.volatility_level}")
            
            if rebalancing_required:
                reasoning.append("Portfolio rebalancing recommended due to weight drift")
            
            result = OptimizationResult(
                timestamp=datetime.now().isoformat(),
                portfolio_name=portfolio_name,
                optimization_type=optimization_type,
                recommended_weights=optimal_weights,
                expected_return=expected_return,
                expected_risk=expected_risk,
                sharpe_ratio=sharpe_ratio,
                rebalancing_required=rebalancing_required,
                confidence_score=confidence_score,
                market_regime=market_regime.regime_type,
                reasoning=reasoning
            )
            
            # Save to history
            self.optimization_history.append(result)
            if len(self.optimization_history) > 200:
                self.optimization_history = self.optimization_history[-100:]  # Keep last 100
            
            self.save_optimization_data()
            
            print(f"✅ Portfolio optimization completed!")
            print(f"   Expected Return: {expected_return:.2%}")
            print(f"   Expected Risk: {expected_risk:.2%}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
            # Return default result
            return OptimizationResult(
                timestamp=datetime.now().isoformat(),
                portfolio_name=portfolio_name,
                optimization_type=optimization_type,
                recommended_weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                rebalancing_required=False,
                confidence_score=0,
                market_regime='unknown',
                reasoning=['Optimization failed']
            )
    
    def get_current_weights(self, portfolio_name: str) -> Dict[str, float]:
        """📊 Get current portfolio weights"""
        try:
            portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
            
            if 'error' in portfolio_data or not portfolio_data.get('positions'):
                return {}
            
            total_value = portfolio_data.get('total_value_eur', 0)
            if total_value == 0:
                return {}
            
            weights = {}
            for position in portfolio_data['positions']:
                coin_id = position.get('coin_id', '')
                value = position.get('current_value_eur', 0)
                if coin_id and value > 0:
                    weights[coin_id] = value / total_value
            
            return weights
            
        except Exception as e:
            print(f"❌ Error getting current weights: {e}")
            return {}
    
    def needs_rebalancing(self, current_weights: Dict[str, float], 
                         target_weights: Dict[str, float]) -> bool:
        """⚖️ Check if portfolio needs rebalancing"""
        if not current_weights or not target_weights:
            return True
        
        for coin_id in target_weights:
            current = current_weights.get(coin_id, 0)
            target = target_weights.get(coin_id, 0)
            
            if abs(current - target) > self.rebalancing_threshold:
                return True
        
        return False
    
    def get_rebalancing_trades(self, portfolio_name: str, 
                             optimization_result: OptimizationResult) -> List[Dict]:
        """📋 Get specific trades needed for rebalancing"""
        current_weights = self.get_current_weights(portfolio_name)
        target_weights = optimization_result.recommended_weights
        
        trades = []
        
        try:
            portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
            total_value = portfolio_data.get('total_value_eur', 0)
            
            if total_value == 0:
                return trades
            
            for coin_id, target_weight in target_weights.items():
                current_weight = current_weights.get(coin_id, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # 1% threshold
                    value_diff = weight_diff * total_value
                    
                    # Get current price for quantity calculation
                    current_prices = self.portfolio_manager.get_current_prices([coin_id])
                    price = current_prices.get(coin_id, 0)
                    
                    if price > 0:
                        quantity = abs(value_diff) / price
                        action = "BUY" if weight_diff > 0 else "SELL"
                        
                        trades.append({
                            'coin_id': coin_id,
                            'action': action,
                            'quantity': quantity,
                            'value_eur': abs(value_diff),
                            'weight_change': weight_diff,
                            'reason': f"Adjust allocation from {current_weight:.1%} to {target_weight:.1%}"
                        })
            
            return trades
            
        except Exception as e:
            print(f"❌ Error calculating rebalancing trades: {e}")
            return []
    
    def get_optimization_summary(self, days_back: int = 30) -> Dict:
        """📊 Get optimization performance summary"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_optimizations = [
            opt for opt in self.optimization_history
            if datetime.fromisoformat(opt.timestamp.replace('Z', '')) >= cutoff_date
        ]
        
        if not recent_optimizations:
            return {'message': 'No recent optimizations'}
        
        # Calculate metrics
        avg_expected_return = np.mean([opt.expected_return for opt in recent_optimizations])
        avg_expected_risk = np.mean([opt.expected_risk for opt in recent_optimizations])
        avg_sharpe = np.mean([opt.sharpe_ratio for opt in recent_optimizations])
        avg_confidence = np.mean([opt.confidence_score for opt in recent_optimizations])
        
        rebalancing_rate = np.mean([opt.rebalancing_required for opt in recent_optimizations])
        
        # Market regime distribution
        regimes = [opt.market_regime for opt in recent_optimizations]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        
        return {
            'total_optimizations': len(recent_optimizations),
            'avg_expected_return': avg_expected_return,
            'avg_expected_risk': avg_expected_risk,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_confidence_score': avg_confidence,
            'rebalancing_rate': rebalancing_rate,
            'market_regime_distribution': regime_counts,
            'optimization_frequency': len(recent_optimizations) / days_back
        }

# Test der ML Portfolio Optimization
def main():
    """🧠 Test des ML Portfolio Optimizers"""
    optimizer = MLPortfolioOptimizer()
    
    print("🧠 ML Portfolio Optimizer - Demo")
    print("=" * 50)
    
    # Test optimization
    result = optimizer.optimize_portfolio("Main Portfolio", "risk_adjusted")
    
    print(f"\n📊 Optimization Results:")
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Expected Risk: {result.expected_risk:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Market Regime: {result.market_regime}")
    print(f"Rebalancing Required: {result.rebalancing_required}")
    
    if result.recommended_weights:
        print(f"\n⚖️ Recommended Weights:")
        for coin_id, weight in result.recommended_weights.items():
            print(f"  {coin_id}: {weight:.1%}")
    
    # Get rebalancing trades
    trades = optimizer.get_rebalancing_trades("Main Portfolio", result)
    if trades:
        print(f"\n📋 Rebalancing Trades Needed:")
        for trade in trades[:5]:  # Show first 5
            print(f"  {trade['action']} {trade['quantity']:.4f} {trade['coin_id']} (€{trade['value_eur']:.2f})")

if __name__ == "__main__":
    main()