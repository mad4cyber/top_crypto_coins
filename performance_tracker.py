#!/usr/bin/env python3
"""
📊 AI Performance Tracking & Backtesting System
Autor: mad4cyber
Version: 1.0 - Performance Edition

🚀 FEATURES:
- AI Prediction Accuracy Tracking
- Real-time Performance Monitoring
- Backtesting Engine for Trading Strategies
- ROI Tracking of AI Recommendations
- Model Performance Comparison
- Advanced Analytics & Visualization
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import warnings
warnings.filterwarnings('ignore')

from ai_predictor import CryptoAIPredictor
from portfolio_manager import PortfolioManager

@dataclass
class PredictionRecord:
    """📈 Einzelne Prediction mit Tracking-Daten"""
    id: str
    coin_id: str
    coin_symbol: str
    prediction_date: str
    target_date: str
    predicted_price: float
    current_price: float
    predicted_change_pct: float
    confidence: float
    model_name: str = "ensemble"
    actual_price: Optional[float] = None
    actual_change_pct: Optional[float] = None
    accuracy_score: Optional[float] = None
    is_resolved: bool = False
    resolved_date: Optional[str] = None
    
    def __post_init__(self):
        if not self.prediction_date:
            self.prediction_date = datetime.now().isoformat()
        if not self.target_date:
            self.target_date = (datetime.now() + timedelta(days=1)).isoformat()

@dataclass
class BacktestResult:
    """📊 Backtesting Ergebnis"""
    strategy_name: str
    coin_id: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict]
    performance_metrics: Dict

class PerformanceTracker:
    """📊 AI Performance Tracking & Backtesting System"""
    
    def __init__(self, data_file: str = "performance_data.json"):
        self.data_file = data_file
        self.ai_predictor = CryptoAIPredictor()
        self.portfolio_manager = PortfolioManager()
        
        # Data Storage
        self.prediction_records: Dict[str, PredictionRecord] = {}
        self.backtest_results: List[BacktestResult] = []
        self.model_performance: Dict[str, Dict] = {}
        
        # Load existing data
        self.load_performance_data()
        
    def load_performance_data(self):
        """📁 Lade Performance-Daten aus Datei"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load prediction records
                for record_id, record_data in data.get('prediction_records', {}).items():
                    self.prediction_records[record_id] = PredictionRecord(**record_data)
                
                # Load backtest results
                for result_data in data.get('backtest_results', []):
                    trades = result_data.pop('trades', [])
                    performance_metrics = result_data.pop('performance_metrics', {})
                    result = BacktestResult(**result_data)
                    result.trades = trades
                    result.performance_metrics = performance_metrics
                    self.backtest_results.append(result)
                
                # Load model performance
                self.model_performance = data.get('model_performance', {})
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden der Performance-Daten: {e}")
    
    def save_performance_data(self):
        """💾 Speichere Performance-Daten in Datei"""
        try:
            data = {
                'prediction_records': {
                    record_id: asdict(record) for record_id, record in self.prediction_records.items()
                },
                'backtest_results': [
                    {
                        **asdict(result),
                        'trades': result.trades,
                        'performance_metrics': result.performance_metrics
                    } for result in self.backtest_results
                ],
                'model_performance': self.model_performance,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Performance-Daten: {e}")
    
    def record_prediction(self, coin_id: str, coin_symbol: str, prediction_data: Dict, 
                         model_name: str = "ensemble") -> str:
        """📝 Zeichne AI-Prognose für späteren Vergleich auf"""
        record_id = f"{coin_id}_{model_name}_{int(time.time())}"
        
        target_date = datetime.now() + timedelta(days=1)  # 24h Prognose
        
        record = PredictionRecord(
            id=record_id,
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            prediction_date=datetime.now().isoformat(),
            target_date=target_date.isoformat(),
            predicted_price=prediction_data.get('predicted_price', 0),
            current_price=prediction_data.get('current_price', 0),
            predicted_change_pct=prediction_data.get('price_change_pct', 0),
            confidence=prediction_data.get('confidence', 0),
            model_name=model_name
        )
        
        self.prediction_records[record_id] = record
        self.save_performance_data()
        
        print(f"📝 Recorded prediction: {coin_symbol} {record.predicted_change_pct:+.2f}% (Confidence: {record.confidence:.1%})")
        return record_id
    
    def update_prediction_actuals(self) -> int:
        """🔄 Update Predictions mit tatsächlichen Preisen"""
        updated_count = 0
        current_time = datetime.now()
        
        # Finde Predictions die resolved werden können
        resolvable_predictions = [
            record for record in self.prediction_records.values()
            if not record.is_resolved and 
               datetime.fromisoformat(record.target_date.replace('Z', '')) <= current_time
        ]
        
        if not resolvable_predictions:
            return 0
        
        # Hole aktuelle Preise für alle relevanten Coins
        coin_ids = list(set([record.coin_id for record in resolvable_predictions]))
        current_prices = self.portfolio_manager.get_current_prices(coin_ids)
        
        for record in resolvable_predictions:
            try:
                actual_price = current_prices.get(record.coin_id, 0)
                
                if actual_price > 0:
                    # Berechne tatsächliche Änderung
                    actual_change_pct = ((actual_price - record.current_price) / record.current_price) * 100
                    
                    # Berechne Accuracy Score
                    # Accuracy basierend auf Direction (richtige Richtung) und Magnitude (wie nah dran)
                    direction_correct = (record.predicted_change_pct > 0) == (actual_change_pct > 0)
                    direction_score = 1.0 if direction_correct else 0.0
                    
                    # Magnitude accuracy (wie nah die Prognose war)
                    magnitude_error = abs(record.predicted_change_pct - actual_change_pct)
                    magnitude_score = max(0, 1 - (magnitude_error / 20))  # 20% error = 0 score
                    
                    # Gewichteter Accuracy Score (60% Direction, 40% Magnitude)
                    accuracy_score = (direction_score * 0.6) + (magnitude_score * 0.4)
                    
                    # Update record
                    record.actual_price = actual_price
                    record.actual_change_pct = actual_change_pct
                    record.accuracy_score = accuracy_score
                    record.is_resolved = True
                    record.resolved_date = current_time.isoformat()
                    
                    updated_count += 1
                    
                    print(f"✅ Updated {record.coin_symbol}: Predicted {record.predicted_change_pct:+.2f}%, Actual {actual_change_pct:+.2f}%, Accuracy: {accuracy_score:.2f}")
            
            except Exception as e:
                print(f"❌ Error updating {record.coin_symbol}: {e}")
        
        if updated_count > 0:
            self.save_performance_data()
            self.update_model_performance_stats()
        
        return updated_count
    
    def update_model_performance_stats(self):
        """📊 Aktualisiere Model Performance Statistiken"""
        resolved_predictions = [r for r in self.prediction_records.values() if r.is_resolved]
        
        if not resolved_predictions:
            return
        
        # Group by model
        model_stats = {}
        
        for record in resolved_predictions:
            model = record.model_name
            if model not in model_stats:
                model_stats[model] = {
                    'predictions': [],
                    'accuracy_scores': [],
                    'confidence_scores': [],
                    'correct_directions': [],
                    'magnitude_errors': []
                }
            
            stats = model_stats[model]
            stats['predictions'].append(record)
            stats['accuracy_scores'].append(record.accuracy_score)
            stats['confidence_scores'].append(record.confidence)
            
            # Direction accuracy
            direction_correct = (record.predicted_change_pct > 0) == (record.actual_change_pct > 0)
            stats['correct_directions'].append(direction_correct)
            
            # Magnitude error
            magnitude_error = abs(record.predicted_change_pct - record.actual_change_pct)
            stats['magnitude_errors'].append(magnitude_error)
        
        # Calculate statistics for each model
        for model, stats in model_stats.items():
            if not stats['accuracy_scores']:
                continue
                
            self.model_performance[model] = {
                'total_predictions': len(stats['predictions']),
                'avg_accuracy_score': np.mean(stats['accuracy_scores']),
                'direction_accuracy': np.mean(stats['correct_directions']),
                'avg_magnitude_error': np.mean(stats['magnitude_errors']),
                'avg_confidence': np.mean(stats['confidence_scores']),
                'best_accuracy': np.max(stats['accuracy_scores']),
                'worst_accuracy': np.min(stats['accuracy_scores']),
                'accuracy_std': np.std(stats['accuracy_scores']),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_performance_summary(self, days_back: int = 30) -> Dict:
        """📊 Hole Performance-Zusammenfassung"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter recent resolved predictions
        recent_predictions = [
            r for r in self.prediction_records.values()
            if r.is_resolved and datetime.fromisoformat(r.prediction_date.replace('Z', '')) >= cutoff_date
        ]
        
        if not recent_predictions:
            return {
                'total_predictions': 0,
                'message': 'No resolved predictions in timeframe'
            }
        
        # Calculate overall stats
        accuracy_scores = [r.accuracy_score for r in recent_predictions]
        confidence_scores = [r.confidence for r in recent_predictions]
        correct_directions = [(r.predicted_change_pct > 0) == (r.actual_change_pct > 0) for r in recent_predictions]
        magnitude_errors = [abs(r.predicted_change_pct - r.actual_change_pct) for r in recent_predictions]
        
        # Group by confidence levels
        high_confidence = [r for r in recent_predictions if r.confidence > 0.7]
        medium_confidence = [r for r in recent_predictions if 0.4 <= r.confidence <= 0.7]
        low_confidence = [r for r in recent_predictions if r.confidence < 0.4]
        
        # Calculate ROI if we had followed all recommendations
        total_return = 0
        winning_trades = 0
        for prediction in recent_predictions:
            if abs(prediction.predicted_change_pct) > 2:  # Only consider significant predictions
                if (prediction.predicted_change_pct > 0 and prediction.actual_change_pct > 0) or \
                   (prediction.predicted_change_pct < 0 and prediction.actual_change_pct < 0):
                    total_return += abs(prediction.actual_change_pct)
                    winning_trades += 1
                else:
                    total_return -= abs(prediction.actual_change_pct)
        
        return {
            'timeframe_days': days_back,
            'total_predictions': len(recent_predictions),
            'avg_accuracy_score': np.mean(accuracy_scores),
            'direction_accuracy': np.mean(correct_directions),
            'avg_magnitude_error': np.mean(magnitude_errors),
            'avg_confidence': np.mean(confidence_scores),
            'accuracy_by_confidence': {
                'high_confidence': {
                    'count': len(high_confidence),
                    'avg_accuracy': np.mean([r.accuracy_score for r in high_confidence]) if high_confidence else 0
                },
                'medium_confidence': {
                    'count': len(medium_confidence),
                    'avg_accuracy': np.mean([r.accuracy_score for r in medium_confidence]) if medium_confidence else 0
                },
                'low_confidence': {
                    'count': len(low_confidence),
                    'avg_accuracy': np.mean([r.accuracy_score for r in low_confidence]) if low_confidence else 0
                }
            },
            'theoretical_roi': {
                'total_return_pct': total_return,
                'winning_trades': winning_trades,
                'total_significant_trades': len([r for r in recent_predictions if abs(r.predicted_change_pct) > 2]),
                'win_rate': winning_trades / len([r for r in recent_predictions if abs(r.predicted_change_pct) > 2]) if recent_predictions else 0
            },
            'model_performance': self.model_performance
        }
    
    def run_backtest_strategy(self, coin_id: str, strategy_name: str, 
                            days_back: int = 30, initial_capital: float = 10000) -> BacktestResult:
        """🔄 Führe Backtesting für eine Trading-Strategie aus"""
        
        print(f"🔄 Starting backtest: {strategy_name} for {coin_id}")
        
        # Get historical predictions for this coin
        historical_predictions = [
            r for r in self.prediction_records.values()
            if r.coin_id == coin_id and r.is_resolved
        ]
        
        if len(historical_predictions) < 5:
            return BacktestResult(
                strategy_name=strategy_name,
                coin_id=coin_id,
                start_date=datetime.now().isoformat(),
                end_date=datetime.now().isoformat(),
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=[],
                performance_metrics={}
            )
        
        # Sort by prediction date
        historical_predictions.sort(key=lambda x: x.prediction_date)
        
        # Backtesting simulation
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        trades = []
        capital_history = [capital]
        max_capital = capital
        max_drawdown = 0
        
        for prediction in historical_predictions:
            try:
                # Strategy: High confidence + significant change prediction
                if prediction.confidence > 0.6 and abs(prediction.predicted_change_pct) > 3:
                    
                    # Entry logic
                    if position == 0:  # Not in position
                        if prediction.predicted_change_pct > 0:  # Predicted up
                            position = 1  # Go long
                            entry_price = prediction.current_price
                            print(f"📈 Long entry at {entry_price:.4f} (Predicted: +{prediction.predicted_change_pct:.2f}%)")
                        elif prediction.predicted_change_pct < 0:  # Predicted down
                            position = -1  # Go short
                            entry_price = prediction.current_price
                            print(f"📉 Short entry at {entry_price:.4f} (Predicted: {prediction.predicted_change_pct:.2f}%)")
                    
                    # Exit logic (next day after target date)
                    if position != 0 and prediction.actual_price:
                        exit_price = prediction.actual_price
                        
                        if position == 1:  # Long position
                            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                        else:  # Short position
                            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                        
                        pnl_amount = capital * (pnl_pct / 100)
                        capital += pnl_amount
                        
                        trades.append({
                            'entry_date': prediction.prediction_date,
                            'exit_date': prediction.resolved_date,
                            'position_type': 'long' if position == 1 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'predicted_change': prediction.predicted_change_pct,
                            'actual_change': prediction.actual_change_pct,
                            'pnl_pct': pnl_pct,
                            'pnl_amount': pnl_amount,
                            'confidence': prediction.confidence,
                            'capital_after': capital
                        })
                        
                        print(f"💰 Exit at {exit_price:.4f}, P&L: {pnl_pct:+.2f}% (${pnl_amount:+.2f})")
                        
                        position = 0  # Close position
                        entry_price = 0
                        
                        # Update drawdown
                        if capital > max_capital:
                            max_capital = capital
                        drawdown = (max_capital - capital) / max_capital * 100
                        max_drawdown = max(max_drawdown, drawdown)
                        
                        capital_history.append(capital)
            
            except Exception as e:
                print(f"❌ Error in backtest step: {e}")
                continue
        
        # Calculate final metrics
        total_return_pct = ((capital - initial_capital) / initial_capital) * 100
        winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
        losing_trades = len([t for t in trades if t['pnl_pct'] <= 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        # Sharpe ratio (simplified)
        if len(capital_history) > 1:
            returns = np.diff(capital_history) / capital_history[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Performance metrics
        performance_metrics = {
            'avg_trade_return': np.mean([t['pnl_pct'] for t in trades]) if trades else 0,
            'best_trade': max([t['pnl_pct'] for t in trades]) if trades else 0,
            'worst_trade': min([t['pnl_pct'] for t in trades]) if trades else 0,
            'avg_holding_period': 1,  # Fixed at 1 day for now
            'total_days': days_back,
            'trades_per_month': len(trades) / (days_back / 30) if days_back > 0 else 0
        }
        
        result = BacktestResult(
            strategy_name=strategy_name,
            coin_id=coin_id,
            start_date=historical_predictions[0].prediction_date if historical_predictions else datetime.now().isoformat(),
            end_date=historical_predictions[-1].resolved_date if historical_predictions else datetime.now().isoformat(),
            initial_capital=initial_capital,
            final_capital=capital,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            performance_metrics=performance_metrics
        )
        
        self.backtest_results.append(result)
        self.save_performance_data()
        
        print(f"✅ Backtest completed: {total_return_pct:+.2f}% return, {win_rate:.1%} win rate")
        return result
    
    def get_coin_performance_ranking(self, days_back: int = 30) -> List[Dict]:
        """🏆 Ranking der Coins nach AI-Performance"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        coin_stats = {}
        
        for record in self.prediction_records.values():
            if not record.is_resolved:
                continue
            if datetime.fromisoformat(record.prediction_date.replace('Z', '')) < cutoff_date:
                continue
                
            coin = record.coin_id
            if coin not in coin_stats:
                coin_stats[coin] = {
                    'coin_id': coin,
                    'coin_symbol': record.coin_symbol,
                    'predictions': [],
                    'accuracy_scores': [],
                    'roi_if_followed': 0,
                    'significant_predictions': 0
                }
            
            stats = coin_stats[coin]
            stats['predictions'].append(record)
            stats['accuracy_scores'].append(record.accuracy_score)
            
            # Calculate ROI if we followed prediction
            if abs(record.predicted_change_pct) > 2:  # Significant prediction
                stats['significant_predictions'] += 1
                if (record.predicted_change_pct > 0 and record.actual_change_pct > 0) or \
                   (record.predicted_change_pct < 0 and record.actual_change_pct < 0):
                    stats['roi_if_followed'] += abs(record.actual_change_pct)
                else:
                    stats['roi_if_followed'] -= abs(record.actual_change_pct) * 0.5  # Partial loss
        
        # Calculate final metrics for each coin
        ranking = []
        for coin, stats in coin_stats.items():
            if len(stats['predictions']) >= 3:  # Minimum predictions for ranking
                ranking.append({
                    'coin_id': stats['coin_id'],
                    'coin_symbol': stats['coin_symbol'],
                    'total_predictions': len(stats['predictions']),
                    'avg_accuracy': np.mean(stats['accuracy_scores']),
                    'theoretical_roi': stats['roi_if_followed'],
                    'significant_predictions': stats['significant_predictions'],
                    'score': np.mean(stats['accuracy_scores']) * 0.7 + (stats['roi_if_followed'] / 100) * 0.3  # Combined score
                })
        
        # Sort by score
        ranking.sort(key=lambda x: x['score'], reverse=True)
        return ranking

# Test der Performance-Funktionen
def main():
    """📊 Test des Performance Tracking Systems"""
    tracker = PerformanceTracker()
    
    print("📊 Performance Tracker - Demo")
    print("=" * 50)
    
    # Update aktuelle Predictions
    updated = tracker.update_prediction_actuals()
    print(f"🔄 Updated {updated} predictions")
    
    # Performance Summary
    summary = tracker.get_performance_summary(days_back=7)
    print(f"\n📈 Performance Summary (7 days):")
    print(f"Total Predictions: {summary['total_predictions']}")
    
    if summary['total_predictions'] > 0:
        print(f"Average Accuracy: {summary['avg_accuracy_score']:.2f}")
        print(f"Direction Accuracy: {summary['direction_accuracy']:.1%}")
        print(f"Theoretical ROI: {summary['theoretical_roi']['total_return_pct']:+.2f}%")
    
    # Coin Ranking
    ranking = tracker.get_coin_performance_ranking(days_back=14)
    print(f"\n🏆 Top Performing Coins:")
    for i, coin in enumerate(ranking[:5]):
        print(f"{i+1}. {coin['coin_symbol']}: {coin['avg_accuracy']:.2f} accuracy, {coin['theoretical_roi']:+.1f}% ROI")

if __name__ == "__main__":
    main()