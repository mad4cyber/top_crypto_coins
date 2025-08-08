#!/usr/bin/env python3
"""
📊 Crypto Portfolio Manager
Autor: mad4cyber
Version: 1.0 - Portfolio Edition

🚀 FEATURES:
- Multi-Coin Portfolio Tracking
- P&L Calculations with AI Predictions
- Risk Management & Position Sizing
- Real-time Portfolio Valuation
- Performance Analytics
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass, asdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ai_predictor import CryptoAIPredictor
from sentiment_analyzer import MarketSentimentAnalyzer

@dataclass
class Position:
    """💰 Einzelne Portfolio Position"""
    coin_id: str
    coin_symbol: str
    quantity: float
    entry_price: float
    entry_date: str
    entry_confidence: float = 0.0
    notes: str = ""
    
    @property
    def entry_value(self) -> float:
        return self.quantity * self.entry_price

@dataclass
class Portfolio:
    """📊 Komplettes Portfolio"""
    name: str
    positions: List[Position]
    cash_balance: float = 10000.0  # Startet mit 10k EUR
    created_date: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

class PortfolioManager:
    """📊 Crypto Portfolio Management System"""
    
    def __init__(self, portfolio_file: str = "portfolio_data.json"):
        self.portfolio_file = portfolio_file
        self.ai_predictor = CryptoAIPredictor()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Lade bestehende Portfolios
        self.load_portfolios()
        
    def load_portfolios(self):
        """📁 Lade Portfolios aus Datei"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                
                for portfolio_name, portfolio_data in data.items():
                    positions = []
                    for pos_data in portfolio_data.get('positions', []):
                        positions.append(Position(**pos_data))
                    
                    portfolio = Portfolio(
                        name=portfolio_data['name'],
                        positions=positions,
                        cash_balance=portfolio_data.get('cash_balance', 10000.0),
                        created_date=portfolio_data.get('created_date', ''),
                        last_updated=portfolio_data.get('last_updated', '')
                    )
                    self.portfolios[portfolio_name] = portfolio
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden der Portfolios: {e}")
        
        # Erstelle Default Portfolio falls keins vorhanden
        if not self.portfolios:
            self.create_portfolio("Main Portfolio")
    
    def save_portfolios(self):
        """💾 Speichere Portfolios in Datei"""
        try:
            data = {}
            for name, portfolio in self.portfolios.items():
                data[name] = {
                    'name': portfolio.name,
                    'positions': [asdict(pos) for pos in portfolio.positions],
                    'cash_balance': portfolio.cash_balance,
                    'created_date': portfolio.created_date,
                    'last_updated': portfolio.last_updated
                }
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Portfolios: {e}")
    
    def create_portfolio(self, name: str, initial_cash: float = 10000.0) -> Portfolio:
        """📊 Erstelle neues Portfolio"""
        portfolio = Portfolio(
            name=name,
            positions=[],
            cash_balance=initial_cash
        )
        
        self.portfolios[name] = portfolio
        self.save_portfolios()
        return portfolio
    
    def add_position(self, portfolio_name: str, coin_id: str, coin_symbol: str, 
                    quantity: float, entry_price: float, confidence: float = 0.0,
                    notes: str = "") -> bool:
        """💰 Füge Position zum Portfolio hinzu"""
        if portfolio_name not in self.portfolios:
            return False
        
        portfolio = self.portfolios[portfolio_name]
        
        # Prüfe ob genug Cash vorhanden
        required_cash = quantity * entry_price
        if required_cash > portfolio.cash_balance:
            print(f"❌ Nicht genug Cash: €{required_cash:.2f} erforderlich, €{portfolio.cash_balance:.2f} verfügbar")
            return False
        
        # Erstelle Position
        position = Position(
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=datetime.now().isoformat(),
            entry_confidence=confidence,
            notes=notes
        )
        
        # Füge Position hinzu und reduziere Cash
        portfolio.positions.append(position)
        portfolio.cash_balance -= required_cash
        portfolio.last_updated = datetime.now().isoformat()
        
        self.save_portfolios()
        print(f"✅ Position hinzugefügt: {quantity} {coin_symbol} @ €{entry_price:.4f}")
        return True
    
    def remove_position(self, portfolio_name: str, coin_id: str, 
                       current_price: float, quantity: Optional[float] = None) -> bool:
        """💸 Entferne Position (Verkauf)"""
        if portfolio_name not in self.portfolios:
            return False
        
        portfolio = self.portfolios[portfolio_name]
        
        # Finde Position
        position_index = -1
        for i, pos in enumerate(portfolio.positions):
            if pos.coin_id == coin_id:
                position_index = i
                break
        
        if position_index == -1:
            print(f"❌ Position für {coin_id} nicht gefunden")
            return False
        
        position = portfolio.positions[position_index]
        
        # Bestimme Verkaufsmenge
        sell_quantity = quantity if quantity is not None else position.quantity
        if sell_quantity > position.quantity:
            print(f"❌ Nicht genug Coins: {sell_quantity} > {position.quantity}")
            return False
        
        # Berechne Verkaufserlös
        sale_proceeds = sell_quantity * current_price
        
        # Update Portfolio
        portfolio.cash_balance += sale_proceeds
        
        if sell_quantity == position.quantity:
            # Komplette Position verkaufen
            portfolio.positions.pop(position_index)
        else:
            # Teilverkauf
            portfolio.positions[position_index].quantity -= sell_quantity
        
        portfolio.last_updated = datetime.now().isoformat()
        self.save_portfolios()
        
        # Berechne P&L
        cost_basis = sell_quantity * position.entry_price
        pnl = sale_proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        print(f"✅ Verkauf: {sell_quantity} {position.coin_symbol} @ €{current_price:.4f}")
        print(f"💰 P&L: €{pnl:.2f} ({pnl_pct:+.2f}%)")
        return True
    
    def get_current_prices(self, coin_ids: List[str]) -> Dict[str, float]:
        """💱 Hole aktuelle Preise für Coins"""
        import requests
        
        prices = {}
        try:
            # CoinGecko API für aktuelle Preise
            ids_param = ','.join(coin_ids)
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=eur"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            for coin_id in coin_ids:
                if coin_id in data:
                    prices[coin_id] = data[coin_id]['eur']
                else:
                    prices[coin_id] = 0.0
                    
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Preise: {e}")
            # Fallback: Verwende AI Predictor für Preise
            for coin_id in coin_ids:
                try:
                    prediction = self.ai_predictor.predict_future_prices(coin_id)
                    prices[coin_id] = prediction.get('current_price', 0.0)
                except:
                    prices[coin_id] = 0.0
        
        return prices
    
    def calculate_portfolio_value(self, portfolio_name: str) -> Dict:
        """💎 Berechne aktuellen Portfolio-Wert"""
        if portfolio_name not in self.portfolios:
            return {'error': 'Portfolio nicht gefunden'}
        
        portfolio = self.portfolios[portfolio_name]
        
        # Hole aktuelle Preise
        coin_ids = list(set([pos.coin_id for pos in portfolio.positions]))
        current_prices = self.get_current_prices(coin_ids)
        
        position_values = []
        total_current_value = 0
        total_cost_basis = 0
        
        for position in portfolio.positions:
            current_price = current_prices.get(position.coin_id, 0)
            current_value = position.quantity * current_price
            cost_basis = position.entry_value
            
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            position_data = {
                'coin_id': position.coin_id,
                'coin_symbol': position.coin_symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_date': position.entry_date,
                'entry_confidence': position.entry_confidence,
                'notes': position.notes
            }
            
            position_values.append(position_data)
            total_current_value += current_value
            total_cost_basis += cost_basis
        
        # Portfolio-Gesamtwerte
        total_portfolio_value = total_current_value + portfolio.cash_balance
        total_pnl = total_current_value - total_cost_basis
        total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        # Asset Allocation
        allocation = {}
        for pos_data in position_values:
            weight = (pos_data['current_value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            allocation[pos_data['coin_symbol']] = weight
        
        cash_weight = (portfolio.cash_balance / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        allocation['CASH'] = cash_weight
        
        return {
            'portfolio_name': portfolio_name,
            'positions': position_values,
            'summary': {
                'total_portfolio_value': total_portfolio_value,
                'total_invested': total_cost_basis,
                'cash_balance': portfolio.cash_balance,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'position_count': len(portfolio.positions),
                'created_date': portfolio.created_date,
                'last_updated': portfolio.last_updated
            },
            'allocation': allocation,
            'performance_metrics': self.calculate_performance_metrics(portfolio, position_values)
        }
    
    def calculate_performance_metrics(self, portfolio: Portfolio, position_values: List[Dict]) -> Dict:
        """📊 Berechne Performance-Metriken"""
        if not position_values:
            return {}
        
        # Gewinner vs Verlierer
        winners = [pos for pos in position_values if pos['pnl'] > 0]
        losers = [pos for pos in position_values if pos['pnl'] < 0]
        
        # Durchschnittliche Haltezeit
        hold_times = []
        for pos in position_values:
            entry_date = datetime.fromisoformat(pos['entry_date'].replace('Z', '+00:00'))
            hold_days = (datetime.now() - entry_date.replace(tzinfo=None)).days
            hold_times.append(hold_days)
        
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        # Beste/Schlechteste Position
        best_position = max(position_values, key=lambda x: x['pnl_pct']) if position_values else {}
        worst_position = min(position_values, key=lambda x: x['pnl_pct']) if position_values else {}
        
        return {
            'winners_count': len(winners),
            'losers_count': len(losers),
            'win_rate': (len(winners) / len(position_values) * 100) if position_values else 0,
            'avg_winner_pct': np.mean([pos['pnl_pct'] for pos in winners]) if winners else 0,
            'avg_loser_pct': np.mean([pos['pnl_pct'] for pos in losers]) if losers else 0,
            'avg_hold_time_days': avg_hold_time,
            'best_position': {
                'symbol': best_position.get('coin_symbol', ''),
                'pnl_pct': best_position.get('pnl_pct', 0)
            },
            'worst_position': {
                'symbol': worst_position.get('coin_symbol', ''),
                'pnl_pct': worst_position.get('pnl_pct', 0)
            }
        }
    
    def suggest_position_size(self, portfolio_name: str, coin_id: str, 
                             max_position_pct: float = 5.0) -> Dict:
        """🎯 Empfehle Positionsgröße basierend auf AI + Portfolio"""
        if portfolio_name not in self.portfolios:
            return {'error': 'Portfolio nicht gefunden'}
        
        portfolio = self.portfolios[portfolio_name]
        portfolio_value = self.calculate_portfolio_value(portfolio_name)
        
        if 'error' in portfolio_value:
            return portfolio_value
        
        # Hole AI-Prognose
        try:
            prediction = self.ai_predictor.predict_future_prices(coin_id)
            confidence = prediction.get('confidence', 0)
            price_change_pct = abs(prediction.get('price_change_pct', 0))
            current_price = prediction.get('current_price', 0)
            
        except Exception as e:
            return {'error': f'AI-Prognose Fehler: {str(e)}'}
        
        # Basis-Positionsgröße basierend auf Konfidenz
        total_value = portfolio_value['summary']['total_portfolio_value']
        
        # Konfidenz-basierte Adjustierung
        if confidence > 0.8:
            confidence_multiplier = 1.5  # Erhöhe Position bei hoher Konfidenz
        elif confidence > 0.6:
            confidence_multiplier = 1.2
        elif confidence > 0.4:
            confidence_multiplier = 1.0
        elif confidence > 0.2:
            confidence_multiplier = 0.5
        else:
            confidence_multiplier = 0.2  # Sehr kleine Position bei niedriger Konfidenz
        
        # Volatilitäts-Adjustierung
        volatility_multiplier = max(0.3, 1 - (price_change_pct / 100))  # Reduziere bei hoher erwarteter Volatilität
        
        # Berechne empfohlene Positionsgröße
        base_position_pct = min(max_position_pct, 10.0)  # Max 10% des Portfolios
        adjusted_position_pct = base_position_pct * confidence_multiplier * volatility_multiplier
        
        recommended_eur = total_value * (adjusted_position_pct / 100)
        recommended_quantity = recommended_eur / current_price if current_price > 0 else 0
        
        # Prüfe verfügbares Cash
        available_cash = portfolio.cash_balance
        max_possible_eur = min(recommended_eur, available_cash)
        max_possible_quantity = max_possible_eur / current_price if current_price > 0 else 0
        
        return {
            'coin_id': coin_id,
            'current_price': current_price,
            'ai_confidence': confidence,
            'expected_price_change_pct': prediction.get('price_change_pct', 0),
            'portfolio_value': total_value,
            'available_cash': available_cash,
            'recommendation': {
                'position_pct': adjusted_position_pct,
                'eur_amount': recommended_eur,
                'quantity': recommended_quantity,
                'confidence_multiplier': confidence_multiplier,
                'volatility_adjustment': volatility_multiplier
            },
            'maximum_possible': {
                'eur_amount': max_possible_eur,
                'quantity': max_possible_quantity
            },
            'risk_assessment': {
                'risk_level': 'LOW' if confidence > 0.7 else 'MEDIUM' if confidence > 0.4 else 'HIGH',
                'stop_loss_pct': price_change_pct * 0.5,  # 50% der erwarteten Bewegung
                'take_profit_pct': price_change_pct * 1.5   # 150% der erwarteten Bewegung
            }
        }
    
    def get_portfolio_risk_metrics(self, portfolio_name: str) -> Dict:
        """⚠️ Berechne Portfolio-Risiko Metriken"""
        portfolio_data = self.calculate_portfolio_value(portfolio_name)
        
        if 'error' in portfolio_data:
            return portfolio_data
        
        positions = portfolio_data['positions']
        allocation = portfolio_data['allocation']
        
        # Konzentrations-Risiko
        max_position_weight = max([weight for symbol, weight in allocation.items() if symbol != 'CASH'], default=0)
        concentration_risk = 'HIGH' if max_position_weight > 30 else 'MEDIUM' if max_position_weight > 15 else 'LOW'
        
        # Diversifikations-Score
        num_positions = len(positions)
        diversification_score = min(10, num_positions * 2)  # Max Score 10
        
        # Volatilitäts-Risiko (basierend auf Positionen mit hoher erwarteter Bewegung)
        high_vol_positions = [pos for pos in positions if abs(pos.get('expected_change_pct', 0)) > 10]
        volatility_risk = len(high_vol_positions) / len(positions) if positions else 0
        
        # Cash-Position
        cash_pct = allocation.get('CASH', 0)
        cash_level = 'HIGH' if cash_pct > 50 else 'MEDIUM' if cash_pct > 20 else 'LOW'
        
        return {
            'portfolio_name': portfolio_name,
            'risk_metrics': {
                'concentration_risk': concentration_risk,
                'max_position_weight': max_position_weight,
                'diversification_score': diversification_score,
                'volatility_risk_ratio': volatility_risk,
                'cash_level': cash_level,
                'cash_percentage': cash_pct,
                'total_positions': num_positions
            },
            'recommendations': self.get_risk_recommendations(concentration_risk, diversification_score, cash_pct)
        }
    
    def get_risk_recommendations(self, concentration_risk: str, diversification_score: int, cash_pct: float) -> List[str]:
        """💡 Risiko-Management Empfehlungen"""
        recommendations = []
        
        if concentration_risk == 'HIGH':
            recommendations.append("⚠️ Reduziere Konzentrations-Risiko: Größte Position ist zu dominant")
        
        if diversification_score < 6:
            recommendations.append("📊 Erhöhe Diversifikation: Mehr Positionen hinzufügen")
        
        if cash_pct < 10:
            recommendations.append("💰 Erhöhe Cash-Reserve für Opportunities")
        elif cash_pct > 60:
            recommendations.append("🚀 Zu viel Cash: Überlege weitere Investments")
        
        if not recommendations:
            recommendations.append("✅ Portfolio-Risiko ist gut ausbalanciert")
        
        return recommendations

# Test der Portfolio-Funktionen
def main():
    """📊 Test des Portfolio Management Systems"""
    pm = PortfolioManager()
    
    print("📊 Portfolio Manager - Demo")
    print("=" * 50)
    
    # Zeige aktuelles Portfolio
    portfolio_data = pm.calculate_portfolio_value("Main Portfolio")
    
    if 'error' not in portfolio_data:
        summary = portfolio_data['summary']
        print(f"💎 Portfolio Wert: €{summary['total_portfolio_value']:.2f}")
        print(f"💰 Cash: €{summary['cash_balance']:.2f}")
        print(f"📈 P&L: €{summary['total_pnl']:.2f} ({summary['total_pnl_pct']:+.2f}%)")
        print(f"📊 Positionen: {summary['position_count']}")
    
    # Teste Position Suggestion
    print("\n🎯 Position Size Empfehlung für Bitcoin:")
    suggestion = pm.suggest_position_size("Main Portfolio", "bitcoin")
    
    if 'error' not in suggestion:
        rec = suggestion['recommendation']
        print(f"💡 Empfohlen: €{rec['eur_amount']:.2f} ({rec['position_pct']:.1f}% des Portfolios)")
        print(f"🪙 Menge: {rec['quantity']:.6f} BTC")
        print(f"📊 AI Konfidenz: {suggestion['ai_confidence']:.1%}")

if __name__ == "__main__":
    main()