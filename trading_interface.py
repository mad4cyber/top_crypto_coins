#!/usr/bin/env python3
"""
💼 Professional Trading Interface System
Autor: mad4cyber
Version: 1.0 - Professional Trading Edition

🚀 FEATURES:
- Advanced Order Management System
- Stop-Loss & Take-Profit Automation
- Professional Risk Management
- Real-time Position Tracking & P&L
- Advanced Order Types (Market, Limit, Stop)
- Portfolio Risk Controls
- Trade Execution Engine
"""

import json
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

from portfolio_manager import PortfolioManager
from performance_tracker import PerformanceTracker

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CLOSING = "closing"

@dataclass
class TradingOrder:
    """📝 Trading Order Structure"""
    order_id: str
    coin_id: str
    coin_symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fees: float = 0.0
    portfolio_name: str = "Main Portfolio"
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

@dataclass
class Position:
    """📊 Trading Position"""
    position_id: str
    coin_id: str
    coin_symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    opened_at: str
    portfolio_name: str = "Main Portfolio"
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_price: Optional[float] = None  # For trailing stops
    min_price: Optional[float] = None  # For trailing stops
    
    def __post_init__(self):
        if not self.opened_at:
            self.opened_at = datetime.now().isoformat()

@dataclass
class TradeExecution:
    """⚡ Trade Execution Record"""
    execution_id: str
    order_id: str
    coin_symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: str
    fees: float
    portfolio_name: str
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class RiskMetrics:
    """⚠️ Risk Management Metrics"""
    portfolio_name: str
    total_exposure: float
    max_position_size: float
    current_drawdown: float
    var_1_day: float  # Value at Risk
    sharpe_ratio: float
    risk_score: float  # 0-100
    margin_usage: float
    open_positions: int
    total_unrealized_pnl: float

class TradingInterface:
    """💼 Professional Trading Interface"""
    
    def __init__(self, data_file: str = "trading_data.json"):
        self.data_file = data_file
        self.portfolio_manager = PortfolioManager()
        self.performance_tracker = PerformanceTracker()
        
        # Trading data
        self.orders: Dict[str, TradingOrder] = {}
        self.positions: Dict[str, Position] = {}
        self.executions: List[TradeExecution] = []
        
        # Risk management
        self.max_position_size_pct = 0.25  # 25% max position size
        self.max_total_exposure = 1.0  # 100% max exposure
        self.stop_loss_default = 0.05  # 5% default stop loss
        self.take_profit_default = 0.15  # 15% default take profit
        self.min_trade_size = 10.0  # $10 minimum trade size
        
        # Trading fees simulation
        self.maker_fee = 0.001  # 0.1%
        self.taker_fee = 0.0015  # 0.15%
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.automation_active = False
        self.automation_thread = None
        
        # Real-time tracking
        self.realtime_data = {
            'positions': {},
            'pnl_history': [],
            'price_updates': {},
            'alerts': []
        }
        
        # Load existing data
        self.load_trading_data()
    
    def load_trading_data(self):
        """📁 Load trading data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load orders
                for order_id, order_data in data.get('orders', {}).items():
                    order_data['order_type'] = OrderType(order_data['order_type'])
                    order_data['side'] = OrderSide(order_data['side'])
                    order_data['status'] = OrderStatus(order_data['status'])
                    self.orders[order_id] = TradingOrder(**order_data)
                
                # Load positions
                for pos_id, pos_data in data.get('positions', {}).items():
                    pos_data['status'] = PositionStatus(pos_data['status'])
                    self.positions[pos_id] = Position(**pos_data)
                
                # Load executions
                for exec_data in data.get('executions', []):
                    exec_data['side'] = OrderSide(exec_data['side'])
                    execution = TradeExecution(**exec_data)
                    self.executions.append(execution)
                    
            except Exception as e:
                print(f"❌ Error loading trading data: {e}")
    
    def save_trading_data(self):
        """💾 Save trading data to file"""
        try:
            data = {
                'orders': {
                    order_id: {
                        **asdict(order),
                        'order_type': order.order_type.value,
                        'side': order.side.value,
                        'status': order.status.value
                    } for order_id, order in self.orders.items()
                },
                'positions': {
                    pos_id: {
                        **asdict(position),
                        'status': position.status.value
                    } for pos_id, position in self.positions.items()
                },
                'executions': [
                    {
                        **asdict(execution),
                        'side': execution.side.value
                    } for execution in self.executions
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving trading data: {e}")
    
    def create_order(self, coin_id: str, coin_symbol: str, order_type: OrderType, 
                    side: OrderSide, quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None, portfolio_name: str = "Main Portfolio",
                    notes: str = "") -> str:
        """📝 Create new trading order"""
        print(f"📝 Creating {order_type.value} {side.value} order: {quantity} {coin_symbol}")
        
        try:
            # Validate order
            validation_result = self.validate_order(coin_id, side, quantity, price, portfolio_name)
            if not validation_result['valid']:
                raise ValueError(validation_result['reason'])
            
            # Generate order ID
            order_id = f"{coin_symbol}_{side.value}_{int(time.time())}"
            
            # Create order
            order = TradingOrder(
                order_id=order_id,
                coin_id=coin_id,
                coin_symbol=coin_symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                portfolio_name=portfolio_name,
                notes=notes
            )
            
            # Process order based on type
            if order_type == OrderType.MARKET:
                # Execute immediately at current market price
                current_price = self.get_current_price(coin_id)
                if current_price > 0:
                    self.execute_order(order, current_price)
                else:
                    order.status = OrderStatus.REJECTED
                    print(f"❌ Order rejected: Unable to get current price for {coin_symbol}")
            else:
                # Place order in order book
                order.status = OrderStatus.OPEN
                print(f"📋 {order_type.value.title()} order placed: {order_id}")
            
            # Store order
            self.orders[order_id] = order
            self.save_trading_data()
            
            return order_id
            
        except Exception as e:
            print(f"❌ Error creating order: {e}")
            raise
    
    def validate_order(self, coin_id: str, side: OrderSide, quantity: float, 
                      price: Optional[float], portfolio_name: str) -> Dict[str, Any]:
        """✅ Validate order parameters"""
        try:
            current_price = self.get_current_price(coin_id)
            if current_price <= 0:
                return {'valid': False, 'reason': 'Unable to get current price'}
            
            order_value = quantity * (price if price else current_price)
            
            # Minimum trade size check
            if order_value < self.min_trade_size:
                return {'valid': False, 'reason': f'Order value ${order_value:.2f} below minimum ${self.min_trade_size}'}
            
            # Get portfolio data
            portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
            if 'error' in portfolio_data:
                return {'valid': False, 'reason': 'Portfolio not found'}
            
            total_value = portfolio_data.get('total_value_eur', 0)
            
            # Position size check
            max_position_value = total_value * self.max_position_size_pct
            if order_value > max_position_value:
                return {'valid': False, 'reason': f'Position size ${order_value:.2f} exceeds max ${max_position_value:.2f}'}
            
            # Available balance check for buy orders
            if side == OrderSide.BUY:
                available_cash = portfolio_data.get('available_cash_eur', 0)
                if order_value > available_cash:
                    return {'valid': False, 'reason': f'Insufficient balance: ${available_cash:.2f} available, ${order_value:.2f} required'}
            
            # For sell orders, check if we have enough quantity
            elif side == OrderSide.SELL:
                current_holdings = self.get_coin_holdings(coin_id, portfolio_name)
                if quantity > current_holdings:
                    return {'valid': False, 'reason': f'Insufficient holdings: {current_holdings} available, {quantity} required'}
            
            return {'valid': True, 'reason': 'Order validated successfully'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    def execute_order(self, order: TradingOrder, execution_price: float):
        """⚡ Execute trading order"""
        print(f"⚡ Executing order {order.order_id} at ${execution_price:.4f}")
        
        try:
            # Calculate fees
            fee_rate = self.maker_fee if order.order_type != OrderType.MARKET else self.taker_fee
            fees = order.quantity * execution_price * fee_rate
            
            # Create execution record
            execution = TradeExecution(
                execution_id=f"exec_{order.order_id}_{int(time.time())}",
                order_id=order.order_id,
                coin_symbol=order.coin_symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                fees=fees,
                portfolio_name=order.portfolio_name
            )
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.fees = fees
            order.updated_at = datetime.now().isoformat()
            
            # Update portfolio
            self.update_portfolio_from_execution(execution)
            
            # Update or create position
            self.update_position_from_execution(execution)
            
            # Store execution
            self.executions.append(execution)
            
            print(f"✅ Order executed: {order.quantity} {order.coin_symbol} at ${execution_price:.4f}")
            print(f"💰 Fees: ${fees:.2f}")
            
        except Exception as e:
            print(f"❌ Error executing order: {e}")
            order.status = OrderStatus.REJECTED
    
    def update_portfolio_from_execution(self, execution: TradeExecution):
        """📊 Update portfolio based on execution"""
        try:
            coin_symbol = execution.coin_symbol
            coin_id = coin_symbol.lower()  # Simplified mapping
            
            if execution.side == OrderSide.BUY:
                # Add position to portfolio
                entry_price = execution.price
                confidence = 0.75  # Default confidence for manual trades
                
                success = self.portfolio_manager.add_position(
                    execution.portfolio_name,
                    coin_id,
                    coin_symbol,
                    execution.quantity,
                    entry_price,
                    confidence,
                    f"Trade execution {execution.execution_id}"
                )
                
                if not success:
                    print(f"⚠️ Warning: Could not add position to portfolio")
            
            elif execution.side == OrderSide.SELL:
                # Remove or reduce position
                current_price = execution.price
                quantity_to_sell = execution.quantity
                
                success = self.portfolio_manager.remove_position(
                    execution.portfolio_name,
                    coin_id,
                    current_price,
                    quantity_to_sell
                )
                
                if not success:
                    print(f"⚠️ Warning: Could not remove position from portfolio")
                    
        except Exception as e:
            print(f"❌ Error updating portfolio from execution: {e}")
    
    def update_position_from_execution(self, execution: TradeExecution):
        """📈 Update position tracking from execution"""
        coin_symbol = execution.coin_symbol
        position_id = f"{coin_symbol}_{execution.portfolio_name}"
        
        try:
            if position_id in self.positions:
                # Update existing position
                position = self.positions[position_id]
                
                if execution.side == OrderSide.BUY:
                    # Add to position (average price calculation)
                    total_cost = (position.quantity * position.avg_entry_price) + (execution.quantity * execution.price)
                    total_quantity = position.quantity + execution.quantity
                    position.avg_entry_price = total_cost / total_quantity
                    position.quantity = total_quantity
                    
                elif execution.side == OrderSide.SELL:
                    # Reduce position
                    position.quantity -= execution.quantity
                    
                    # Calculate realized P&L
                    realized_pnl = execution.quantity * (execution.price - position.avg_entry_price)
                    position.realized_pnl += realized_pnl
                    
                    if position.quantity <= 0:
                        position.status = PositionStatus.CLOSED
                
                # Update current price and unrealized P&L
                position.current_price = execution.price
                if position.quantity > 0:
                    position.unrealized_pnl = position.quantity * (position.current_price - position.avg_entry_price)
                
            else:
                # Create new position
                if execution.side == OrderSide.BUY:
                    position = Position(
                        position_id=position_id,
                        coin_id=execution.coin_symbol.lower(),
                        coin_symbol=execution.coin_symbol,
                        side='long',
                        quantity=execution.quantity,
                        avg_entry_price=execution.price,
                        current_price=execution.price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        status=PositionStatus.OPEN,
                        portfolio_name=execution.portfolio_name
                    )
                    
                    self.positions[position_id] = position
                    
        except Exception as e:
            print(f"❌ Error updating position: {e}")
    
    def get_current_price(self, coin_id: str) -> float:
        """💰 Get current price for coin (simulated)"""
        try:
            # In production, this would call real exchange APIs
            # For now, simulate price movement
            base_prices = {
                'bitcoin': 45000,
                'ethereum': 3200,
                'binancecoin': 320,
                'solana': 85,
                'cardano': 0.8,
                'ripple': 0.6,
                'dogecoin': 0.12
            }
            
            base_price = base_prices.get(coin_id, 100)
            
            # Add random price movement ±2%
            price_movement = np.random.uniform(-0.02, 0.02)
            current_price = base_price * (1 + price_movement)
            
            return round(current_price, 8)
            
        except Exception as e:
            print(f"❌ Error getting current price for {coin_id}: {e}")
            return 0.0
    
    def get_coin_holdings(self, coin_id: str, portfolio_name: str) -> float:
        """📊 Get current holdings for a coin"""
        try:
            portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
            
            if 'error' in portfolio_data:
                return 0.0
            
            positions = portfolio_data.get('positions', [])
            
            for position in positions:
                if position.get('coin_id') == coin_id:
                    return position.get('quantity', 0.0)
            
            return 0.0
            
        except Exception as e:
            print(f"❌ Error getting coin holdings: {e}")
            return 0.0
    
    def create_stop_loss_order(self, position_id: str, stop_price: float) -> Optional[str]:
        """🛑 Create stop-loss order for position"""
        if position_id not in self.positions:
            print(f"❌ Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        try:
            order_id = self.create_order(
                coin_id=position.coin_id,
                coin_symbol=position.coin_symbol,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,  # Always sell to close long positions
                quantity=position.quantity,
                stop_price=stop_price,
                portfolio_name=position.portfolio_name,
                notes=f"Stop-loss for position {position_id}"
            )
            
            # Update position with stop-loss price
            position.stop_loss_price = stop_price
            
            print(f"🛑 Stop-loss order created: {order_id} at ${stop_price:.4f}")
            return order_id
            
        except Exception as e:
            print(f"❌ Error creating stop-loss order: {e}")
            return None
    
    def create_take_profit_order(self, position_id: str, target_price: float) -> Optional[str]:
        """🎯 Create take-profit order for position"""
        if position_id not in self.positions:
            print(f"❌ Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        try:
            order_id = self.create_order(
                coin_id=position.coin_id,
                coin_symbol=position.coin_symbol,
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                quantity=position.quantity,
                price=target_price,
                portfolio_name=position.portfolio_name,
                notes=f"Take-profit for position {position_id}"
            )
            
            # Update position with take-profit price
            position.take_profit_price = target_price
            
            print(f"🎯 Take-profit order created: {order_id} at ${target_price:.4f}")
            return order_id
            
        except Exception as e:
            print(f"❌ Error creating take-profit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """❌ Cancel pending order"""
        if order_id not in self.orders:
            print(f"❌ Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            print(f"❌ Cannot cancel order {order_id} - Status: {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now().isoformat()
        self.save_trading_data()
        
        print(f"✅ Order cancelled: {order_id}")
        return True
    
    def get_open_orders(self, portfolio_name: str = None) -> List[TradingOrder]:
        """📋 Get all open orders"""
        open_orders = [
            order for order in self.orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]
        ]
        
        if portfolio_name:
            open_orders = [o for o in open_orders if o.portfolio_name == portfolio_name]
        
        return open_orders
    
    def get_open_positions(self, portfolio_name: str = None) -> List[Position]:
        """📊 Get all open positions"""
        open_positions = [
            position for position in self.positions.values()
            if position.status == PositionStatus.OPEN
        ]
        
        if portfolio_name:
            open_positions = [p for p in open_positions if p.portfolio_name == portfolio_name]
        
        return open_positions
    
    def update_positions_pnl(self):
        """💰 Update P&L for all open positions"""
        updated_count = 0
        
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                try:
                    # Get current price
                    current_price = self.get_current_price(position.coin_id)
                    
                    if current_price > 0:
                        # Update current price and unrealized P&L
                        position.current_price = current_price
                        position.unrealized_pnl = position.quantity * (current_price - position.avg_entry_price)
                        
                        # Update max/min prices for trailing stops
                        if position.max_price is None or current_price > position.max_price:
                            position.max_price = current_price
                        if position.min_price is None or current_price < position.min_price:
                            position.min_price = current_price
                        
                        updated_count += 1
                
                except Exception as e:
                    print(f"❌ Error updating P&L for {position.coin_symbol}: {e}")
        
        if updated_count > 0:
            self.save_trading_data()
            
        return updated_count
    
    def calculate_risk_metrics(self, portfolio_name: str = "Main Portfolio") -> RiskMetrics:
        """⚠️ Calculate risk management metrics"""
        try:
            # Get portfolio data
            portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
            total_value = portfolio_data.get('total_value_eur', 0)
            
            # Get open positions for this portfolio
            open_positions = self.get_open_positions(portfolio_name)
            
            # Calculate metrics
            total_exposure = sum(pos.quantity * pos.current_price for pos in open_positions)
            max_position = max((pos.quantity * pos.current_price for pos in open_positions), default=0)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
            
            # Current drawdown
            if total_value > 0:
                current_drawdown = abs(min(0, total_unrealized_pnl / total_value * 100))
            else:
                current_drawdown = 0
            
            # Simple VaR calculation (95% confidence, 1 day)
            if len(open_positions) > 0:
                position_values = [pos.quantity * pos.current_price for pos in open_positions]
                var_1_day = np.percentile(position_values, 5) * 0.02  # Assume 2% daily volatility
            else:
                var_1_day = 0
            
            # Risk score (0-100, higher = riskier)
            risk_factors = [
                (total_exposure / total_value if total_value > 0 else 0) * 30,  # Exposure risk
                (len(open_positions) / 10) * 20,  # Concentration risk
                current_drawdown * 0.5,  # Drawdown risk
                (max_position / total_value if total_value > 0 else 0) * 20  # Position size risk
            ]
            risk_score = min(100, sum(risk_factors))
            
            return RiskMetrics(
                portfolio_name=portfolio_name,
                total_exposure=total_exposure,
                max_position_size=max_position,
                current_drawdown=current_drawdown,
                var_1_day=var_1_day,
                sharpe_ratio=0.5,  # Simplified
                risk_score=risk_score,
                margin_usage=0.0,  # Not applicable for spot trading
                open_positions=len(open_positions),
                total_unrealized_pnl=total_unrealized_pnl
            )
            
        except Exception as e:
            print(f"❌ Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_name=portfolio_name,
                total_exposure=0,
                max_position_size=0,
                current_drawdown=0,
                var_1_day=0,
                sharpe_ratio=0,
                risk_score=0,
                margin_usage=0,
                open_positions=0,
                total_unrealized_pnl=0
            )
    
    def get_trading_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """📊 Get trading activity summary"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter recent executions
        recent_executions = [
            exec for exec in self.executions
            if datetime.fromisoformat(exec.timestamp.replace('Z', '')) >= cutoff_date
        ]
        
        if not recent_executions:
            return {'message': 'No recent trading activity'}
        
        # Calculate metrics
        total_trades = len(recent_executions)
        total_volume = sum(exec.quantity * exec.price for exec in recent_executions)
        total_fees = sum(exec.fees for exec in recent_executions)
        
        buy_trades = [e for e in recent_executions if e.side == OrderSide.BUY]
        sell_trades = [e for e in recent_executions if e.side == OrderSide.SELL]
        
        # P&L calculation (simplified)
        realized_pnl = 0
        for sell in sell_trades:
            # Find corresponding buy (simplified matching)
            matching_buys = [b for b in buy_trades if b.coin_symbol == sell.coin_symbol]
            if matching_buys:
                avg_buy_price = np.mean([b.price for b in matching_buys])
                realized_pnl += sell.quantity * (sell.price - avg_buy_price)
        
        return {
            'days_back': days_back,
            'total_trades': total_trades,
            'total_volume_usd': total_volume,
            'total_fees': total_fees,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'realized_pnl': realized_pnl,
            'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
            'most_traded_coin': max(set(e.coin_symbol for e in recent_executions), 
                                  key=lambda x: len([e for e in recent_executions if e.coin_symbol == x])) if recent_executions else None
        }
    
    def start_automation(self):
        """🤖 Start automated stop-loss and take-profit execution"""
        if self.automation_active:
            print("⚠️ Automation already running")
            return False
        
        self.automation_active = True
        self.automation_thread = threading.Thread(target=self._automation_worker, daemon=True)
        self.automation_thread.start()
        
        print("🤖 Trading automation started")
        return True
    
    def stop_automation(self):
        """⏹️ Stop automated trading"""
        self.automation_active = False
        
        if self.automation_thread and self.automation_thread.is_alive():
            self.automation_thread.join(timeout=5.0)
        
        print("⏹️ Trading automation stopped")
        return True
    
    def _automation_worker(self):
        """🔄 Background worker for automation"""
        print("🔄 Automation worker started")
        
        while self.automation_active:
            try:
                # Check and execute stop-loss/take-profit orders
                self._check_stop_loss_orders()
                self._check_take_profit_orders()
                self._check_trailing_stops()
                
                # Update position P&L
                self.update_positions_pnl()
                
                # Sleep for 10 seconds before next check
                time.sleep(10)
                
            except Exception as e:
                print(f"❌ Error in automation worker: {e}")
                time.sleep(30)  # Wait longer on error
        
        print("🔄 Automation worker stopped")
    
    def _check_stop_loss_orders(self):
        """🛑 Check and execute stop-loss orders"""
        executed_orders = []
        
        for order in self.orders.values():
            if (order.order_type == OrderType.STOP and 
                order.status == OrderStatus.OPEN and 
                order.side == OrderSide.SELL):
                
                try:
                    current_price = self.get_current_price(order.coin_id)
                    
                    # Execute stop-loss if price hits stop price
                    if current_price <= order.stop_price:
                        print(f"🛑 Executing stop-loss: {order.coin_symbol} at ${current_price:.4f}")
                        self.execute_order(order, current_price)
                        executed_orders.append(order.order_id)
                        
                except Exception as e:
                    print(f"❌ Error checking stop-loss for {order.order_id}: {e}")
        
        return executed_orders
    
    def _check_take_profit_orders(self):
        """🎯 Check and execute take-profit orders"""
        executed_orders = []
        
        for order in self.orders.values():
            if (order.order_type == OrderType.LIMIT and 
                order.status == OrderStatus.OPEN and 
                order.side == OrderSide.SELL and
                'take-profit' in order.notes.lower()):
                
                try:
                    current_price = self.get_current_price(order.coin_id)
                    
                    # Execute take-profit if price reaches target
                    if current_price >= order.price:
                        print(f"🎯 Executing take-profit: {order.coin_symbol} at ${current_price:.4f}")
                        self.execute_order(order, current_price)
                        executed_orders.append(order.order_id)
                        
                except Exception as e:
                    print(f"❌ Error checking take-profit for {order.order_id}: {e}")
        
        return executed_orders
    
    def _check_trailing_stops(self):
        """📈 Check and update trailing stop orders"""
        updated_orders = []
        
        for order in self.orders.values():
            if (order.order_type == OrderType.TRAILING_STOP and 
                order.status == OrderStatus.OPEN):
                
                try:
                    current_price = self.get_current_price(order.coin_id)
                    
                    # Find corresponding position
                    position_id = f"{order.coin_symbol}_{order.portfolio_name}"
                    position = self.positions.get(position_id)
                    
                    if not position or position.trailing_stop_pct is None:
                        continue
                    
                    # Update trailing stop price
                    trail_pct = position.trailing_stop_pct / 100
                    
                    if position.side == 'long':
                        # For long positions, trail up with the price
                        if position.max_price is None or current_price > position.max_price:
                            position.max_price = current_price
                            
                        new_stop_price = position.max_price * (1 - trail_pct)
                        
                        # Update stop price if it's higher
                        if new_stop_price > order.stop_price:
                            print(f"📈 Updating trailing stop for {order.coin_symbol}: ${new_stop_price:.4f}")
                            order.stop_price = new_stop_price
                            order.updated_at = datetime.now().isoformat()
                            updated_orders.append(order.order_id)
                        
                        # Execute if current price hits stop
                        if current_price <= order.stop_price:
                            print(f"🛑 Executing trailing stop: {order.coin_symbol} at ${current_price:.4f}")
                            self.execute_order(order, current_price)
                
                except Exception as e:
                    print(f"❌ Error checking trailing stop for {order.order_id}: {e}")
        
        if updated_orders:
            self.save_trading_data()
        
        return updated_orders
    
    def set_position_automation(self, position_id: str, stop_loss_pct: float = None, 
                               take_profit_pct: float = None, trailing_stop_pct: float = None):
        """⚙️ Set automated stop-loss and take-profit for a position"""
        if position_id not in self.positions:
            print(f"❌ Position {position_id} not found")
            return False
        
        position = self.positions[position_id]
        current_price = position.current_price
        
        try:
            # Set stop-loss
            if stop_loss_pct:
                stop_price = current_price * (1 - stop_loss_pct / 100)
                stop_order_id = self.create_stop_loss_order(position_id, stop_price)
                if stop_order_id:
                    print(f"🛑 Stop-loss set at ${stop_price:.4f} ({stop_loss_pct}%)")
            
            # Set take-profit
            if take_profit_pct:
                target_price = current_price * (1 + take_profit_pct / 100)
                tp_order_id = self.create_take_profit_order(position_id, target_price)
                if tp_order_id:
                    print(f"🎯 Take-profit set at ${target_price:.4f} ({take_profit_pct}%)")
            
            # Set trailing stop
            if trailing_stop_pct:
                position.trailing_stop_pct = trailing_stop_pct
                position.max_price = current_price  # Initialize max price
                
                # Create trailing stop order
                trail_stop_price = current_price * (1 - trailing_stop_pct / 100)
                
                order_id = self.create_order(
                    coin_id=position.coin_id,
                    coin_symbol=position.coin_symbol,
                    order_type=OrderType.TRAILING_STOP,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    stop_price=trail_stop_price,
                    portfolio_name=position.portfolio_name,
                    notes=f"Trailing stop {trailing_stop_pct}% for position {position_id}"
                )
                
                if order_id:
                    print(f"📈 Trailing stop set at {trailing_stop_pct}%")
            
            self.save_trading_data()
            return True
            
        except Exception as e:
            print(f"❌ Error setting position automation: {e}")
            return False
    
    def get_automation_status(self) -> Dict[str, Any]:
        """📊 Get automation status and statistics"""
        active_stop_losses = len([
            order for order in self.orders.values() 
            if order.order_type == OrderType.STOP and order.status == OrderStatus.OPEN
        ])
        
        active_take_profits = len([
            order for order in self.orders.values() 
            if order.order_type == OrderType.LIMIT and order.status == OrderStatus.OPEN 
            and 'take-profit' in order.notes.lower()
        ])
        
        active_trailing_stops = len([
            order for order in self.orders.values() 
            if order.order_type == OrderType.TRAILING_STOP and order.status == OrderStatus.OPEN
        ])
        
        protected_positions = len([
            pos for pos in self.positions.values()
            if pos.status == PositionStatus.OPEN and 
            (pos.stop_loss_price or pos.take_profit_price or pos.trailing_stop_pct)
        ])
        
        return {
            'automation_active': self.automation_active,
            'active_stop_losses': active_stop_losses,
            'active_take_profits': active_take_profits,
            'active_trailing_stops': active_trailing_stops,
            'protected_positions': protected_positions,
            'total_positions': len(self.get_open_positions()),
            'protection_coverage': protected_positions / len(self.get_open_positions()) if self.get_open_positions() else 0
        }
    
    def start_realtime_tracking(self):
        """🔄 Start real-time position and P&L tracking"""
        if self.monitoring_active:
            print("⚠️ Real-time tracking already running")
            return False
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._realtime_worker, daemon=True)
        self.monitor_thread.start()
        
        print("🔄 Real-time tracking started")
        return True
    
    def stop_realtime_tracking(self):
        """⏹️ Stop real-time tracking"""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        print("⏹️ Real-time tracking stopped")
        return True
    
    def _realtime_worker(self):
        """🔄 Background worker for real-time tracking"""
        print("🔄 Real-time tracking worker started")
        
        while self.monitoring_active:
            try:
                # Update position prices and P&L
                self.update_realtime_pnl()
                
                # Check for significant P&L changes
                self.check_pnl_alerts()
                
                # Update price data
                self.update_price_data()
                
                # Sleep for 5 seconds before next update
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error in real-time tracking: {e}")
                time.sleep(10)  # Wait longer on error
        
        print("🔄 Real-time tracking worker stopped")
    
    def update_realtime_pnl(self):
        """💰 Update real-time P&L for all positions"""
        total_unrealized_pnl = 0
        total_realized_pnl = 0
        
        for position_id, position in self.positions.items():
            if position.status == PositionStatus.OPEN:
                try:
                    # Get current price
                    current_price = self.get_current_price(position.coin_id)
                    
                    if current_price > 0:
                        # Update position data
                        old_price = position.current_price
                        old_pnl = position.unrealized_pnl
                        
                        position.current_price = current_price
                        position.unrealized_pnl = position.quantity * (current_price - position.avg_entry_price)
                        
                        # Track P&L changes
                        pnl_change = position.unrealized_pnl - old_pnl
                        
                        # Store in realtime data
                        self.realtime_data['positions'][position_id] = {
                            'coin_symbol': position.coin_symbol,
                            'current_price': current_price,
                            'price_change': current_price - old_price,
                            'price_change_pct': ((current_price - old_price) / old_price * 100) if old_price > 0 else 0,
                            'unrealized_pnl': position.unrealized_pnl,
                            'pnl_change': pnl_change,
                            'quantity': position.quantity,
                            'avg_entry_price': position.avg_entry_price,
                            'last_update': datetime.now().isoformat()
                        }
                        
                        total_unrealized_pnl += position.unrealized_pnl
                        
                except Exception as e:
                    print(f"❌ Error updating P&L for {position.coin_symbol}: {e}")
        
        # Add total realized P&L
        for position in self.positions.values():
            total_realized_pnl += position.realized_pnl
        
        # Store P&L history
        pnl_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'position_count': len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        }
        
        self.realtime_data['pnl_history'].append(pnl_snapshot)
        
        # Keep only last 1000 records
        if len(self.realtime_data['pnl_history']) > 1000:
            self.realtime_data['pnl_history'] = self.realtime_data['pnl_history'][-1000:]
        
        return total_unrealized_pnl + total_realized_pnl
    
    def check_pnl_alerts(self):
        """🚨 Check for significant P&L changes and generate alerts"""
        if not self.realtime_data['pnl_history']:
            return
        
        current_pnl = self.realtime_data['pnl_history'][-1]['total_pnl']
        
        # Check for significant position changes
        for position_id, pos_data in self.realtime_data['positions'].items():
            # Alert for significant P&L change (>5% in 5 minutes)
            pnl_change_pct = abs(pos_data.get('pnl_change', 0)) / pos_data['quantity'] * 100 if pos_data['quantity'] > 0 else 0
            
            if pnl_change_pct > 5:  # 5% change threshold
                alert = {
                    'id': f"pnl_alert_{int(time.time())}",
                    'type': 'pnl_change',
                    'severity': 'high' if pnl_change_pct > 10 else 'medium',
                    'message': f"{pos_data['coin_symbol']}: {pnl_change_pct:.1f}% P&L change",
                    'position_id': position_id,
                    'timestamp': datetime.now().isoformat(),
                    'data': pos_data
                }
                
                self.realtime_data['alerts'].append(alert)
                print(f"🚨 P&L Alert: {alert['message']}")
        
        # Keep only last 100 alerts
        if len(self.realtime_data['alerts']) > 100:
            self.realtime_data['alerts'] = self.realtime_data['alerts'][-100:]
    
    def update_price_data(self):
        """📊 Update current price data for all tracked coins"""
        tracked_coins = set()
        
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                tracked_coins.add(position.coin_id)
        
        for coin_id in tracked_coins:
            try:
                current_price = self.get_current_price(coin_id)
                
                if current_price > 0:
                    old_data = self.realtime_data['price_updates'].get(coin_id, {})
                    old_price = old_data.get('price', current_price)
                    
                    self.realtime_data['price_updates'][coin_id] = {
                        'price': current_price,
                        'change': current_price - old_price,
                        'change_pct': ((current_price - old_price) / old_price * 100) if old_price > 0 else 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                print(f"❌ Error updating price for {coin_id}: {e}")
    
    def get_realtime_summary(self) -> Dict[str, Any]:
        """📊 Get real-time tracking summary"""
        if not self.realtime_data['pnl_history']:
            return {'message': 'No real-time data available'}
        
        latest_pnl = self.realtime_data['pnl_history'][-1]
        
        # Calculate P&L change from 1 hour ago
        hour_ago = datetime.now() - timedelta(hours=1)
        hour_ago_pnl = None
        
        for record in reversed(self.realtime_data['pnl_history']):
            record_time = datetime.fromisoformat(record['timestamp'].replace('Z', ''))
            if record_time <= hour_ago:
                hour_ago_pnl = record['total_pnl']
                break
        
        pnl_change_1h = (latest_pnl['total_pnl'] - hour_ago_pnl) if hour_ago_pnl else 0
        
        # Count positions by performance
        winning_positions = 0
        losing_positions = 0
        
        for pos_data in self.realtime_data['positions'].values():
            if pos_data['unrealized_pnl'] > 0:
                winning_positions += 1
            elif pos_data['unrealized_pnl'] < 0:
                losing_positions += 1
        
        return {
            'tracking_active': self.monitoring_active,
            'last_update': latest_pnl['timestamp'],
            'total_pnl': latest_pnl['total_pnl'],
            'total_unrealized_pnl': latest_pnl['total_unrealized_pnl'],
            'total_realized_pnl': latest_pnl['total_realized_pnl'],
            'pnl_change_1h': pnl_change_1h,
            'active_positions': latest_pnl['position_count'],
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'recent_alerts': len([a for a in self.realtime_data['alerts'] 
                                if datetime.fromisoformat(a['timestamp'].replace('Z', '')) >= datetime.now() - timedelta(hours=1)]),
            'position_updates': len(self.realtime_data['positions']),
            'tracked_coins': len(self.realtime_data['price_updates'])
        }
    
    def get_position_realtime_data(self, position_id: str = None) -> Dict[str, Any]:
        """📊 Get real-time data for specific position or all positions"""
        if position_id:
            return self.realtime_data['positions'].get(position_id, {'error': 'Position not found'})
        
        return {
            'positions': self.realtime_data['positions'],
            'price_updates': self.realtime_data['price_updates'],
            'last_update': datetime.now().isoformat()
        }
    
    def get_pnl_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """📈 Get P&L history for charting"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return [
            record for record in self.realtime_data['pnl_history']
            if datetime.fromisoformat(record['timestamp'].replace('Z', '')) >= cutoff_time
        ]
    
    def get_recent_alerts(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """🚨 Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return [
            alert for alert in self.realtime_data['alerts']
            if datetime.fromisoformat(alert['timestamp'].replace('Z', '')) >= cutoff_time
        ]

# Test der Trading Interface
def main():
    """💼 Test des Professional Trading Interface"""
    interface = TradingInterface()
    
    print("💼 Professional Trading Interface - Demo")
    print("=" * 50)
    
    # Test market buy order
    try:
        order_id = interface.create_order(
            coin_id='bitcoin',
            coin_symbol='BTC',
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.001,
            portfolio_name="Main Portfolio",
            notes="Demo buy order"
        )
        
        print(f"✅ Market buy order created: {order_id}")
        
        # Update positions
        interface.update_positions_pnl()
        
        # Get risk metrics
        risk_metrics = interface.calculate_risk_metrics("Main Portfolio")
        print(f"\n⚠️ Risk Metrics:")
        print(f"   Total Exposure: ${risk_metrics.total_exposure:.2f}")
        print(f"   Open Positions: {risk_metrics.open_positions}")
        print(f"   Risk Score: {risk_metrics.risk_score:.1f}/100")
        print(f"   Unrealized P&L: ${risk_metrics.total_unrealized_pnl:.2f}")
        
        # Get trading summary
        summary = interface.get_trading_summary(7)
        print(f"\n📊 Trading Summary (7 days):")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Total Volume: ${summary.get('total_volume_usd', 0):.2f}")
        print(f"   Total Fees: ${summary.get('total_fees', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Demo trade failed: {e}")

if __name__ == "__main__":
    import os
    main()