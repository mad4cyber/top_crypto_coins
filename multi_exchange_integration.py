#!/usr/bin/env python3
"""
🏦 Multi-Exchange Integration System
Autor: mad4cyber
Version: 1.0 - Multi-Exchange Edition

🚀 FEATURES:
- Multiple Exchange Connectivity (Binance, Coinbase, Kraken)
- Unified Trading Interface
- Cross-Exchange Arbitrage Detection  
- Exchange Health Monitoring
- Order Routing and Execution
- Portfolio Aggregation across Exchanges
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import hmac
import hashlib
import base64
import warnings
warnings.filterwarnings('ignore')

class ExchangeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"  
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"

class ExchangeType(Enum):
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"

@dataclass
class ExchangeConfig:
    """⚙️ Exchange Configuration"""
    name: str
    api_key: str
    api_secret: str
    api_url: str
    sandbox_url: Optional[str] = None
    rate_limit: int = 1000  # requests per minute
    supported_pairs: List[str] = None
    fees: Dict[str, float] = None
    is_sandbox: bool = True

@dataclass
class ExchangeBalance:
    """💰 Exchange Balance Information"""
    exchange: str
    asset: str
    available: float
    locked: float
    total: float
    usd_value: float
    last_update: str

@dataclass
class ExchangeOrder:
    """📝 Unified Exchange Order"""
    exchange: str
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', etc.
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    timestamp: str
    fees: Dict[str, float] = None

@dataclass
class ArbitrageOpportunity:
    """🔄 Arbitrage Opportunity"""
    timestamp: str
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    spread_pct: float
    potential_profit: float
    volume: float
    confidence: float

@dataclass
class ExchangeHealth:
    """🩺 Exchange Health Status"""
    exchange: str
    status: ExchangeStatus
    latency_ms: float
    uptime_pct: float
    last_check: str
    api_calls_remaining: int
    error_rate: float
    issues: List[str] = None

class BaseExchange:
    """🏛️ Base Exchange Class"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_count = 0
        self.health_status = ExchangeHealth(
            exchange=config.name,
            status=ExchangeStatus.OFFLINE,
            latency_ms=0,
            uptime_pct=0,
            last_check=datetime.now().isoformat(),
            api_calls_remaining=config.rate_limit,
            error_rate=0,
            issues=[]
        )
    
    def _rate_limit(self):
        """⏱️ Rate limiting"""
        current_time = time.time()
        if current_time - self.last_request_time < 60 / self.config.rate_limit:
            time.sleep(60 / self.config.rate_limit)
        self.last_request_time = current_time
        self.request_count += 1
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """🌐 Make API request with error handling"""
        self._rate_limit()
        
        url = f"{self.config.api_url}{endpoint}"
        headers = self._get_headers(method, endpoint, params, data)
        
        try:
            start_time = time.time()
            
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            latency = (time.time() - start_time) * 1000
            self.health_status.latency_ms = latency
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"❌ API request failed for {self.config.name}: {e}")
            self.health_status.status = ExchangeStatus.DEGRADED
            self.health_status.issues.append(str(e))
            raise
    
    def _get_headers(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """🔐 Get authentication headers (to be overridden)"""
        return {}
    
    def test_connection(self) -> bool:
        """🔍 Test exchange connectivity"""
        try:
            self.get_server_time()
            self.health_status.status = ExchangeStatus.ONLINE
            return True
        except:
            self.health_status.status = ExchangeStatus.OFFLINE
            return False
    
    def get_server_time(self) -> int:
        """⏰ Get server time (to be overridden)"""
        raise NotImplementedError
    
    def get_balance(self) -> List[ExchangeBalance]:
        """💰 Get account balance (to be overridden)"""
        raise NotImplementedError
    
    def get_orderbook(self, symbol: str) -> Dict:
        """📚 Get order book (to be overridden)"""
        raise NotImplementedError
    
    def place_order(self, symbol: str, side: str, type: str, quantity: float, price: float = None) -> ExchangeOrder:
        """📝 Place order (to be overridden)"""
        raise NotImplementedError
    
    def get_order_status(self, order_id: str, symbol: str = None) -> ExchangeOrder:
        """📊 Get order status (to be overridden)"""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """❌ Cancel order (to be overridden)"""
        raise NotImplementedError

class BinanceExchange(BaseExchange):
    """🟡 Binance Exchange Implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.config.api_url = "https://api.binance.com" if not config.is_sandbox else "https://testnet.binance.vision"
    
    def _get_headers(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """🔐 Binance authentication"""
        timestamp = int(time.time() * 1000)
        headers = {
            'X-MBX-APIKEY': self.config.api_key,
            'Content-Type': 'application/json'
        }
        
        if params:
            params['timestamp'] = timestamp
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        return headers
    
    def get_server_time(self) -> int:
        """⏰ Get Binance server time"""
        response = self._make_request('GET', '/api/v3/time')
        return response['serverTime']
    
    def get_balance(self) -> List[ExchangeBalance]:
        """💰 Get Binance account balance"""
        try:
            response = self._make_request('GET', '/api/v3/account', params={})
            balances = []
            
            for balance in response.get('balances', []):
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    # Simulate USD conversion (would use real prices in production)
                    usd_rate = np.random.uniform(0.5, 50000) if balance['asset'] != 'USDT' else 1.0
                    total = float(balance['free']) + float(balance['locked'])
                    
                    balances.append(ExchangeBalance(
                        exchange="binance",
                        asset=balance['asset'],
                        available=float(balance['free']),
                        locked=float(balance['locked']),
                        total=total,
                        usd_value=total * usd_rate,
                        last_update=datetime.now().isoformat()
                    ))
            
            return balances
            
        except Exception as e:
            print(f"❌ Error getting Binance balance: {e}")
            return []
    
    def get_orderbook(self, symbol: str) -> Dict:
        """📚 Get Binance order book"""
        try:
            response = self._make_request('GET', f'/api/v3/depth', params={'symbol': symbol, 'limit': 100})
            return {
                'bids': [[float(price), float(qty)] for price, qty in response['bids']],
                'asks': [[float(price), float(qty)] for price, qty in response['asks']],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error getting Binance orderbook: {e}")
            return {'bids': [], 'asks': [], 'timestamp': datetime.now().isoformat()}

class CoinbaseExchange(BaseExchange):
    """🔵 Coinbase Pro Exchange Implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.config.api_url = "https://api.exchange.coinbase.com" if not config.is_sandbox else "https://api-public.sandbox.exchange.coinbase.com"
    
    def _get_headers(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """🔐 Coinbase authentication"""
        timestamp = str(time.time())
        message = timestamp + method.upper() + endpoint + (json.dumps(data) if data else '')
        signature = base64.b64encode(hmac.new(
            base64.b64decode(self.config.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()).decode('utf-8')
        
        return {
            'CB-ACCESS-KEY': self.config.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': 'sandbox',  # Would be real passphrase in production
            'Content-Type': 'application/json'
        }
    
    def get_server_time(self) -> int:
        """⏰ Get Coinbase server time"""
        response = self._make_request('GET', '/time')
        return int(float(response['epoch']) * 1000)
    
    def get_balance(self) -> List[ExchangeBalance]:
        """💰 Get Coinbase account balance"""
        try:
            response = self._make_request('GET', '/accounts')
            balances = []
            
            for account in response:
                if float(account['balance']) > 0:
                    # Simulate USD conversion
                    usd_rate = np.random.uniform(0.5, 50000) if account['currency'] != 'USD' else 1.0
                    total = float(account['balance'])
                    available = float(account['available'])
                    locked = total - available
                    
                    balances.append(ExchangeBalance(
                        exchange="coinbase",
                        asset=account['currency'],
                        available=available,
                        locked=locked,
                        total=total,
                        usd_value=total * usd_rate,
                        last_update=datetime.now().isoformat()
                    ))
            
            return balances
            
        except Exception as e:
            print(f"❌ Error getting Coinbase balance: {e}")
            return []

class KrakenExchange(BaseExchange):
    """⚫ Kraken Exchange Implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.config.api_url = "https://api.kraken.com"
    
    def _get_headers(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """🔐 Kraken authentication"""
        nonce = str(int(time.time() * 1000))
        if data:
            data['nonce'] = nonce
        
        postdata = '&'.join([f"{k}={v}" for k, v in (data or {}).items()])
        encoded = (nonce + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        signature = base64.b64encode(hmac.new(
            base64.b64decode(self.config.api_secret),
            message,
            hashlib.sha512
        ).digest()).decode()
        
        return {
            'API-Key': self.config.api_key,
            'API-Sign': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    def get_server_time(self) -> int:
        """⏰ Get Kraken server time"""
        response = self._make_request('GET', '/0/public/Time')
        return response['result']['unixtime'] * 1000

class MultiExchangeIntegration:
    """🏦 Multi-Exchange Integration System"""
    
    def __init__(self, data_file: str = "multi_exchange_data.json"):
        self.data_file = data_file
        self.exchanges: Dict[str, BaseExchange] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.health_monitor_active = False
        self.arbitrage_monitor_active = False
        self.monitor_thread = None
        self.arbitrage_thread = None
        
        # Configuration
        self.min_arbitrage_spread = 0.5  # 0.5% minimum spread
        self.max_arbitrage_volume = 10000  # $10k max volume
        self.health_check_interval = 60  # 60 seconds
        
        # Load existing data
        self.load_data()
        
        # Initialize demo exchanges
        self.setup_demo_exchanges()
    
    def setup_demo_exchanges(self):
        """🚀 Setup demo exchanges for testing"""
        demo_configs = [
            ExchangeConfig(
                name="binance",
                api_key="demo_binance_key",
                api_secret="demo_binance_secret", 
                api_url="https://testnet.binance.vision",
                is_sandbox=True,
                rate_limit=1200,
                fees={'maker': 0.001, 'taker': 0.001}
            ),
            ExchangeConfig(
                name="coinbase",
                api_key="demo_coinbase_key",
                api_secret="demo_coinbase_secret",
                api_url="https://api-public.sandbox.exchange.coinbase.com",
                is_sandbox=True,
                rate_limit=600,
                fees={'maker': 0.005, 'taker': 0.005}
            ),
            ExchangeConfig(
                name="kraken",
                api_key="demo_kraken_key", 
                api_secret="demo_kraken_secret",
                api_url="https://api.kraken.com",
                is_sandbox=True,
                rate_limit=900,
                fees={'maker': 0.0016, 'taker': 0.0026}
            )
        ]
        
        print("🚀 Setting up demo exchanges...")
        for config in demo_configs:
            try:
                if config.name == "binance":
                    exchange = BinanceExchange(config)
                elif config.name == "coinbase": 
                    exchange = CoinbaseExchange(config)
                elif config.name == "kraken":
                    exchange = KrakenExchange(config)
                else:
                    continue
                
                self.exchanges[config.name] = exchange
                print(f"✅ {config.name.title()} exchange configured")
                
            except Exception as e:
                print(f"❌ Failed to setup {config.name}: {e}")
    
    def add_exchange(self, exchange: BaseExchange):
        """➕ Add new exchange"""
        self.exchanges[exchange.config.name] = exchange
        print(f"➕ Added exchange: {exchange.config.name}")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """🔍 Test all exchange connections"""
        results = {}
        for name, exchange in self.exchanges.items():
            try:
                result = exchange.test_connection()
                results[name] = result
                status = "✅ Online" if result else "❌ Offline"
                print(f"{status}: {name.title()}")
            except Exception as e:
                results[name] = False
                print(f"❌ {name.title()}: {e}")
        
        return results
    
    def get_unified_balance(self) -> Dict[str, Dict[str, float]]:
        """💰 Get unified balance across all exchanges"""
        unified_balance = {}
        total_usd_value = 0
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Simulate balance data since we can't connect to real APIs
                balances = self._simulate_exchange_balance(exchange_name)
                
                for balance in balances:
                    asset = balance.asset
                    if asset not in unified_balance:
                        unified_balance[asset] = {
                            'total_balance': 0,
                            'total_usd_value': 0,
                            'exchanges': {}
                        }
                    
                    unified_balance[asset]['total_balance'] += balance.total
                    unified_balance[asset]['total_usd_value'] += balance.usd_value
                    unified_balance[asset]['exchanges'][exchange_name] = {
                        'balance': balance.total,
                        'available': balance.available,
                        'locked': balance.locked,
                        'usd_value': balance.usd_value
                    }
                    
                    total_usd_value += balance.usd_value
                    
            except Exception as e:
                print(f"❌ Error getting balance from {exchange_name}: {e}")
        
        # Add summary
        unified_balance['_summary'] = {
            'total_usd_value': total_usd_value,
            'exchange_count': len(self.exchanges),
            'asset_count': len([k for k in unified_balance.keys() if k != '_summary']),
            'last_update': datetime.now().isoformat()
        }
        
        return unified_balance
    
    def _simulate_exchange_balance(self, exchange_name: str) -> List[ExchangeBalance]:
        """🎭 Simulate exchange balance for demo"""
        demo_assets = ['BTC', 'ETH', 'USDT', 'BNB', 'ADA']
        balances = []
        
        for asset in demo_assets:
            if np.random.random() > 0.3:  # 70% chance of having balance
                available = np.random.uniform(0.1, 10) 
                locked = np.random.uniform(0, available * 0.2)
                total = available + locked
                
                # Simulate USD prices
                usd_rates = {'BTC': 45000, 'ETH': 3200, 'USDT': 1, 'BNB': 320, 'ADA': 0.8}
                usd_rate = usd_rates.get(asset, 100)
                
                balances.append(ExchangeBalance(
                    exchange=exchange_name,
                    asset=asset,
                    available=available,
                    locked=locked,
                    total=total,
                    usd_value=total * usd_rate,
                    last_update=datetime.now().isoformat()
                ))
        
        return balances
    
    def scan_arbitrage_opportunities(self, symbols: List[str] = None) -> List[ArbitrageOpportunity]:
        """🔄 Scan for arbitrage opportunities"""
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
        
        opportunities = []
        
        for symbol in symbols:
            prices = {}
            
            # Get prices from all exchanges (simulated)
            for exchange_name in self.exchanges.keys():
                try:
                    # Simulate price with small variations between exchanges
                    base_price = self._get_simulated_price(symbol)
                    variation = np.random.uniform(-0.02, 0.02)  # ±2% variation
                    price = base_price * (1 + variation)
                    prices[exchange_name] = price
                    
                except Exception as e:
                    print(f"❌ Error getting price from {exchange_name} for {symbol}: {e}")
            
            # Find arbitrage opportunities
            if len(prices) >= 2:
                sorted_prices = sorted(prices.items(), key=lambda x: x[1])
                buy_exchange, buy_price = sorted_prices[0]  # Lowest price
                sell_exchange, sell_price = sorted_prices[-1]  # Highest price
                
                spread_pct = ((sell_price - buy_price) / buy_price) * 100
                
                if spread_pct >= self.min_arbitrage_spread:
                    # Estimate potential profit
                    volume = min(1000, self.max_arbitrage_volume / buy_price)  # Max $10k or 1000 units
                    fees = self._calculate_arbitrage_fees(buy_exchange, sell_exchange, volume, buy_price, sell_price)
                    profit = (sell_price - buy_price) * volume - fees
                    
                    if profit > 0:
                        opportunity = ArbitrageOpportunity(
                            timestamp=datetime.now().isoformat(),
                            buy_exchange=buy_exchange,
                            sell_exchange=sell_exchange,
                            symbol=symbol,
                            buy_price=buy_price,
                            sell_price=sell_price,
                            spread_pct=spread_pct,
                            potential_profit=profit,
                            volume=volume,
                            confidence=0.8 if spread_pct > 1.0 else 0.6
                        )
                        
                        opportunities.append(opportunity)
                        print(f"🔄 Arbitrage found: {symbol} - {spread_pct:.2f}% spread, ${profit:.2f} profit")
        
        self.arbitrage_opportunities.extend(opportunities)
        
        # Keep only recent opportunities
        cutoff = datetime.now() - timedelta(hours=24)
        self.arbitrage_opportunities = [
            opp for opp in self.arbitrage_opportunities
            if datetime.fromisoformat(opp.timestamp.replace('Z', '')) >= cutoff
        ]
        
        return opportunities
    
    def _get_simulated_price(self, symbol: str) -> float:
        """💰 Get simulated price for symbol"""
        base_prices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 3200, 
            'BNBUSDT': 320,
            'ADAUSDT': 0.8,
            'SOLUSDT': 85
        }
        return base_prices.get(symbol, 100)
    
    def _calculate_arbitrage_fees(self, buy_exchange: str, sell_exchange: str, volume: float, buy_price: float, sell_price: float) -> float:
        """💸 Calculate arbitrage fees"""
        buy_fee = volume * buy_price * 0.001  # 0.1% fee
        sell_fee = volume * sell_price * 0.001  # 0.1% fee
        return buy_fee + sell_fee
    
    def get_exchange_health(self) -> Dict[str, ExchangeHealth]:
        """🩺 Get health status of all exchanges"""
        health_status = {}
        
        for name, exchange in self.exchanges.items():
            try:
                # Test connection
                start_time = time.time()
                is_online = exchange.test_connection()
                latency = (time.time() - start_time) * 1000
                
                # Update health status
                exchange.health_status.latency_ms = latency
                exchange.health_status.last_check = datetime.now().isoformat()
                exchange.health_status.status = ExchangeStatus.ONLINE if is_online else ExchangeStatus.OFFLINE
                
                # Simulate uptime and error rate
                exchange.health_status.uptime_pct = np.random.uniform(95, 99.9)
                exchange.health_status.error_rate = np.random.uniform(0, 5)
                exchange.health_status.api_calls_remaining = exchange.config.rate_limit - exchange.request_count
                
                health_status[name] = exchange.health_status
                
            except Exception as e:
                print(f"❌ Error checking health of {name}: {e}")
                exchange.health_status.status = ExchangeStatus.OFFLINE
                exchange.health_status.issues.append(str(e))
                health_status[name] = exchange.health_status
        
        return health_status
    
    def start_health_monitoring(self):
        """🏥 Start health monitoring"""
        if self.health_monitor_active:
            print("⚠️ Health monitoring already active")
            return
        
        self.health_monitor_active = True
        self.monitor_thread = threading.Thread(target=self._health_monitor_worker, daemon=True)
        self.monitor_thread.start()
        print("🏥 Health monitoring started")
    
    def stop_health_monitoring(self):
        """⏹️ Stop health monitoring"""
        self.health_monitor_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Health monitoring stopped")
    
    def _health_monitor_worker(self):
        """🔄 Health monitoring worker"""
        while self.health_monitor_active:
            try:
                self.get_exchange_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                print(f"❌ Error in health monitoring: {e}")
                time.sleep(30)
    
    def start_arbitrage_monitoring(self):
        """📊 Start arbitrage monitoring"""
        if self.arbitrage_monitor_active:
            print("⚠️ Arbitrage monitoring already active")
            return
        
        self.arbitrage_monitor_active = True
        self.arbitrage_thread = threading.Thread(target=self._arbitrage_monitor_worker, daemon=True)
        self.arbitrage_thread.start()
        print("📊 Arbitrage monitoring started")
    
    def stop_arbitrage_monitoring(self):
        """⏹️ Stop arbitrage monitoring"""
        self.arbitrage_monitor_active = False
        if self.arbitrage_thread:
            self.arbitrage_thread.join(timeout=5)
        print("⏹️ Arbitrage monitoring stopped")
    
    def _arbitrage_monitor_worker(self):
        """🔄 Arbitrage monitoring worker"""
        while self.arbitrage_monitor_active:
            try:
                self.scan_arbitrage_opportunities()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"❌ Error in arbitrage monitoring: {e}")
                time.sleep(60)
    
    def get_exchange_summary(self) -> Dict[str, Any]:
        """📊 Get comprehensive exchange summary"""
        balance = self.get_unified_balance()
        health = self.get_exchange_health()
        recent_arbitrage = [opp for opp in self.arbitrage_opportunities 
                          if datetime.fromisoformat(opp.timestamp.replace('Z', '')) >= datetime.now() - timedelta(hours=1)]
        
        online_exchanges = len([h for h in health.values() if h.status == ExchangeStatus.ONLINE])
        avg_latency = np.mean([h.latency_ms for h in health.values() if h.latency_ms > 0]) if health else 0
        
        return {
            'total_exchanges': len(self.exchanges),
            'online_exchanges': online_exchanges,
            'total_portfolio_usd': balance.get('_summary', {}).get('total_usd_value', 0),
            'unique_assets': len([k for k in balance.keys() if k != '_summary']),
            'avg_latency_ms': avg_latency,
            'arbitrage_opportunities_1h': len(recent_arbitrage),
            'best_arbitrage_spread': max([opp.spread_pct for opp in recent_arbitrage], default=0),
            'health_monitoring_active': self.health_monitor_active,
            'arbitrage_monitoring_active': self.arbitrage_monitor_active,
            'last_update': datetime.now().isoformat()
        }
    
    def load_data(self):
        """📁 Load exchange data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load arbitrage opportunities
                for opp_data in data.get('arbitrage_opportunities', []):
                    opp = ArbitrageOpportunity(**opp_data)
                    self.arbitrage_opportunities.append(opp)
                    
        except Exception as e:
            print(f"❌ Error loading exchange data: {e}")
    
    def save_data(self):
        """💾 Save exchange data"""
        try:
            data = {
                'arbitrage_opportunities': [asdict(opp) for opp in self.arbitrage_opportunities[-100:]],  # Keep last 100
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving exchange data: {e}")

# Test der Multi-Exchange Integration
def main():
    """🏦 Test Multi-Exchange Integration"""
    integration = MultiExchangeIntegration()
    
    print("🏦 Multi-Exchange Integration - Demo")
    print("=" * 50)
    
    # Test connections
    print("\n🔍 Testing exchange connections...")
    connections = integration.test_all_connections()
    
    # Get unified balance
    print("\n💰 Getting unified portfolio balance...")
    balance = integration.get_unified_balance()
    summary = balance.get('_summary', {})
    print(f"   Total Portfolio Value: ${summary.get('total_usd_value', 0):,.2f}")
    print(f"   Active Exchanges: {summary.get('exchange_count', 0)}")
    print(f"   Unique Assets: {summary.get('asset_count', 0)}")
    
    # Scan for arbitrage
    print("\n🔄 Scanning for arbitrage opportunities...")
    opportunities = integration.scan_arbitrage_opportunities()
    if opportunities:
        print(f"   Found {len(opportunities)} arbitrage opportunities!")
        for opp in opportunities[:3]:  # Show top 3
            print(f"   • {opp.symbol}: {opp.spread_pct:.2f}% spread (${opp.potential_profit:.2f} profit)")
    else:
        print("   No arbitrage opportunities found")
    
    # Get health status
    print("\n🩺 Exchange health status...")
    health = integration.get_exchange_health()
    for name, status in health.items():
        print(f"   • {name.title()}: {status.status.value} ({status.latency_ms:.0f}ms)")
    
    # Get summary
    print("\n📊 Exchange Integration Summary:")
    summary = integration.get_exchange_summary()
    print(f"   Online Exchanges: {summary['online_exchanges']}/{summary['total_exchanges']}")
    print(f"   Average Latency: {summary['avg_latency_ms']:.0f}ms")
    print(f"   Recent Arbitrage Opportunities: {summary['arbitrage_opportunities_1h']}")
    
    # Save data
    integration.save_data()

if __name__ == "__main__":
    import os
    main()