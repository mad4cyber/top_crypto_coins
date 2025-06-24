#!/usr/bin/env python3
"""
Erweiterte Kryptowährungs-Analyse mit verbesserter Architektur
Autor: mad4cyber
Version: 2.0
"""

import argparse
import logging
import os
import json
import yaml
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tabulate import tabulate
from pycoingecko import CoinGeckoAPI

# Konfiguration für bessere Farbdarstellung
try:
    from colorama import init, Fore, Style
    init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


@dataclass
class CryptoConfig:
    """Konfigurationsklasse für die Kryptowährungs-Analyse"""
    currency: str = "eur"
    language: str = "de"
    num_coins: int = 10
    sort_by: str = "market_cap"
    table_format: str = "grid"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'CryptoConfig':
        """Lade Konfiguration aus YAML-Datei"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                defaults = config_data.get('defaults', {})
                return cls(**defaults)
        except (FileNotFoundError, yaml.YAMLError):
            return cls()


class CryptoLogger:
    """Verbessertes Logging-System"""
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger('crypto_analyzer')
        self._setup_logger(config_path)
    
    def _setup_logger(self, config_path: str):
        """Logger konfigurieren"""
        log_level = logging.INFO
        log_file = "crypto_analyzer.log"
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    log_config = config.get('logging', {})
                    log_level = getattr(logging, log_config.get('level', 'INFO'))
                    log_file = log_config.get('file', log_file)
            except Exception:
                pass
        
        # Logger konfigurieren
        self.logger.setLevel(log_level)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)


class PortfolioManager:
    """Portfolio-Management für persönliche Kryptowährungen"""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """Portfolio aus Datei laden"""
        try:
            with open(self.portfolio_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {"holdings": {}, "alerts": []}
    
    def save_portfolio(self):
        """Portfolio in Datei speichern"""
        with open(self.portfolio_file, 'w') as file:
            json.dump(self.portfolio, file, indent=2)
    
    def add_holding(self, symbol: str, amount: float, purchase_price: float = None):
        """Kryptowährung zum Portfolio hinzufügen"""
        if symbol not in self.portfolio["holdings"]:
            self.portfolio["holdings"][symbol] = []
        
        self.portfolio["holdings"][symbol].append({
            "amount": amount,
            "purchase_price": purchase_price,
            "date": datetime.now().isoformat()
        })
        self.save_portfolio()
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Aktuellen Portfolio-Wert berechnen"""
        portfolio_value = {}
        total_value = 0
        total_invested = 0
        
        for symbol, holdings in self.portfolio["holdings"].items():
            if symbol.upper() in current_prices:
                current_price = current_prices[symbol.upper()]
                symbol_value = 0
                symbol_invested = 0
                total_amount = 0
                
                for holding in holdings:
                    amount = holding["amount"]
                    purchase_price = holding.get("purchase_price", 0)
                    
                    total_amount += amount
                    symbol_value += amount * current_price
                    symbol_invested += amount * purchase_price if purchase_price else 0
                
                portfolio_value[symbol] = {
                    "amount": total_amount,
                    "current_value": symbol_value,
                    "invested": symbol_invested,
                    "profit_loss": symbol_value - symbol_invested if symbol_invested > 0 else 0,
                    "current_price": current_price
                }
                
                total_value += symbol_value
                total_invested += symbol_invested
        
        portfolio_value["_total"] = {
            "current_value": total_value,
            "invested": total_invested,
            "profit_loss": total_value - total_invested if total_invested > 0 else 0
        }
        
        return portfolio_value


class CryptoAnalyzer:
    """Hauptklasse für die Kryptowährungs-Analyse"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = CryptoConfig.from_yaml(config_path)
        self.logger = CryptoLogger(config_path)
        self.portfolio = PortfolioManager()
        self.cg = CoinGeckoAPI()
        self.cache = {}
        self.cache_ttl = 300  # 5 Minuten
        
        # Sprache aus Umgebungsvariable berücksichtigen
        self.language = os.environ.get('CRYPTO_LANG', self.config.language)
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Daten aus Cache abrufen"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Daten in Cache speichern"""
        self.cache[key] = (data, time.time())
    
    def get_crypto_data(self, num: int = None, currency: str = None, 
                       sort_by: str = None, show_volume: bool = False, 
                       show_supply: bool = False) -> pd.DataFrame:
        """Kryptowährungsdaten mit Caching abrufen"""
        num = num or self.config.num_coins
        currency = currency or self.config.currency
        sort_by = sort_by or self.config.sort_by
        
        cache_key = f"crypto_data_{num}_{currency}_{sort_by}_{show_volume}_{show_supply}"
        
        # Cache prüfen
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            self.logger.info(f"Verwende gecachte Daten für {cache_key}")
            return cached_data
        
        try:
            self.logger.info(f"Lade {num} Kryptowährungen von CoinGecko API")
            coins = self.cg.get_coins_markets(
                vs_currency=currency, 
                order=sort_by, 
                per_page=num,
                page=1, 
                sparkline=False, 
                price_change_percentage='24h,7d,30d'
            )
            
            df = pd.DataFrame(coins)
            
            # Basis-Spalten
            base_columns = [
                'id', 'symbol', 'name', 'market_cap_rank', 'current_price', 
                'market_cap', 'price_change_percentage_24h_in_currency',
                'price_change_percentage_7d_in_currency',
                'price_change_percentage_30d_in_currency'
            ]
            
            # Zusätzliche Spalten
            if show_volume:
                base_columns.append('total_volume')
            if show_supply:
                base_columns.extend(['circulating_supply', 'max_supply'])
            
            df = df[base_columns]
            
            # Deutsche Spaltennamen
            column_names = [
                'id', 'symbol', 'name', 'rang', 'preis', 'marktkapitalisierung',
                '24h_änderung', '7d_änderung', '30d_änderung'
            ]
            
            if show_volume:
                column_names.append('handelsvolumen')
            if show_supply:
                column_names.extend(['umlaufmenge', 'max_menge'])
            
            df.columns = column_names
            df['symbol'] = df['symbol'].str.upper()
            
            # In Cache speichern
            self._set_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Daten: {e}"
            self.logger.error(error_msg)
            if self.language == "de":
                print(f"❌ {error_msg}")
            else:
                print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def format_currency(self, value: float, currency: str = None) -> str:
        """Währung formatieren"""
        currency = currency or self.config.currency
        
        currency_symbols = {
            "eur": "€", "usd": "$", "gbp": "£", "chf": "CHF"
        }
        
        symbol = currency_symbols.get(currency.lower(), currency.upper())
        
        if value >= 1e9:
            return f"{symbol}{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{symbol}{value/1e6:.2f}M"
        elif value >= 1000:
            return f"{symbol}{value:,.0f}"
        else:
            return f"{symbol}{value:.2f}"
    
    def colorize_change(self, change: float) -> str:
        """Preisänderung mit Farben formatieren"""
        if not COLORS_AVAILABLE:
            return f"{change:+.2f}%"
        
        if change > 0:
            return f"{Fore.GREEN}{change:+.2f}%{Style.RESET_ALL}"
        elif change < 0:
            return f"{Fore.RED}{change:+.2f}%{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{change:+.2f}%{Style.RESET_ALL}"
    
    def display_portfolio(self):
        """Portfolio anzeigen"""
        if not self.portfolio.portfolio["holdings"]:
            msg = "📝 Kein Portfolio gefunden. Füge Coins hinzu mit --add-to-portfolio" if self.language == "de" else "📝 No portfolio found. Add coins with --add-to-portfolio"
            print(msg)
            return
        
        # Aktuelle Preise für Portfolio-Coins abrufen
        symbols = list(self.portfolio.portfolio["holdings"].keys())
        df = self.get_crypto_data(num=250)  # Mehr Coins für Portfolio-Matching
        
        current_prices = {}
        for _, row in df.iterrows():
            if row['symbol'] in [s.upper() for s in symbols]:
                current_prices[row['symbol']] = row['preis']
        
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        
        # Portfolio-Tabelle erstellen
        portfolio_data = []
        for symbol, data in portfolio_value.items():
            if symbol != "_total":
                profit_loss = data["profit_loss"]
                profit_loss_pct = (profit_loss / data["invested"] * 100) if data["invested"] > 0 else 0
                
                portfolio_data.append([
                    symbol.upper(),
                    f"{data['amount']:.4f}",
                    self.format_currency(data["current_price"]),
                    self.format_currency(data["current_value"]),
                    self.format_currency(data["invested"]) if data["invested"] > 0 else "N/A",
                    self.colorize_change(profit_loss_pct) if data["invested"] > 0 else "N/A",
                    self.format_currency(profit_loss) if data["invested"] > 0 else "N/A"
                ])
        
        headers = ["Symbol", "Menge", "Preis", "Wert", "Investiert", "Gewinn/Verlust %", "Gewinn/Verlust"]
        if self.language == "en":
            headers = ["Symbol", "Amount", "Price", "Value", "Invested", "Profit/Loss %", "Profit/Loss"]
        
        print("\n" + "="*100)
        title = "📊 Portfolio Übersicht" if self.language == "de" else "📊 Portfolio Overview"
        print(f"{title:^100}")
        print("="*100)
        
        print(tabulate(portfolio_data, headers=headers, tablefmt="grid"))
        
        # Gesamt-Statistiken
        total = portfolio_value["_total"]
        if total["invested"] > 0:
            total_pct = (total["profit_loss"] / total["invested"]) * 100
            print(f"\n💰 Gesamtwert: {self.format_currency(total['current_value'])}")
            print(f"💸 Investiert: {self.format_currency(total['invested'])}")
            print(f"📈 Gewinn/Verlust: {self.format_currency(total['profit_loss'])} ({total_pct:+.2f}%)")


def create_enhanced_cli():
    """Erweiterte CLI mit neuen Features"""
    parser = argparse.ArgumentParser(
        description="Erweiterte Kryptowährungs-Analyse v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s -n 5 --compact                        # Top 5 Coins kompakt
  %(prog)s --portfolio                           # Portfolio anzeigen
  %(prog)s --add-to-portfolio BTC 0.1 45000     # Bitcoin zum Portfolio hinzufügen
  %(prog)s --price-alert BTC:50000:above        # Preisalarm setzen
  %(prog)s --export csv --currency usd          # Nach CSV exportieren in USD
        """
    )
    
    # Basis-Optionen
    parser.add_argument("-n", "--num", type=int, default=10, 
                       help="Anzahl der Kryptowährungen")
    parser.add_argument("-c", "--currency", type=str, default="eur",
                       help="Währung (eur, usd, gbp, chf)")
    parser.add_argument("--compact", action="store_true",
                       help="Kompakte Anzeige")
    parser.add_argument("--show-volume", action="store_true",
                       help="Handelsvolumen anzeigen")
    parser.add_argument("--show-supply", action="store_true",
                       help="Umlaufmenge anzeigen")
    
    # Portfolio-Features
    parser.add_argument("--portfolio", action="store_true",
                       help="Portfolio anzeigen")
    parser.add_argument("--add-to-portfolio", nargs=3, metavar=("SYMBOL", "AMOUNT", "PRICE"),
                       help="Coin zum Portfolio hinzufügen: SYMBOL MENGE PREIS")
    
    # Export
    parser.add_argument("--export", choices=["csv", "json", "excel"],
                       help="Exportformat")
    
    # Preisalarme
    parser.add_argument("--price-alert", nargs="+",
                       help="Preisalarme: SYMBOL:PREIS:above|below")
    
    # Konfiguration
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Konfigurationsdatei")
    
    return parser


def main():
    """Hauptfunktion"""
    parser = create_enhanced_cli()
    args = parser.parse_args()
    
    # Analyzer initialisieren
    analyzer = CryptoAnalyzer(args.config)
    
    # Portfolio-Management
    if args.add_to_portfolio:
        symbol, amount, price = args.add_to_portfolio
        analyzer.portfolio.add_holding(symbol.upper(), float(amount), float(price))
        msg = f"✅ {symbol.upper()} zum Portfolio hinzugefügt" if analyzer.language == "de" else f"✅ Added {symbol.upper()} to portfolio"
        print(msg)
        return
    
    if args.portfolio:
        analyzer.display_portfolio()
        return
    
    # Kryptowährungsdaten abrufen
    loading_msg = "🔄 Lade Kryptowährungsdaten..." if analyzer.language == "de" else "🔄 Loading cryptocurrency data..."
    print(loading_msg)
    
    df = analyzer.get_crypto_data(
        num=args.num,
        currency=args.currency,
        show_volume=args.show_volume,
        show_supply=args.show_supply
    )
    
    if df.empty:
        return
    
    # Preisalarme prüfen
    if args.price_alert:
        for alert in args.price_alert:
            parts = alert.split(':')
            if len(parts) == 3:
                symbol, target_price, alert_type = parts
                symbol = symbol.upper()
                target_price = float(target_price)
                
                coin_data = df[df['symbol'] == symbol]
                if not coin_data.empty:
                    current_price = coin_data['preis'].iloc[0]
                    coin_name = coin_data['name'].iloc[0]
                    
                    if (alert_type.lower() == "above" and current_price > target_price) or \
                       (alert_type.lower() == "below" and current_price < target_price):
                        print(f"🚨 ALARM: {coin_name} ({symbol}) ist {alert_type} {analyzer.format_currency(target_price)}! "
                              f"Aktuell: {analyzer.format_currency(current_price)}")
    
    # Tabelle anzeigen (vereinfacht für diese Demo)
    print("\n" + "="*80)
    title = f"Top {len(df)} Kryptowährungen" if analyzer.language == "de" else f"Top {len(df)} Cryptocurrencies"
    print(f"{title:^80}")
    print("="*80)
    
    # Kompakte Ansicht für Demo
    display_cols = ['rang', 'symbol', 'name', 'preis', 'marktkapitalisierung', '24h_änderung']
    if args.compact:
        df_display = df[display_cols].head()
    else:
        df_display = df.head()
    
    print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))
    
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    time_msg = f"\n📅 Daten abgerufen am: {timestamp}" if analyzer.language == "de" else f"\n📅 Data fetched at: {timestamp}"
    print(time_msg)


if __name__ == "__main__":
    main()
