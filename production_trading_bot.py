#!/usr/bin/env python3
"""
🔐 Production-Ready Live Trading Bot
Autor: mad4cyber
Version: 1.0 - Production Edition

🛡️ SICHERHEITS-FEATURES:
- Multi-Level Safety Checks
- Conservative Parameter
- Live Trading Preparation
- Risk Management
- Portfolio Protection
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

# API-Keys und Parameter
try:
    import config
    from trading_parameters import *
except ImportError:
    print("❌ config.py oder trading_parameters.py nicht gefunden!")
    exit(1)

# Trading Libraries
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

from trading_signals_demo import SimpleTradingSignals

@dataclass
class SafetyCheck:
    """🛡️ Sicherheitscheck-Ergebnis"""
    is_safe: bool
    warnings: List[str]
    recommendations: List[str]

class ProductionTradingBot:
    """🔐 Production-Ready Trading Bot mit maximaler Sicherheit"""
    
    def __init__(self, mode: TradingMode, risk_profile: RiskProfile):
        self.console = Console()
        self.mode = mode
        self.risk_profile = risk_profile
        
        # Parameter laden
        self.params = get_trading_parameters(mode, risk_profile)
        
        # API Client Setup
        self.client = None
        self.setup_api_client()
        
        # Signal Generator
        self.signal_generator = SimpleTradingSignals()
        
        # Trading State
        self.active_positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.start_balance = 0.0
        
        # Safety Locks
        self.emergency_stop = False
        self.max_daily_loss_reached = False
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ProductionTradingBot')
    
    def setup_api_client(self):
        """🔧 API Client mit Sicherheitschecks einrichten"""
        try:
            if self.mode == TradingMode.TESTNET:
                self.client = Client(
                    api_key=config.BINANCE_TESTNET_API_KEY,
                    api_secret=config.BINANCE_TESTNET_SECRET,
                    testnet=True
                )
                self.console.print("✅ [green]Binance Testnet verbunden[/green]")
            
            elif self.mode in [TradingMode.LIVE_DEMO, TradingMode.LIVE_SMALL, TradingMode.LIVE_NORMAL]:
                # Live API Setup (wenn verfügbar)
                if hasattr(config, 'BINANCE_LIVE_API_KEY') and config.BINANCE_LIVE_API_KEY != "dein_live_api_key_hier":
                    self.client = Client(
                        api_key=config.BINANCE_LIVE_API_KEY,
                        api_secret=config.BINANCE_LIVE_SECRET,
                        testnet=False
                    )
                    self.console.print("⚠️ [red]BINANCE LIVE API VERBUNDEN![/red]")
                else:
                    self.console.print("❌ [red]Live API-Keys nicht konfiguriert - verwende Testnet[/red]")
                    self.mode = TradingMode.TESTNET
                    self.setup_api_client()
                    return
            
            # API-Verbindung testen
            if self.client:
                account = self.client.get_account()
                self.start_balance = self.get_usdt_balance()
                self.console.print(f"💰 [green]Start Balance: ${self.start_balance:,.2f} USDT[/green]")
                
        except Exception as e:
            self.console.print(f"❌ [red]API-Setup Fehler: {e}[/red]")
            self.client = None
    
    def get_usdt_balance(self) -> float:
        """💰 USDT-Guthaben abrufen"""
        if not self.client:
            return 0.0
        
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des USDT-Guthabens: {e}")
            return 0.0
    
    def perform_comprehensive_safety_check(self) -> SafetyCheck:
        """🛡️ Umfassende Sicherheitsprüfung vor Trading"""
        warnings = []\n        recommendations = []\n        is_safe = True\n        \n        # 1. Balance Check\n        current_balance = self.get_usdt_balance()\n        if current_balance < 100 and self.mode in [TradingMode.LIVE_DEMO, TradingMode.LIVE_SMALL, TradingMode.LIVE_NORMAL]:\n            warnings.append("⚠️ USDT Balance unter $100")\n            recommendations.append("💡 Erhöhe Balance auf mindestens $100 für sicheres Trading")\n            if self.mode != TradingMode.LIVE_DEMO:\n                is_safe = False\n        \n        # 2. Parameter Safety Check\n        param_safe, param_warnings = validate_live_trading_safety(self.mode, current_balance, self.params)\n        if not param_safe:\n            is_safe = False\n        warnings.extend(param_warnings)\n        \n        # 3. API Connection Check\n        if not self.client:\n            warnings.append("❌ Keine API-Verbindung")\n            is_safe = False\n        \n        # 4. Daily Loss Check\n        if self.daily_pnl < -current_balance * 0.05:  # -5% Daily Loss Limit\n            warnings.append("🚨 Tägliches Verlustlimit erreicht (-5%)")\n            recommendations.append("💡 Stoppe Trading für heute")\n            self.max_daily_loss_reached = True\n            is_safe = False\n        \n        # 5. Position Size Check\n        max_position = calculate_optimal_position_size(\n            current_balance, 0.90, 3.0, 'BTCUSDT', self.params\n        )\n        if max_position > current_balance * 0.20:\n            warnings.append("⚠️ Maximale Positionsgröße über 20% des Portfolios")\n            recommendations.append("💡 Reduziere base_position_size_usd in den Parametern")\n        \n        # 6. Mode-spezifische Checks\n        if self.mode == TradingMode.LIVE_NORMAL and current_balance < 500:\n            warnings.append("⚠️ Balance unter $500 für normales Live Trading")\n            recommendations.append("💡 Verwende LIVE_SMALL Modus für kleinere Balances")\n        \n        return SafetyCheck(is_safe, warnings, recommendations)\n    \n    def display_safety_dashboard(self, safety_check: SafetyCheck):\n        """🛡️ Sicherheits-Dashboard anzeigen"""\n        self.console.print(Panel.fit("🛡️ Live Trading Safety Dashboard", style="bold red"))\n        \n        # Status\n        status_color = "green" if safety_check.is_safe else "red"\n        status_text = "SICHER ✅" if safety_check.is_safe else "UNSICHER ❌"\n        self.console.print(f"🔒 [bold {status_color}]Safety Status: {status_text}[/bold {status_color}]")\n        \n        # Current Settings\n        settings_table = Table(title="📊 Aktuelle Trading-Einstellungen", box=box.SIMPLE)\n        settings_table.add_column("Parameter", style="cyan")\n        settings_table.add_column("Wert", style="white")\n        settings_table.add_column("Sicherheit", style="green")\n        \n        # Risk per Trade\n        risk_color = "green" if self.params.max_risk_per_trade <= 0.01 else "yellow" if self.params.max_risk_per_trade <= 0.02 else "red"\n        risk_safety = "Sehr sicher" if self.params.max_risk_per_trade <= 0.01 else "Sicher" if self.params.max_risk_per_trade <= 0.02 else "Riskant"\n        settings_table.add_row("💰 Risk per Trade", f"{self.params.max_risk_per_trade:.1%}", f"[{risk_color}]{risk_safety}[/{risk_color}]")\n        \n        # AI Confidence\n        conf_color = "green" if self.params.min_ai_confidence >= 0.90 else "yellow" if self.params.min_ai_confidence >= 0.85 else "red"\n        conf_safety = "Sehr sicher" if self.params.min_ai_confidence >= 0.90 else "Sicher" if self.params.min_ai_confidence >= 0.85 else "Riskant"\n        settings_table.add_row("🧠 Min AI Confidence", f"{self.params.min_ai_confidence:.1%}", f"[{conf_color}]{conf_safety}[/{conf_color}]")\n        \n        # Position Size\n        pos_color = "green" if self.params.base_position_size_usd <= 50 else "yellow" if self.params.base_position_size_usd <= 100 else "red"\n        pos_safety = "Sehr sicher" if self.params.base_position_size_usd <= 50 else "Sicher" if self.params.base_position_size_usd <= 100 else "Riskant"\n        settings_table.add_row("📏 Base Position Size", f"${self.params.base_position_size_usd:.0f}", f"[{pos_color}]{pos_safety}[/{pos_color}]")\n        \n        # Max Positions\n        max_pos_color = "green" if self.params.max_positions <= 3 else "yellow" if self.params.max_positions <= 5 else "red"\n        max_pos_safety = "Sehr sicher" if self.params.max_positions <= 3 else "Sicher" if self.params.max_positions <= 5 else "Riskant"\n        settings_table.add_row("📊 Max Positions", str(self.params.max_positions), f"[{max_pos_color}]{max_pos_safety}[/{max_pos_color}]")\n        \n        self.console.print(settings_table)\n        \n        # Warnings\n        if safety_check.warnings:\n            warning_table = Table(title="⚠️ Sicherheitswarnungen", box=box.SIMPLE)\n            warning_table.add_column("Warnung", style="red")\n            for warning in safety_check.warnings:\n                warning_table.add_row(warning)\n            self.console.print(warning_table)\n        \n        # Recommendations\n        if safety_check.recommendations:\n            rec_table = Table(title="💡 Empfehlungen", box=box.SIMPLE)\n            rec_table.add_column("Empfehlung", style="cyan")\n            for rec in safety_check.recommendations:\n                rec_table.add_row(rec)\n            self.console.print(rec_table)\n    \n    async def safe_execute_trade(self, symbol: str, signal_data: Dict) -> bool:\n        """🔒 Sicherer Trade mit allen Checks"""        \n        try:\n            # Pre-Trade Safety Check\n            safety = self.perform_comprehensive_safety_check()\n            if not safety.is_safe:\n                self.console.print("🚨 [red]Trade abgebrochen - Sicherheitschecks fehlgeschlagen![/red]")\n                return False\n            \n            # Position Size berechnen\n            current_balance = self.get_usdt_balance()\n            position_size_usd = calculate_optimal_position_size(\n                current_balance,\n                signal_data['confidence'],\n                signal_data['price_change_pct'],\n                symbol,\n                self.params\n            )\n            \n            # Minimale Positionsgröße für Live Trading\n            if self.mode != TradingMode.TESTNET:\n                min_position_usd = 10.0 if self.mode == TradingMode.LIVE_DEMO else 25.0\n                if position_size_usd < min_position_usd:\n                    self.console.print(f"⚠️ [yellow]Position zu klein (${position_size_usd:.2f} < ${min_position_usd})[/yellow]")\n                    return False\n            \n            # Aktuelle Preise abrufen\n            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])\n            quantity = position_size_usd / current_price\n            \n            # Minimum quantity check\n            if quantity < 0.001:  # Binance minimum\n                self.console.print(f"⚠️ [yellow]Quantity zu klein: {quantity:.6f}[/yellow]")\n                return False\n            \n            # Daily Trade Limit Check\n            if self.daily_trades >= self.params.max_daily_trades:\n                self.console.print("⚠️ [yellow]Tägliches Trade-Limit erreicht[/yellow]")\n                return False\n            \n            # Final Confirmation für Live Trading\n            if self.mode in [TradingMode.LIVE_SMALL, TradingMode.LIVE_NORMAL]:\n                self.console.print(f"\\n🔥 [bold red]LIVE TRADING ORDER:[/bold red]")\n                self.console.print(f"💰 Symbol: {symbol}")\n                self.console.print(f"📊 Amount: {quantity:.6f} (${position_size_usd:.2f})")\n                self.console.print(f"🎯 AI Confidence: {signal_data['confidence']:.1%}")\n                self.console.print(f"📈 Expected: {signal_data['price_change_pct']:+.2f}%")\n                \n                if not Confirm.ask("\\n⚠️ [bold red]BESTÄTIGE LIVE TRADE MIT ECHTEM GELD[/bold red]"):\n                    self.console.print("❌ [yellow]Trade vom Benutzer abgebrochen[/yellow]")\n                    return False\n            \n            # Order ausführen\n            self.console.print(f\"🔄 [yellow]Führe {symbol} Trade aus...[/yellow]\")\n            \n            if signal_data['signal'] in ['STRONG_BUY', 'BUY']:\n                order = self.client.order_market(\n                    symbol=symbol,\n                    side=SIDE_BUY,\n                    quantity=quantity\n                )\n            else:\n                # Für SELL: Prüfe ob Position vorhanden\n                # Vereinfacht für Demo\n                self.console.print("⚠️ [yellow]SELL-Orders noch nicht implementiert[/yellow]")\n                return False\n            \n            # Erfolgreiche Order\n            self.console.print(f\"✅ [green]Order erfolgreich: {order['orderId']}[/green]\")\n            self.console.print(f\"📊 [green]Status: {order['status']} | Executed: {order['executedQty']}[/green]\")\n            \n            # Stats aktualisieren\n            self.daily_trades += 1\n            \n            # Stop-Loss/Take-Profit Orders setzen (vereinfacht)\n            # TODO: Implementiere OCO Orders für Stop-Loss/Take-Profit\n            \n            return True\n            \n        except BinanceAPIException as e:\n            self.console.print(f\"❌ [red]Binance API Fehler: {e.message}[/red]\")\n            return False\n        except Exception as e:\n            self.console.print(f\"❌ [red]Unbekannter Fehler: {e}[/red]\")\n            return False\n    \n    async def run_production_trading(self, duration_hours: float = 1.0):\n        \"\"\"🚀 Production Trading mit allen Sicherheitschecks\"\"\"        \n        self.console.print(Panel.fit(f\"🔐 Production Trading Bot - {self.mode.value.upper()}\", style=\"bold red\"))\n        \n        # Initial Safety Check\n        safety = self.perform_comprehensive_safety_check()\n        self.display_safety_dashboard(safety)\n        \n        if not safety.is_safe:\n            self.console.print(\"\\n🚨 [red]TRADING NICHT SICHER - Bot wird nicht gestartet![/red]\")\n            return\n        \n        # Final Confirmation\n        if self.mode in [TradingMode.LIVE_DEMO, TradingMode.LIVE_SMALL, TradingMode.LIVE_NORMAL]:\n            self.console.print(f\"\\n⚠️ [bold red]ACHTUNG: Live Trading für {duration_hours} Stunden![/bold red]\")\n            if not Confirm.ask(\"Möchtest du fortfahren?\"):\n                self.console.print(\"❌ Trading abgebrochen\")\n                return\n        \n        # Trading Loop\n        end_time = datetime.now() + timedelta(hours=duration_hours)\n        cycle = 0\n        \n        self.console.print(f\"\\n🚀 [green]Production Trading gestartet für {duration_hours} Stunden![/green]\")\n        \n        try:\n            while datetime.now() < end_time and not self.emergency_stop and not self.max_daily_loss_reached:\n                cycle += 1\n                self.console.print(f\"\\n🔄 [cyan]Trading-Zyklus {cycle}[/cyan]\")\n                \n                # Safety Check vor jedem Zyklus\n                safety = self.perform_comprehensive_safety_check()\n                if not safety.is_safe:\n                    self.console.print(\"🚨 [red]Sicherheitscheck fehlgeschlagen - stoppe Trading![/red]\")\n                    break\n                \n                # AI-Signale generieren\n                self.console.print(\"🧠 [yellow]Generiere AI-Signale...[/yellow]\")\n                signals = await self.signal_generator.generate_signals()\n                \n                if signals:\n                    # Beste Signale filtern\n                    high_confidence_signals = [\n                        signal for signal in signals\n                        if signal['confidence'] >= self.params.min_ai_confidence\n                        and abs(signal['price_change_pct']) >= self.params.min_price_change_threshold\n                        and signal['signal'] in ['STRONG_BUY', 'BUY']\n                    ]\n                    \n                    if high_confidence_signals:\n                        best_signal = high_confidence_signals[0]\n                        symbol = best_signal['symbol'].replace('COIN', '') + 'USDT'\n                        \n                        if symbol in config.TRADING_SYMBOLS:\n                            self.console.print(f\"\\n🎯 [green]Starkes Signal: {symbol}[/green]\")\n                            self.console.print(f\"📊 Signal: {best_signal['signal']} ({best_signal['confidence']:.1%})\")\n                            self.console.print(f\"📈 Erwartung: {best_signal['price_change_pct']:+.2f}%\")\n                            \n                            # Trade ausführen\n                            success = await self.safe_execute_trade(symbol, best_signal)\n                            \n                            if success:\n                                self.console.print(\"✅ [green]Trade erfolgreich ausgeführt![/green]\")\n                                # Pause nach erfolgreichem Trade\n                                await asyncio.sleep(300)  # 5 Minuten Pause\n                            else:\n                                self.console.print(\"❌ [red]Trade fehlgeschlagen[/red]\")\n                    else:\n                        self.console.print(\"⏸️ [yellow]Keine hochwertigen Signale gefunden[/yellow]\")\n                else:\n                    self.console.print(\"❌ [red]Keine AI-Signale generiert[/red]\")\n                \n                # Warten bis zum nächsten Zyklus (15 Minuten)\n                if datetime.now() < end_time:\n                    self.console.print(\"⏳ [dim]Warte 15 Minuten bis zum nächsten Zyklus...[/dim]\")\n                    await asyncio.sleep(900)  # 15 Minuten\n                \n        except KeyboardInterrupt:\n            self.console.print(\"\\n🛑 [yellow]Trading manuell gestoppt[/yellow]\")\n        except Exception as e:\n            self.console.print(f\"\\n❌ [red]Kritischer Fehler: {e}[/red]\")\n        finally:\n            # Final Stats\n            final_balance = self.get_usdt_balance()\n            total_pnl = final_balance - self.start_balance\n            \n            self.console.print(\"\\n📊 [green]Trading Session beendet[/green]\")\n            self.console.print(f\"💰 Start Balance: ${self.start_balance:,.2f}\")\n            self.console.print(f\"💰 End Balance: ${final_balance:,.2f}\")\n            self.console.print(f\"📈 Total P&L: ${total_pnl:+,.2f} ({(total_pnl/self.start_balance)*100:+.2f}%)\")\n            self.console.print(f\"📊 Trades ausgeführt: {self.daily_trades}\")\n\n\nasync def main():\n    \"\"\"🎯 Hauptfunktion für Production Trading\"\"\"    \n    console = Console()\n    \n    console.print(Panel.fit(\"🔐 Production Trading Bot Setup\", style=\"bold magenta\"))\n    \n    # Mode Selection\n    console.print(\"\\n🔧 Wähle Trading-Modus:\")\n    console.print(\"1. Testnet (Sicher, Fake-Geld)\")\n    console.print(\"2. Live Demo (Echtes Geld, Mini-Beträge <$25)\")\n    console.print(\"3. Live Small (Echtes Geld, kleine Beträge <$100)\")\n    console.print(\"4. Live Normal (Echtes Geld, normale Beträge)\")\n    \n    mode_choice = Prompt.ask(\"Wähle Modus\", choices=[\"1\", \"2\", \"3\", \"4\"], default=\"1\")\n    \n    mode_map = {\n        \"1\": TradingMode.TESTNET,\n        \"2\": TradingMode.LIVE_DEMO,\n        \"3\": TradingMode.LIVE_SMALL,\n        \"4\": TradingMode.LIVE_NORMAL\n    }\n    trading_mode = mode_map[mode_choice]\n    \n    # Risk Profile Selection\n    console.print(\"\\n⚖️ Wähle Risk Profile:\")\n    console.print(\"1. Conservative (Sehr sicher, kleine Gewinne)\")\n    console.print(\"2. Moderate (Ausgewogen)\")\n    console.print(\"3. Aggressive (Höher Risiko, höhere Gewinne)\")\n    \n    risk_choice = Prompt.ask(\"Wähle Risk Profile\", choices=[\"1\", \"2\", \"3\"], default=\"1\")\n    \n    risk_map = {\n        \"1\": RiskProfile.CONSERVATIVE,\n        \"2\": RiskProfile.MODERATE,\n        \"3\": RiskProfile.AGGRESSIVE\n    }\n    risk_profile = risk_map[risk_choice]\n    \n    # Duration Selection\n    if trading_mode == TradingMode.TESTNET:\n        duration = float(Prompt.ask(\"Trading-Dauer (Stunden)\", default=\"0.5\"))\n    else:\n        duration = float(Prompt.ask(\"Trading-Dauer (Stunden)\", default=\"0.25\"))  # 15 Min für Live\n    \n    # Bot erstellen und starten\n    bot = ProductionTradingBot(trading_mode, risk_profile)\n    await bot.run_production_trading(duration)\n\n\nif __name__ == \"__main__\":\n    if not BINANCE_AVAILABLE:\n        print(\"❌ Binance Library fehlt: pip install python-binance\")\n    else:\n        asyncio.run(main())
