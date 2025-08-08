#!/usr/bin/env python3
"""
⚡ Real-time Crypto Alert Manager
Autor: mad4cyber
Version: 1.0 - Alert Edition

🚀 FEATURES:
- High-Confidence AI Signal Alerts
- Price Movement Alerts
- Portfolio Risk Alerts
- Stop-Loss/Take-Profit Notifications
- Email & Browser Notifications
- Alert History & Statistics
"""

import json
import time
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import threading
import warnings
warnings.filterwarnings('ignore')

from ai_predictor import CryptoAIPredictor
from portfolio_manager import PortfolioManager
from sentiment_analyzer import MarketSentimentAnalyzer

@dataclass
class Alert:
    """⚡ Einzelne Alert-Definition"""
    id: str
    alert_type: str  # 'confidence', 'price_change', 'portfolio_risk', 'stop_loss', 'take_profit'
    coin_id: str
    coin_symbol: str
    condition: Dict  # Bedingungen für Alert-Trigger
    message: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    created_date: str
    triggered_date: str = ""
    is_active: bool = True
    is_triggered: bool = False
    trigger_count: int = 0
    user_email: str = ""
    
    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()

@dataclass
class AlertNotification:
    """📨 Alert Benachrichtigung"""
    alert_id: str
    coin_symbol: str
    message: str
    priority: str
    timestamp: str
    data: Dict = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class AlertManager:
    """⚡ Real-time Crypto Alert Management System"""
    
    def __init__(self, alerts_file: str = "alerts_data.json"):
        self.alerts_file = alerts_file
        self.ai_predictor = CryptoAIPredictor()
        self.portfolio_manager = PortfolioManager()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # Alert Storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[AlertNotification] = []
        
        # Notification callbacks
        self.notification_callbacks: List[Callable] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.check_interval = 300  # 5 Minuten
        
        # Load existing alerts
        self.load_alerts()
        
        # Email configuration (optional)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'enabled': False
        }
    
    def load_alerts(self):
        """📁 Lade Alerts aus Datei"""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                
                # Load active alerts
                for alert_id, alert_data in data.get('active_alerts', {}).items():
                    self.active_alerts[alert_id] = Alert(**alert_data)
                
                # Load history
                for notification_data in data.get('alert_history', []):
                    self.alert_history.append(AlertNotification(**notification_data))
                    
            except Exception as e:
                print(f"❌ Fehler beim Laden der Alerts: {e}")
    
    def save_alerts(self):
        """💾 Speichere Alerts in Datei"""
        try:
            data = {
                'active_alerts': {
                    alert_id: asdict(alert) for alert_id, alert in self.active_alerts.items()
                },
                'alert_history': [asdict(notification) for notification in self.alert_history[-100:]]  # Keep last 100
            }
            
            with open(self.alerts_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Alerts: {e}")
    
    def create_confidence_alert(self, coin_id: str, coin_symbol: str, 
                               min_confidence: float = 0.7, user_email: str = "") -> str:
        """🎯 Erstelle High-Confidence Alert"""
        alert_id = f"confidence_{coin_id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            alert_type="confidence",
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            condition={
                'min_confidence': min_confidence,
                'min_price_change': 5.0  # Mindestens 5% erwartete Bewegung
            },
            message=f"🚀 High-Confidence Signal für {coin_symbol}: AI Konfidenz >{min_confidence:.0%}",
            priority="high",
            created_date=datetime.now().isoformat(),
            user_email=user_email
        )
        
        self.active_alerts[alert_id] = alert
        self.save_alerts()
        return alert_id
    
    def create_price_alert(self, coin_id: str, coin_symbol: str, 
                          target_price: float, direction: str = "above", user_email: str = "") -> str:
        """💰 Erstelle Preis-Alert"""
        alert_id = f"price_{coin_id}_{direction}_{int(time.time())}"
        
        direction_text = "steigt über" if direction == "above" else "fällt unter"
        
        alert = Alert(
            id=alert_id,
            alert_type="price_change",
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            condition={
                'target_price': target_price,
                'direction': direction
            },
            message=f"💰 {coin_symbol} {direction_text} €{target_price:.4f}",
            priority="medium",
            created_date=datetime.now().isoformat(),
            user_email=user_email
        )
        
        self.active_alerts[alert_id] = alert
        self.save_alerts()
        return alert_id
    
    def create_portfolio_risk_alert(self, portfolio_name: str, max_loss_pct: float = -10.0, 
                                   user_email: str = "") -> str:
        """⚠️ Erstelle Portfolio Risk Alert"""
        alert_id = f"portfolio_risk_{portfolio_name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            alert_type="portfolio_risk",
            coin_id="portfolio",
            coin_symbol=portfolio_name,
            condition={
                'max_loss_pct': max_loss_pct,
                'portfolio_name': portfolio_name
            },
            message=f"⚠️ Portfolio Verlust Warnung: {portfolio_name} unter {max_loss_pct}%",
            priority="critical",
            created_date=datetime.now().isoformat(),
            user_email=user_email
        )
        
        self.active_alerts[alert_id] = alert
        self.save_alerts()
        return alert_id
    
    def create_stop_loss_alert(self, coin_id: str, coin_symbol: str, 
                              stop_loss_price: float, user_email: str = "") -> str:
        """🛑 Erstelle Stop-Loss Alert"""
        alert_id = f"stop_loss_{coin_id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            alert_type="stop_loss",
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            condition={
                'stop_loss_price': stop_loss_price
            },
            message=f"🛑 STOP LOSS TRIGGERED: {coin_symbol} gefallen auf €{stop_loss_price:.4f}",
            priority="critical",
            created_date=datetime.now().isoformat(),
            user_email=user_email
        )
        
        self.active_alerts[alert_id] = alert
        self.save_alerts()
        return alert_id
    
    def create_take_profit_alert(self, coin_id: str, coin_symbol: str, 
                                take_profit_price: float, user_email: str = "") -> str:
        """🎯 Erstelle Take-Profit Alert"""
        alert_id = f"take_profit_{coin_id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            alert_type="take_profit",
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            condition={
                'take_profit_price': take_profit_price
            },
            message=f"🎯 TAKE PROFIT: {coin_symbol} erreicht €{take_profit_price:.4f}",
            priority="high",
            created_date=datetime.now().isoformat(),
            user_email=user_email
        )
        
        self.active_alerts[alert_id] = alert
        self.save_alerts()
        return alert_id
    
    def check_confidence_alerts(self):
        """🎯 Prüfe Confidence-basierte Alerts"""
        confidence_alerts = [alert for alert in self.active_alerts.values() 
                           if alert.alert_type == "confidence" and alert.is_active]
        
        for alert in confidence_alerts:
            try:
                # Hole AI-Prognose
                prediction = self.ai_predictor.predict_future_prices(alert.coin_id)
                
                if 'error' in prediction:
                    continue
                
                confidence = prediction.get('confidence', 0)
                price_change_pct = abs(prediction.get('price_change_pct', 0))
                
                # Prüfe Bedingungen
                min_confidence = alert.condition.get('min_confidence', 0.7)
                min_price_change = alert.condition.get('min_price_change', 5.0)
                
                if confidence >= min_confidence and price_change_pct >= min_price_change:
                    # Alert triggered!
                    enhanced_message = (
                        f"🚀 {alert.coin_symbol} HIGH-CONFIDENCE SIGNAL!\n"
                        f"📊 AI Konfidenz: {confidence:.1%}\n"
                        f"📈 Erwartete Bewegung: {prediction.get('price_change_pct', 0):+.2f}%\n"
                        f"💰 Aktueller Preis: €{prediction.get('current_price', 0):.4f}\n"
                        f"🎯 24h Prognose: €{prediction.get('predicted_price', 0):.4f}"
                    )
                    
                    self.trigger_alert(alert, enhanced_message, {
                        'confidence': confidence,
                        'price_change_pct': prediction.get('price_change_pct', 0),
                        'current_price': prediction.get('current_price', 0),
                        'predicted_price': prediction.get('predicted_price', 0)
                    })
                    
            except Exception as e:
                print(f"❌ Fehler bei Confidence Alert {alert.id}: {e}")
    
    def check_price_alerts(self):
        """💰 Prüfe Preis-basierte Alerts"""
        price_alerts = [alert for alert in self.active_alerts.values() 
                       if alert.alert_type in ["price_change", "stop_loss", "take_profit"] and alert.is_active]
        
        if not price_alerts:
            return
        
        # Hole aktuelle Preise für alle relevanten Coins
        coin_ids = list(set([alert.coin_id for alert in price_alerts]))
        current_prices = self.portfolio_manager.get_current_prices(coin_ids)
        
        for alert in price_alerts:
            try:
                current_price = current_prices.get(alert.coin_id, 0)
                if current_price == 0:
                    continue
                
                triggered = False
                
                if alert.alert_type == "price_change":
                    target_price = alert.condition.get('target_price', 0)
                    direction = alert.condition.get('direction', 'above')
                    
                    if direction == "above" and current_price >= target_price:
                        triggered = True
                    elif direction == "below" and current_price <= target_price:
                        triggered = True
                        
                elif alert.alert_type == "stop_loss":
                    stop_loss_price = alert.condition.get('stop_loss_price', 0)
                    if current_price <= stop_loss_price:
                        triggered = True
                        
                elif alert.alert_type == "take_profit":
                    take_profit_price = alert.condition.get('take_profit_price', 0)
                    if current_price >= take_profit_price:
                        triggered = True
                
                if triggered:
                    enhanced_message = f"{alert.message}\n💰 Aktueller Preis: €{current_price:.4f}"
                    self.trigger_alert(alert, enhanced_message, {
                        'current_price': current_price,
                        'trigger_price': alert.condition.get('target_price', 
                                       alert.condition.get('stop_loss_price',
                                       alert.condition.get('take_profit_price', 0)))
                    })
                    
            except Exception as e:
                print(f"❌ Fehler bei Price Alert {alert.id}: {e}")
    
    def check_portfolio_alerts(self):
        """📊 Prüfe Portfolio-basierte Alerts"""
        portfolio_alerts = [alert for alert in self.active_alerts.values() 
                           if alert.alert_type == "portfolio_risk" and alert.is_active]
        
        for alert in portfolio_alerts:
            try:
                portfolio_name = alert.condition.get('portfolio_name', 'Main Portfolio')
                portfolio_data = self.portfolio_manager.calculate_portfolio_value(portfolio_name)
                
                if 'error' in portfolio_data:
                    continue
                
                total_pnl_pct = portfolio_data['summary'].get('total_pnl_pct', 0)
                max_loss_pct = alert.condition.get('max_loss_pct', -10.0)
                
                if total_pnl_pct <= max_loss_pct:
                    enhanced_message = (
                        f"⚠️ PORTFOLIO RISK ALERT!\n"
                        f"📊 {portfolio_name}\n"
                        f"📉 Aktueller Verlust: {total_pnl_pct:.2f}%\n"
                        f"💰 Portfolio Wert: €{portfolio_data['summary']['total_portfolio_value']:.2f}\n"
                        f"🛑 Erwäge Risk Management Maßnahmen!"
                    )
                    
                    self.trigger_alert(alert, enhanced_message, {
                        'portfolio_pnl_pct': total_pnl_pct,
                        'portfolio_value': portfolio_data['summary']['total_portfolio_value'],
                        'portfolio_positions': len(portfolio_data.get('positions', []))
                    })
                    
            except Exception as e:
                print(f"❌ Fehler bei Portfolio Alert {alert.id}: {e}")
    
    def trigger_alert(self, alert: Alert, message: str, data: Dict = None):
        """🔔 Löse Alert aus"""
        alert.is_triggered = True
        alert.triggered_date = datetime.now().isoformat()
        alert.trigger_count += 1
        
        # Erstelle Notification
        notification = AlertNotification(
            alert_id=alert.id,
            coin_symbol=alert.coin_symbol,
            message=message,
            priority=alert.priority,
            timestamp=datetime.now().isoformat(),
            data=data or {}
        )
        
        # Speichere in History
        self.alert_history.append(notification)
        
        # Deaktiviere Alert (einmalig) oder setze Cooldown
        if alert.alert_type in ["stop_loss", "take_profit"]:
            alert.is_active = False  # Einmalige Alerts
        else:
            # Setze 1h Cooldown für wiederholbare Alerts
            alert.is_active = False
            # Reaktiviere nach 1h (vereinfacht)
        
        # Sende Benachrichtigungen
        self.send_notifications(notification)
        
        # Speichere Updates
        self.save_alerts()
        
        print(f"🔔 Alert triggered: {alert.coin_symbol} - {alert.alert_type}")
    
    def send_notifications(self, notification: AlertNotification):
        """📨 Sende Benachrichtigungen"""
        
        # Browser/Console Notification (immer)
        print(f"\n🔔 CRYPTO ALERT [{notification.priority.upper()}]")
        print(f"📊 {notification.coin_symbol}")
        print(f"💬 {notification.message}")
        print(f"⏰ {notification.timestamp}")
        print("-" * 50)
        
        # Call registered callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                print(f"❌ Notification callback error: {e}")
        
        # Email notification (if configured)
        if self.email_config.get('enabled', False):
            self.send_email_notification(notification)
    
    def send_email_notification(self, notification: AlertNotification):
        """📧 Sende Email-Benachrichtigung"""
        try:
            if not self.email_config.get('sender_email') or not self.email_config.get('sender_password'):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = notification.alert.user_email if hasattr(notification, 'alert') else ""
            msg['Subject'] = f"🔔 Crypto Alert: {notification.coin_symbol}"
            
            body = f"""
            Crypto AI Alert Notification
            
            Symbol: {notification.coin_symbol}
            Priority: {notification.priority.upper()}
            Time: {notification.timestamp}
            
            Message:
            {notification.message}
            
            ---
            Crypto AI Predictor
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['sender_email'], msg['To'], text)
            server.quit()
            
            print(f"📧 Email alert sent to {msg['To']}")
            
        except Exception as e:
            print(f"❌ Email notification error: {e}")
    
    def start_monitoring(self):
        """🔍 Starte Real-time Monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Alert monitoring started...")
    
    def stop_monitoring(self):
        """⏹️ Stoppe Monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⏹️ Alert monitoring stopped...")
    
    def _monitor_loop(self):
        """🔄 Monitoring Loop"""
        while self.is_monitoring:
            try:
                print(f"🔍 Checking alerts... ({datetime.now().strftime('%H:%M:%S')})")
                
                # Prüfe alle Alert-Typen
                self.check_confidence_alerts()
                self.check_price_alerts()
                self.check_portfolio_alerts()
                
                # Warte bis zum nächsten Check
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Kurze Pause bei Fehlern
    
    def add_notification_callback(self, callback: Callable):
        """📢 Füge Notification Callback hinzu"""
        self.notification_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Dict]:
        """📋 Hole aktive Alerts"""
        return [asdict(alert) for alert in self.active_alerts.values() if alert.is_active]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """📊 Hole Alert History"""
        return [asdict(notification) for notification in self.alert_history[-limit:]]
    
    def delete_alert(self, alert_id: str) -> bool:
        """🗑️ Lösche Alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.save_alerts()
            return True
        return False
    
    def get_alert_statistics(self) -> Dict:
        """📈 Alert Statistiken"""
        total_alerts = len(self.active_alerts)
        active_alerts = len([a for a in self.active_alerts.values() if a.is_active])
        triggered_alerts = len([a for a in self.active_alerts.values() if a.is_triggered])
        
        # History stats
        recent_notifications = [n for n in self.alert_history 
                              if datetime.fromisoformat(n.timestamp) > datetime.now() - timedelta(days=7)]
        
        priority_counts = {}
        for notification in recent_notifications:
            priority_counts[notification.priority] = priority_counts.get(notification.priority, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_alerts': triggered_alerts,
            'recent_notifications_7d': len(recent_notifications),
            'priority_distribution': priority_counts,
            'monitoring_active': self.is_monitoring,
            'check_interval_minutes': self.check_interval / 60
        }

# Test des Alert Systems
def main():
    """⚡ Test der Alert-Funktionen"""
    alert_manager = AlertManager()
    
    print("⚡ Alert Manager - Demo")
    print("=" * 50)
    
    # Erstelle Test-Alerts
    alert_id1 = alert_manager.create_confidence_alert('bitcoin', 'BTC', min_confidence=0.8)
    alert_id2 = alert_manager.create_price_alert('ethereum', 'ETH', target_price=3000, direction='above')
    alert_id3 = alert_manager.create_portfolio_risk_alert('Main Portfolio', max_loss_pct=-5.0)
    
    print(f"✅ Created alerts: {alert_id1}, {alert_id2}, {alert_id3}")
    
    # Zeige aktive Alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"📋 Active alerts: {len(active_alerts)}")
    
    # Zeige Statistiken
    stats = alert_manager.get_alert_statistics()
    print(f"📊 Alert Stats: {stats}")
    
    # Starte Monitoring für Demo (kurz)
    print("\n🔍 Starting monitoring for 30 seconds...")
    alert_manager.check_interval = 10  # 10 Sekunden für Demo
    alert_manager.start_monitoring()
    
    time.sleep(30)
    alert_manager.stop_monitoring()
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()