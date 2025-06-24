#!/usr/bin/env python3
"""
🧠 AI-Powered Kryptowährungs-Preisprognose
Autor: mad4cyber
Version: 4.0 - AI Edition
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Deep Learning (optional) - Deaktiviert für bessere Kompatibilität
DL_AVAILABLE = False

from pycoingecko import CoinGeckoAPI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box

class CryptoAIPredictor:
    """🧠 AI-Powered Kryptowährungs-Preisprognose"""
    
    def __init__(self):
        self.console = Console()
        self.cg = CoinGeckoAPI()
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
    def fetch_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """📈 Historische Daten für AI-Training abrufen"""
        try:
            # Historische Preisdaten von CoinGecko
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='eur',
                days=days,
                interval='daily'
            )
            
            # DataFrame erstellen
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(p[0]/1000) for p in prices],
                'price': [p[1] for p in prices],
                'volume': [v[1] for v in volumes],
                'market_cap': [m[1] for m in market_caps]
            })
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            self.console.print(f"❌ [red]Fehler beim Abrufen historischer Daten: {e}[/red]")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Features für ML-Modell erstellen"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Technische Indikatoren
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_30'] = df['price'].rolling(window=30).mean()
        df['price_std_7'] = df['price'].rolling(window=7).std()
        
        # Preisänderungen
        df['price_change_1d'] = df['price'].pct_change(1)
        df['price_change_7d'] = df['price'].pct_change(7)
        df['price_change_30d'] = df['price'].pct_change(30)
        
        # Volume-Indikatoren
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_7']
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_upper'] = df['price_ma_30'] + (df['price_std_7'] * 2)
        df['bb_lower'] = df['price_ma_30'] - (df['price_std_7'] * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Future price (Target)
        df['future_price_1d'] = df['price'].shift(-1)
        df['future_price_7d'] = df['price'].shift(-7)
        
        # NaN-Werte entfernen
        df = df.dropna()
        
        return df
    
    def train_ml_models(self, df: pd.DataFrame, target_days: int = 1) -> Dict:
        """🤖 Multiple ML-Modelle trainieren"""
        if not ML_AVAILABLE:
            self.console.print("❌ [red]Scikit-learn nicht verfügbar. Installiere mit: pip install scikit-learn[/red]")
            return {}
        
        target_col = f'future_price_{target_days}d'
        if target_col not in df.columns:
            self.console.print(f"❌ [red]Target-Spalte {target_col} nicht gefunden[/red]")
            return {}
        
        # Features auswählen
        feature_cols = [
            'price', 'volume', 'market_cap',
            'price_ma_7', 'price_ma_30', 'price_std_7',
            'price_change_1d', 'price_change_7d', 'price_change_30d',
            'volume_ma_7', 'volume_ratio', 'rsi', 'bb_position'
        ]
        
        # Verfügbare Features filtern
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].values
        y = df[target_col].values
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Skalierung
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        performance = {}
        
        # 1. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        models['random_forest'] = rf_model
        performance['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'feature_importance': dict(zip(available_features, rf_model.feature_importances_))
        }
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = gb_model
        performance['gradient_boosting'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred),
            'feature_importance': dict(zip(available_features, gb_model.feature_importances_))
        }
        
        # 3. Linear Regression (Baseline)
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        models['linear_regression'] = lr_model
        performance['linear_regression'] = {
            'mae': mean_absolute_error(y_test, lr_pred),
            'r2': r2_score(y_test, lr_pred),
            'feature_importance': dict(zip(available_features, lr_model.coef_))
        }
        
        return {
            'models': models,
            'scaler': scaler,
            'performance': performance,
            'features': available_features
        }
    
    def create_lstm_model(self, df: pd.DataFrame, sequence_length: int = 60) -> Optional[Dict]:
        """🧠 LSTM Deep Learning Modell erstellen"""
        if not DL_AVAILABLE:
            self.console.print("❌ [red]TensorFlow nicht verfügbar. Installiere mit: pip install tensorflow[/red]")
            return None
        
        # Daten vorbereiten
        prices = df['price'].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Sequenzen erstellen
        X, y = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Train/Test Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # LSTM Modell
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Training
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluation
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test_actual, predictions)
        r2 = r2_score(y_test_actual, predictions)
        
        return {
            'model': model,
            'scaler': scaler,
            'performance': {'mae': mae, 'r2': r2},
            'sequence_length': sequence_length
        }
    
    def predict_future_prices(self, coin_id: str, days_ahead: int = 7) -> Dict:
        """🔮 Zukünftige Preise vorhersagen"""
        with Progress() as progress:
            task = progress.add_task("🧠 AI-Prognose wird erstellt...", total=100)
            
            # Historische Daten laden
            progress.update(task, advance=20, description="📊 Lade historische Daten...")
            df = self.fetch_historical_data(coin_id, days=365)  # 1 Jahr (API Limit)
            
            if df.empty:
                return {'error': 'Keine historischen Daten verfügbar'}
            
            # Features erstellen
            progress.update(task, advance=20, description="🔧 Erstelle Features...")
            df_features = self.create_features(df)
            
            # ML-Modelle trainieren
            progress.update(task, advance=30, description="🤖 Trainiere ML-Modelle...")
            ml_results = self.train_ml_models(df_features, target_days=1)
            
            # LSTM-Modell übersprungen (TensorFlow nicht verfügbar)
            progress.update(task, advance=20, description="✅ ML-Modelle fertig...")
            lstm_results = None
            
            progress.update(task, advance=10, description="🔮 Erstelle Prognosen...")
            
            # Aktuelle Features für Prognose
            current_features = df_features.iloc[-1][ml_results['features']].values.reshape(1, -1)
            current_features_scaled = ml_results['scaler'].transform(current_features)
            
            # Prognosen von verschiedenen Modellen
            predictions = {}
            for model_name, model in ml_results['models'].items():
                pred = model.predict(current_features_scaled)[0]
                predictions[model_name] = pred
            
            # Ensemble-Prognose (Durchschnitt)
            ensemble_prediction = np.mean(list(predictions.values()))
            
            current_price = df['price'].iloc[-1]
            price_change_pct = ((ensemble_prediction - current_price) / current_price) * 100
            
            return {
                'coin_id': coin_id,
                'current_price': current_price,
                'predicted_price': ensemble_prediction,
                'price_change_pct': price_change_pct,
                'confidence': max(ml_results['performance'][model]['r2'] for model in ml_results['performance']),
                'model_predictions': predictions,
                'model_performance': ml_results['performance'],
                'prediction_date': datetime.now().isoformat(),
                'target_date': (datetime.now() + timedelta(days=days_ahead)).isoformat()
            }
    
    def display_ai_analysis(self, coin_id: str):
        """🎨 AI-Analyse visuell darstellen"""
        self.console.print(Panel.fit("🧠 AI-Powered Krypto-Prognose", style="bold magenta"))
        
        # Prognose erstellen
        result = self.predict_future_prices(coin_id)
        
        if 'error' in result:
            self.console.print(f"❌ [red]{result['error']}[/red]")
            return
        
        # Haupttabelle
        table = Table(title=f"🔮 AI-Prognose für {coin_id.upper()}", box=box.ROUNDED)
        table.add_column("📊 Metrik", style="cyan")
        table.add_column("💰 Wert", style="white")
        table.add_column("📈 Status", style="green")
        
        # Prognose-Details
        current = result['current_price']
        predicted = result['predicted_price']
        change_pct = result['price_change_pct']
        confidence = result['confidence']
        
        status_icon = "🚀" if change_pct > 0 else "📉"
        status_color = "green" if change_pct > 0 else "red"
        
        table.add_row("Aktueller Preis", f"€{current:.2f}", "📊")
        table.add_row("Prognostizierter Preis (24h)", f"€{predicted:.2f}", f"[{status_color}]{status_icon}[/{status_color}]")
        table.add_row("Erwartete Änderung", f"{change_pct:+.2f}%", f"[{status_color}]{status_icon}[/{status_color}]")
        table.add_row("AI-Konfidenz", f"{confidence:.1%}", "🎯")
        
        self.console.print(table)
        
        # Modell-Performance
        perf_table = Table(title="🤖 Modell-Performance", box=box.SIMPLE)
        perf_table.add_column("Modell", style="cyan")
        perf_table.add_column("R² Score", style="green")
        perf_table.add_column("MAE", style="yellow")
        perf_table.add_column("Prognose", style="white")
        
        for model_name, perf in result['model_performance'].items():
            model_pred = result['model_predictions'][model_name]
            perf_table.add_row(
                model_name.replace('_', ' ').title(),
                f"{perf['r2']:.3f}",
                f"€{perf['mae']:.2f}",
                f"€{model_pred:.2f}"
            )
        
        self.console.print(perf_table)


def main():
    """🧠 AI-Prognose Hauptfunktion"""
    predictor = CryptoAIPredictor()
    
    # Beispiel-Prognose für Bitcoin
    predictor.display_ai_analysis('bitcoin')


if __name__ == "__main__":
    if not ML_AVAILABLE:
        print("📦 Installiere ML-Dependencies:")
        print("pip install scikit-learn")
    else:
        main()
