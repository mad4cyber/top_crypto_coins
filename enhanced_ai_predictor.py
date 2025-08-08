#!/usr/bin/env python3
"""
🧠 Enhanced AI Predictor mit verbesserter Genauigkeit
Autor: mad4cyber
Version: 2.0 - Enhanced Edition

🚀 VERBESSERUNGEN:
- Erweiterte Datenquellen (2 Jahre Historie)
- Zusätzliche technische Indikatoren
- Verbesserte Feature-Engineering
- Intelligentere Model-Gewichtung
- Confidence-basierte Filterung
- Multi-Timeframe Analyse
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
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler, RobustScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from pycoingecko import CoinGeckoAPI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import box

class EnhancedCryptoAIPredictor:
    """🧠 Verbesserte AI-Powered Kryptowährungs-Preisprognose"""
    
    def __init__(self):
        self.console = Console()
        self.cg = CoinGeckoAPI()
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def fetch_extended_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """📈 Erweiterte historische Daten (2 Jahre) für besseres Training"""
        try:
            # Hole mehr Daten für bessere Pattern-Erkennung
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='eur',
                days=days,  # 2 Jahre statt 1 Jahr
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
            
            # Zusätzliche Markt-Metriken
            df['price_volume_ratio'] = df['price'] / df['volume']
            df['market_cap_volume_ratio'] = df['market_cap'] / df['volume']
            
            return df
            
        except Exception as e:
            self.console.print(f"❌ [red]Fehler beim Abrufen historischer Daten: {e}[/red]")
            return pd.DataFrame()
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Erweiterte Features mit mehr technischen Indikatoren"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # === GRUNDLEGENDE FEATURES ===
        
        # Moving Averages (verschiedene Zeiträume)
        for period in [5, 7, 14, 21, 50, 100, 200]:
            if len(df) >= period:
                df[f'price_ma_{period}'] = df['price'].rolling(window=period).mean()
                df[f'price_std_{period}'] = df['price'].rolling(window=period).std()
                df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Preisänderungen (mehrere Zeiträume)
        for period in [1, 3, 7, 14, 30]:
            df[f'price_change_{period}d'] = df['price'].pct_change(period)
            df[f'volume_change_{period}d'] = df['volume'].pct_change(period)
        
        # === ERWEITERTE TECHNISCHE INDIKATOREN ===
        
        # RSI für verschiedene Perioden
        for period in [14, 21, 30]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (verschiedene Parameter)
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands (verschiedene Perioden und Std-Abweichungen)
        for period in [20, 50]:
            for std_mult in [1.5, 2, 2.5]:
                ma = df['price'].rolling(window=period).mean()
                std = df['price'].rolling(window=period).std()
                df[f'bb_upper_{period}_{std_mult}'] = ma + (std * std_mult)
                df[f'bb_lower_{period}_{std_mult}'] = ma - (std * std_mult)
                df[f'bb_position_{period}_{std_mult}'] = (df['price'] - df[f'bb_lower_{period}_{std_mult}']) / (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}'])
                df[f'bb_width_{period}_{std_mult}'] = (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}']) / ma
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_period = df['price'].rolling(window=period).min()
            high_period = df['price'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * ((df['price'] - low_period) / (high_period - low_period))
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Average True Range (ATR) - Volatilität
        high = df['price'].rolling(window=2).max()
        low = df['price'].rolling(window=2).min()
        high_close = np.abs(df['price'] - df['price'].shift(1))
        low_close = np.abs(low - df['price'].shift(1))
        true_range = np.maximum(high - low, np.maximum(high_close, low_close))
        
        for period in [14, 21, 30]:
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # Williams %R
        for period in [14, 21]:
            high_period = df['price'].rolling(window=period).max()
            low_period = df['price'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * ((high_period - df['price']) / (high_period - low_period))
        
        # === VOLUME-BASIERTE INDIKATOREN ===
        
        # On Balance Volume (OBV)
        df['price_direction'] = np.where(df['price'] > df['price'].shift(1), 1, 
                                np.where(df['price'] < df['price'].shift(1), -1, 0))
        df['obv'] = (df['volume'] * df['price_direction']).cumsum()
        df['obv_ma_10'] = df['obv'].rolling(window=10).mean()
        
        # Volume Rate of Change
        for period in [10, 20]:
            df[f'volume_roc_{period}'] = df['volume'].pct_change(period)
        
        # === MOMENTUM INDIKATOREN ===
        
        # Rate of Change (ROC)
        for period in [10, 20, 30]:
            df[f'price_roc_{period}'] = df['price'].pct_change(period)
        
        # Commodity Channel Index (CCI)
        for period in [20, 30]:
            typical_price = df['price']  # Vereinfacht da wir nur Close-Preise haben
            ma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (typical_price - ma_tp) / (0.015 * mean_deviation)
        
        # === VOLATILITÄTS-INDIKATOREN ===
        
        # Historische Volatilität
        for period in [10, 20, 30, 60]:
            log_returns = np.log(df['price'] / df['price'].shift(1))
            df[f'volatility_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(365)
        
        # === TREND-INDIKATOREN ===
        
        # ADX (Average Directional Index) - Simplified
        for period in [14, 21]:
            high = df['price'].rolling(window=2).max()
            low = df['price'].rolling(window=2).min()
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            df[f'plus_di_{period}'] = 100 * plus_dm.rolling(window=period).mean() / df[f'atr_{period}']
            df[f'minus_di_{period}'] = 100 * minus_dm.rolling(window=period).mean() / df[f'atr_{period}']
            df[f'dx_{period}'] = 100 * np.abs(df[f'plus_di_{period}'] - df[f'minus_di_{period}']) / (df[f'plus_di_{period}'] + df[f'minus_di_{period}'])
            df[f'adx_{period}'] = df[f'dx_{period}'].rolling(window=period).mean()
        
        # === ZEITBASIERTE FEATURES ===
        
        # Wochentag-Effekt
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Monatliche Zyklen
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # === TARGET VARIABLEN ===
        
        # Multiple Zeitrahmen für Prognosen
        for days in [1, 3, 7, 14, 30]:
            df[f'future_price_{days}d'] = df['price'].shift(-days)
            df[f'future_return_{days}d'] = df['price'].pct_change(-days)
        
        # === ZUSÄTZLICHE FEATURES ===
        
        # Price Position in Range
        for period in [50, 100, 200]:
            if len(df) >= period:
                period_high = df['price'].rolling(window=period).max()
                period_low = df['price'].rolling(window=period).min()
                df[f'price_position_{period}'] = (df['price'] - period_low) / (period_high - period_low)
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * df['price'].pct_change()).cumsum()
        df['vpt_ma'] = df['vpt'].rolling(window=10).mean()
        
        # Market Cap Momentum
        df['mcap_momentum'] = df['market_cap'].pct_change(10)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_enhanced_ml_models(self, df: pd.DataFrame, target_days: int = 1) -> Dict:
        """🤖 Erweiterte ML-Modelle mit besserer Validierung"""
        if not ML_AVAILABLE:
            self.console.print("❌ [red]Scikit-learn nicht verfügbar. Installiere mit: pip install scikit-learn[/red]")
            return {}
        
        target_col = f'future_price_{target_days}d'
        if target_col not in df.columns:
            self.console.print(f"❌ [red]Target-Spalte {target_col} nicht gefunden[/red]")
            return {}
        
        # Automatische Feature-Selektion (nur numerische Features)
        feature_cols = [col for col in df.columns 
                       if col not in ['price', 'volume', 'market_cap'] + [f'future_price_{d}d' for d in [1,3,7,14,30]] + [f'future_return_{d}d' for d in [1,3,7,14,30]]
                       and df[col].dtype in ['float64', 'int64']]
        
        # Nur Features mit ausreichend Daten
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() >= len(df) * 0.8:  # Mindestens 80% valide Daten
                valid_features.append(col)
        
        if len(valid_features) < 5:
            self.console.print("❌ [red]Nicht genügend valide Features für Training[/red]")
            return {}
        
        X = df[valid_features].values
        y = df[target_col].values
        
        # Time Series Split für bessere Validierung
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train/Test Split (zeitbasiert)
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Robuste Skalierung (weniger sensitiv auf Outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        performance = {}
        
        # 1. Random Forest (erweitert)
        rf_model = RandomForestRegressor(
            n_estimators=200,  # Mehr Trees
            max_depth=10,      # Kontrollierte Tiefe
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1         # Parallele Verarbeitung
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Cross-Validation für bessere Bewertung
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        
        models['random_forest'] = rf_model
        performance['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'feature_importance': dict(zip(valid_features, rf_model.feature_importances_))
        }
        
        # 2. Gradient Boosting (erweitert)
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,  # Langsameres Lernen
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,      # Subsampling für Regularization
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        
        models['gradient_boosting'] = gb_model
        performance['gradient_boosting'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred),
            'cv_r2_mean': gb_cv_scores.mean(),
            'cv_r2_std': gb_cv_scores.std(),
            'feature_importance': dict(zip(valid_features, gb_model.feature_importances_))
        }
        
        # 3. Ridge Regression (Regularisiert)
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        ridge_pred = ridge_model.predict(X_test_scaled)
        ridge_cv_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        
        models['ridge_regression'] = ridge_model
        performance['ridge_regression'] = {
            'mae': mean_absolute_error(y_test, ridge_pred),
            'r2': r2_score(y_test, ridge_pred),
            'cv_r2_mean': ridge_cv_scores.mean(),
            'cv_r2_std': ridge_cv_scores.std(),
            'feature_importance': dict(zip(valid_features, np.abs(ridge_model.coef_)))
        }
        
        # 4. XGBoost (wenn verfügbar)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
            
            models['xgboost'] = xgb_model
            performance['xgboost'] = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'r2': r2_score(y_test, xgb_pred),
                'cv_r2_mean': xgb_cv_scores.mean(),
                'cv_r2_std': xgb_cv_scores.std(),
                'feature_importance': dict(zip(valid_features, xgb_model.feature_importances_))
            }
        
        return {
            'models': models,
            'scaler': scaler,
            'performance': performance,
            'features': valid_features,
            'feature_count': len(valid_features)
        }
    
    def intelligent_ensemble_prediction(self, ml_results: Dict, current_features: np.ndarray) -> Dict:
        """🎯 Intelligente Ensemble-Prognose mit adaptiver Gewichtung"""
        if not ml_results or 'models' not in ml_results:
            return {'error': 'Keine ML-Ergebnisse verfügbar'}
        
        models = ml_results['models']
        performance = ml_results['performance']
        scaler = ml_results['scaler']
        
        current_features_scaled = scaler.transform(current_features)
        
        predictions = {}
        weights = {}
        confidence_scores = {}
        
        # Prognosen von allen Modellen
        for model_name, model in models.items():
            pred = model.predict(current_features_scaled)[0]
            predictions[model_name] = pred
            
            # Erweiterte Gewichtung basierend auf:
            perf = performance[model_name]
            r2_score = max(0, perf['r2'])  # Negative R² auf 0 setzen
            cv_r2 = max(0, perf['cv_r2_mean'])  # Cross-Validation R²
            cv_stability = 1 / (1 + perf['cv_r2_std'])  # Weniger Gewichtung bei instabilen Modellen
            
            # Kombinierte Gewichtung
            weight = (r2_score * 0.4 + cv_r2 * 0.4 + cv_stability * 0.2)
            weights[model_name] = weight
            confidence_scores[model_name] = cv_r2  # Verwende CV R² als Konfidenz
        
        # Normalisiere Gewichtungen
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Fallback: Equal weighting
            weights = {k: 1/len(models) for k in models.keys()}
        
        # Ensemble-Prognose
        ensemble_prediction = sum(predictions[model] * weights[model] for model in models.keys())
        
        # Gesamtkonfidenz (gewichteter Durchschnitt der CV R²-Werte)
        overall_confidence = sum(confidence_scores[model] * weights[model] for model in models.keys())
        
        # Confidence-Intervall basierend auf Model-Varianz
        pred_variance = np.var([predictions[model] for model in models.keys()])
        confidence_interval = 1.96 * np.sqrt(pred_variance)  # 95% CI
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'model_weights': weights,
            'overall_confidence': overall_confidence,
            'confidence_interval': confidence_interval,
            'model_count': len(models)
        }
    
    def predict_future_prices_enhanced(self, coin_id: str, days_ahead: int = 1) -> Dict:
        """🔮 Erweiterte Zukunftsprognose mit verbesserter Genauigkeit"""
        # Erweiterte historische Daten laden (1 Jahr - API Limit)
        df = self.fetch_extended_historical_data(coin_id, days=365)
        
        if df.empty:
            return {'error': 'Keine historischen Daten verfügbar'}
        
        # Erweiterte Features erstellen
        df_features = self.create_advanced_features(df)
        
        if df_features.empty:
            return {'error': 'Feature-Erstellung fehlgeschlagen'}
        
        # Erweiterte ML-Modelle trainieren
        ml_results = self.train_enhanced_ml_models(df_features, target_days=days_ahead)
        
        if not ml_results:
            return {'error': 'ML-Modelltraining fehlgeschlagen'}
        
        # Aktuelle Features für Prognose
        current_features = df_features.iloc[-1][ml_results['features']].values.reshape(1, -1)
        
        # Intelligente Ensemble-Prognose
        ensemble_result = self.intelligent_ensemble_prediction(ml_results, current_features)
        
        if 'error' in ensemble_result:
            return ensemble_result
        
        current_price = df['price'].iloc[-1]
        predicted_price = ensemble_result['ensemble_prediction']
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Qualitätsbewertung
        quality_score = self.assess_prediction_quality(ml_results, ensemble_result)
        
        return {
            'coin_id': coin_id,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pct': price_change_pct,
            'confidence': ensemble_result['overall_confidence'],
            'confidence_interval': ensemble_result['confidence_interval'],
            'quality_score': quality_score,
            'individual_predictions': ensemble_result['individual_predictions'],
            'model_weights': ensemble_result['model_weights'],
            'model_performance': ml_results['performance'],
            'feature_count': ml_results['feature_count'],
            'prediction_date': datetime.now().isoformat(),
            'target_date': (datetime.now() + timedelta(days=days_ahead)).isoformat(),
            'data_period_days': len(df),
            'recommendation': self.generate_recommendation(price_change_pct, ensemble_result['overall_confidence'], quality_score)
        }
    
    def assess_prediction_quality(self, ml_results: Dict, ensemble_result: Dict) -> Dict:
        """📊 Bewerte die Qualität der Prognose"""
        performance = ml_results['performance']
        
        # Durchschnittliche Cross-Validation Performance
        avg_cv_r2 = np.mean([perf['cv_r2_mean'] for perf in performance.values()])
        avg_cv_std = np.mean([perf['cv_r2_std'] for perf in performance.values()])
        
        # Model Agreement (wie ähnlich sind die Prognosen?)
        predictions = list(ensemble_result['individual_predictions'].values())
        pred_std = np.std(predictions) if len(predictions) > 1 else 0
        pred_mean = np.mean(predictions)
        coefficient_of_variation = pred_std / abs(pred_mean) if pred_mean != 0 else 1
        
        # Qualitäts-Score (0-1)
        quality_components = {
            'model_performance': max(0, avg_cv_r2),  # 0-1
            'model_stability': max(0, 1 - avg_cv_std),  # Höher = stabiler
            'model_agreement': max(0, 1 - coefficient_of_variation),  # Höher = mehr Agreement
            'feature_richness': min(1, ml_results['feature_count'] / 50),  # Normalized auf 50 Features
            'data_sufficiency': min(1, len(ml_results['features']) / 100)  # Normalisiert auf 100 Datenpunkte
        }
        
        # Gewichteter Quality Score
        weights = {
            'model_performance': 0.4,
            'model_stability': 0.2,
            'model_agreement': 0.2,
            'feature_richness': 0.1,
            'data_sufficiency': 0.1
        }
        
        overall_quality = sum(quality_components[k] * weights[k] for k in weights.keys())
        
        return {
            'overall_score': overall_quality,
            'components': quality_components,
            'interpretation': self.interpret_quality_score(overall_quality)
        }
    
    def interpret_quality_score(self, score: float) -> str:
        """📋 Interpretiere Quality Score"""
        if score >= 0.8:
            return "Sehr hoch - Prognose sehr vertrauenswürdig"
        elif score >= 0.6:
            return "Hoch - Prognose vertrauenswürdig"
        elif score >= 0.4:
            return "Mittel - Prognose mit Vorsicht verwenden"
        elif score >= 0.2:
            return "Niedrig - Prognose unsicher"
        else:
            return "Sehr niedrig - Prognose nicht empfohlen"
    
    def generate_recommendation(self, price_change_pct: float, confidence: float, quality_score: Dict) -> str:
        """💡 Generiere Trading-Empfehlung basierend auf AI-Analyse"""
        overall_quality = quality_score['overall_score']
        
        # Nur bei hoher Qualität und Konfidenz Empfehlungen geben
        if overall_quality < 0.4 or confidence < 0.3:
            return "❓ NEUTRAL - Unzureichende Datenqualität für verlässliche Empfehlung"
        
        if price_change_pct > 10 and confidence > 0.7 and overall_quality > 0.6:
            return "🚀 STRONG BUY - Hohe Wahrscheinlichkeit für starke Gewinne"
        elif price_change_pct > 5 and confidence > 0.6:
            return "🟢 BUY - Positive Prognose mit guter Konfidenz"
        elif price_change_pct > 2 and confidence > 0.5:
            return "📈 WEAK BUY - Leicht positive Prognose"
        elif -2 <= price_change_pct <= 2:
            return "⚪ HOLD - Seitwärtsbewegung erwartet"
        elif price_change_pct < -5 and confidence > 0.6:
            return "🔴 SELL - Negative Prognose mit guter Konfidenz"
        elif price_change_pct < -10 and confidence > 0.7 and overall_quality > 0.6:
            return "💥 STRONG SELL - Hohe Wahrscheinlichkeit für starke Verluste"
        else:
            return "❓ NEUTRAL - Unklare Signale, weitere Analyse empfohlen"

# Test der erweiterten AI
def main():
    """🧠 Test der erweiterten AI-Funktionen"""
    predictor = EnhancedCryptoAIPredictor()
    
    # Test mit Bitcoin
    print("🧠 Teste erweiterte AI mit Bitcoin...")
    result = predictor.predict_future_prices_enhanced('bitcoin')
    
    if 'error' not in result:
        print(f"\n📊 Ergebnisse:")
        print(f"Aktuell: €{result['current_price']:.2f}")
        print(f"Prognose: €{result['predicted_price']:.2f}")
        print(f"Änderung: {result['price_change_pct']:+.2f}%")
        print(f"Konfidenz: {result['confidence']:.1%}")
        print(f"Qualität: {result['quality_score']['overall_score']:.1%}")
        print(f"Features: {result['feature_count']}")
        print(f"Empfehlung: {result['recommendation']}")
    else:
        print(f"❌ Fehler: {result['error']}")

if __name__ == "__main__":
    main()