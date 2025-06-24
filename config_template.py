#!/usr/bin/env python3
"""
🔐 Binance API-Key Konfiguration Template
Autor: mad4cyber
Version: 1.0

⚠️ WICHTIG: 
- Kopiere diese Datei zu 'config.py'
- Fülle deine echten API-Keys ein
- Füge 'config.py' zu .gitignore hinzu!
"""

# Binance Testnet API-Keys
# Besorge dir diese von: https://testnet.binance.vision/
BINANCE_TESTNET_API_KEY = "dein_testnet_api_key_hier"
BINANCE_TESTNET_SECRET = "dein_testnet_secret_hier"

# Binance Live API-Keys (NUR für echtes Trading!)
# NIEMALS diese Keys committen oder teilen!
BINANCE_LIVE_API_KEY = "dein_live_api_key_hier"  # VORSICHT!
BINANCE_LIVE_SECRET = "dein_live_secret_hier"    # VORSICHT!

# Trading-Einstellungen
INITIAL_BALANCE = 10000.0  # Startkapital
MAX_POSITIONS = 5          # Maximale gleichzeitige Positionen
RISK_PER_TRADE = 0.02     # 2% Risiko pro Trade
MIN_CONFIDENCE = 0.75      # 75% Mindest-AI-Konfidenz

# Überwachte Trading-Paare (Top 10 nach Volumen)
TRADING_SYMBOLS = [
    # Top Volume Pairs
    'ETHUSDT',      # #1 - Ethereum
    'BTCUSDT',      # #2 - Bitcoin
    'AAVEUSDT',     # #3 - Aave
    'SUIUSDT',      # #4 - Sui
    'XRPUSDT',      # #5 - Ripple
    'SOLUSDT',      # #6 - Solana
    'VIRTUALUSDT',  # #7 - Virtual
    'HBARUSDT',     # #8 - Hedera
    'BNBUSDT',      # #9 - Binance Coin
    'FUNUSDT'       # #10 - FunToken
]

# Alternative: Klassische Top 10 Market Cap
TRADING_SYMBOLS_CLASSIC = [
    'BTCUSDT',      # Bitcoin
    'ETHUSDT',      # Ethereum
    'XRPUSDT',      # Ripple
    'BNBUSDT',      # Binance Coin
    'SOLUSDT',      # Solana
    'ADAUSDT',      # Cardano
    'DOGEUSDT',     # Dogecoin
    'TRXUSDT',      # Tron
    'AVAXUSDT',     # Avalanche
    'LINKUSDT'      # Chainlink
]

# Sicherheitscheck
import os
if os.path.basename(__file__) == 'config.py':
    print("⚠️ WARNUNG: config.py sollte NIEMALS committet werden!")
    print("💡 Füge 'config.py' zu .gitignore hinzu!")
