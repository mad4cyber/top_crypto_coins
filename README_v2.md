# 🚀 Erweiterte Kryptowährungs-Analyse v2.0

Ein professionelles Python-Tool zur Analyse von Kryptowährungen mit Portfolio-Management, Preisalarmen und erweiterten Features.

## ✨ Features

### 🔥 Neue Features v2.0
- **📊 Portfolio-Management**: Verfolge deine eigenen Kryptowährungen
- **🚨 Erweiterte Preisalarme**: Intelligente Benachrichtigungen
- **⚡ Caching-System**: Bessere Performance durch lokales Caching
- **🎨 Farbige Ausgabe**: Visuelle Hervorhebung von Gewinn/Verlust
- **📝 Professionelles Logging**: Vollständige Aktivitäts-Protokollierung
- **⚙️ YAML-Konfiguration**: Flexible Konfigurationsmöglichkeiten
- **🏗️ Klassen-basierte Architektur**: Sauberer, wartbarer Code

### 📈 Grundfunktionen
- Top Kryptowährungsdaten von CoinGecko API
- Mehrere Währungen (EUR, USD, GBP, CHF)
- Export in CSV, JSON, Excel
- Kompakte und detaillierte Ansichten
- Deutsche und englische Lokalisierung
- Handelsvolumen und Umlaufmenge

## 🚀 Installation

1. **Repository klonen:**
```bash
git clone https://github.com/mad4cyber/top_crypto_coins.git
cd top_crypto_coins
```

2. **Virtual Environment erstellen:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

## 💻 Verwendung

### 📊 Basis-Kryptowährungsanalyse
```bash
# Standard-Anzeige (Top 10 in EUR)
python3 crypto_analyzer.py

# Top 5 Coins kompakt
python3 crypto_analyzer.py -n 5 --compact

# Mit Handelsvolumen und Umlaufmenge
python3 crypto_analyzer.py --show-volume --show-supply

# Verschiedene Währungen
python3 crypto_analyzer.py --currency usd
python3 crypto_analyzer.py --currency gbp
```

### 💼 Portfolio-Management
```bash
# Bitcoin zum Portfolio hinzufügen
python3 crypto_analyzer.py --add-to-portfolio BTC 0.1 85000

# Ethereum hinzufügen
python3 crypto_analyzer.py --add-to-portfolio ETH 2.5 2800

# Portfolio anzeigen
python3 crypto_analyzer.py --portfolio
```

**Portfolio-Beispiel:**
```
====================================================================================================
                                       📊 Portfolio Übersicht                                        
====================================================================================================
+----------+---------+---------+--------+--------------+--------------------+------------------+
| Symbol   |   Menge | Preis   | Wert   | Investiert   | Gewinn/Verlust %   | Gewinn/Verlust   |
+==========+=========+=========+========+==============+====================+==================+
| BTC      |     0.1 | €90,572 | €9,057 | €8,500       | +6.56%             | €557.20          |
+----------+---------+---------+--------+--------------+--------------------+------------------+
| ETH      |     2   | €2,073  | €4,146 | €5,000       | -17.07%            | €-853.68         |
+----------+---------+---------+--------+--------------+--------------------+------------------+

💰 Gesamtwert: €13,204
💸 Investiert: €13,500
📈 Gewinn/Verlust: €-296.48 (-2.20%)
```

### 🚨 Preisalarme
```bash
# Einzelner Alarm
python3 crypto_analyzer.py --price-alert BTC:100000:above

# Mehrere Alarme
python3 crypto_analyzer.py --price-alert BTC:90000:above ETH:3000:below SOL:200:above

# Alarme mit Anzeige kombinieren
python3 crypto_analyzer.py --price-alert DOGE:0.50:above --compact
```

### 📤 Daten-Export
```bash
# CSV-Export
python3 crypto_analyzer.py --export csv

# JSON-Export mit USD-Preisen
python3 crypto_analyzer.py --export json --currency usd

# Excel-Export (benötigt openpyxl)
python3 crypto_analyzer.py --export excel
```

### ⚙️ Erweiterte Optionen
```bash
# Hilfe anzeigen
python3 crypto_analyzer.py --help

# Eigene Konfigurationsdatei
python3 crypto_analyzer.py --config my_config.yaml

# Vollständige Analyse mit allen Features
python3 crypto_analyzer.py -n 20 --show-volume --show-supply --price-alert BTC:95000:above
```

## 🔧 Konfiguration

### Umgebungsvariablen
```bash
# Sprache festlegen
export CRYPTO_LANG=de  # oder 'en' für Englisch

# Script ausführen
python3 crypto_analyzer.py
```

### YAML-Konfiguration
Erstelle eine `config.yaml` für benutzerdefinierte Einstellungen:

```yaml
defaults:
  currency: "eur"
  language: "de"
  num_coins: 15
  table_format: "grid"

api:
  coingecko:
    timeout: 30
    retries: 3
```

## 📁 Projektstruktur

```
top_crypto_coins/
├── crypto_analyzer.py      # 🆕 Erweiterte v2.0 Version
├── top_coins.py            # Original v1.0 Version
├── config.yaml             # 🆕 Konfigurationsdatei
├── portfolio.json          # 🆕 Portfolio-Daten
├── requirements.txt        # Dependencies
├── README.md              # Diese Datei
├── SECURITY.md            # Sicherheitsrichtlinien
└── venv/                  # Virtual Environment
```

## 🎯 Anwendungsfälle

### 👤 Für Privatanleger
- Portfolio-Tracking mit Gewinn/Verlust-Berechnung
- Preisalarme für wichtige Schwellenwerte
- Regelmäßige Marktübersicht

### 📊 Für Analysten
- Export von Marktdaten
- Historische Preis-Tracking
- Automatisierte Berichte

### 🤖 Für Entwickler
- API-Integration
- Datenanalyse-Pipeline
- Automatisierte Trading-Signale

## 🔒 Sicherheit

- Alle Dependencies auf neueste sichere Versionen
- Keine API-Keys erforderlich (CoinGecko public API)
- Lokale Datenspeicherung
- Siehe [SECURITY.md](SECURITY.md) für Details

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Committe deine Änderungen
4. Push zum Branch
5. Erstelle einen Pull Request

## 📜 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details

## 🙏 Credits

- **API**: [CoinGecko](https://www.coingecko.com/api)
- **Autor**: mad4cyber
- **Version**: 2.0

---

**⭐ Wenn dir dieses Projekt gefällt, gib ihm einen Stern auf GitHub!**
