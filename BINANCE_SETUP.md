# 🔐 Binance Testnet API-Keys Setup

## 📋 Schritt-für-Schritt Anleitung

### **1. Binance Testnet Account erstellen**
- Gehe zu: https://testnet.binance.vision/
- Klicke auf **"Register"** (Registrieren)
- Erstelle einen Account mit Email/Passwort
- Bestätige deine Email

### **2. API-Keys generieren**
- Logge dich in dein Testnet-Account ein
- Gehe zu **"API Management"** (API-Verwaltung)
- Klicke **"Create API"** (API erstellen)
- Gib einen Namen ein (z.B. "Trading Bot")
- **Aktiviere folgende Berechtigungen:**
  - ✅ **Spot & Margin Trading**
  - ✅ **Futures Trading** (optional)
  - ❌ **Withdraw** (NICHT aktivieren!)
- Speichere **API Key** und **Secret Key** sicher!

### **3. Lokale Konfiguration**

```bash
# 1. Kopiere Template
cp config_template.py config.py

# 2. Bearbeite config.py
nano config.py  # oder mit deinem Editor
```

### **4. API-Keys eintragen**

Öffne `config.py` und fülle aus:

```python
# Binance Testnet API-Keys
BINANCE_TESTNET_API_KEY = "dein_echter_testnet_api_key"
BINANCE_TESTNET_SECRET = "dein_echter_testnet_secret"
```

### **5. Testnet-Guthaben erhalten**

- Gehe zu **"Faucet"** im Testnet
- Fordere kostenloses Testgeld an:
  - BTC: 1.0 BTC
  - USDT: 100,000 USDT
  - ETH: 100 ETH
- Das Geld ist NICHT echt - nur für Tests!

## 🧪 **Testing**

### **Basis-Test durchführen:**

```bash
# Trading Bot mit Setup-Check starten
python3 secure_trading_bot.py
```

### **Erwartete Ausgabe:**
```
✅ Konfiguration geladen
✅ API-Keys konfiguriert
✅ Binance Library verfügbar
✅ Binance Testnet verbunden!
📊 Account Status: SPOT
```

## ⚠️ **Sicherheit**

### **WICHTIG - Niemals committen:**
- `config.py` ist in `.gitignore`
- **NIEMALS** echte API-Keys teilen
- **NIEMALS** Live-API-Keys für Tests verwenden

### **API-Key Berechtigungen:**
- ✅ **Reading** (Lesen)
- ✅ **Spot Trading** (Handel)
- ❌ **Withdrawal** (Abhebung) - NIEMALS!
- ❌ **Futures** (nur wenn benötigt)

### **IP-Whitelist (Empfohlen):**
- Beschränke API-Keys auf deine IP
- Gehe zu API-Management → Edit → IP Access Restriction

## 🔧 **Troubleshooting**

### **Fehler: "Invalid API Key"**
```bash
# Prüfe API-Key Format
echo $BINANCE_TESTNET_API_KEY
# Sollte etwa so aussehen: vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A
```

### **Fehler: "Signature verification failed"**
```bash
# Prüfe Secret Key
# Stelle sicher, dass keine Leerzeichen/Zeilenumbrüche vorhanden sind
```

### **Fehler: "IP not in whitelist"**
```bash
# Deaktiviere IP-Whitelist temporär oder füge deine IP hinzu
```

## 📊 **Test-Strategien**

### **1. Minimaler Test:**
- Kleine Order: 0.001 BTC
- Market Order (sofortige Ausführung)
- Testnet-Geld verwenden

### **2. Paper Trading parallel:**
- Vergleiche Testnet vs. Paper Trading
- Gleiche Signale, verschiedene Ausführung

### **3. Performance-Vergleich:**
- API-Latenz messen
- Order-Ausführungszeiten
- Slippage analysieren

## 🚀 **Nächste Schritte**

Nach erfolgreichem Testnet-Setup:

1. **Automated Trading:** Vollautomatischer Bot
2. **Risk Management:** Stop-Loss/Take-Profit
3. **Portfolio-Rebalancing:** Multi-Asset Strategien
4. **Live Trading:** Erst nach ausgiebigen Tests!

---

## 📞 **Support**

Bei Problemen:
- Prüfe zunächst diese Anleitung
- Teste API-Keys direkt auf Binance Testnet
- Überprüfe Internetverbindung
- Schaue in die Bot-Logs (`*.log` Dateien)

**WICHTIG:** Verwende NIEMALS echte API-Keys für Tests!
