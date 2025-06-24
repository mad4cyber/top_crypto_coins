# Sicherheitsrichtlinien

## Unterstützte Versionen

| Version | Unterstützt          |
| ------- | -------------------- |
| 1.0.x   | :white_check_mark:   |

## Sicherheitsprüfungen

### Letzte Überprüfung: 24.06.2025

✅ **Alle Dependencies sind sicher und aktuell:**
- `requests >= 2.32.4` (behebt CVE-2024-35195, CVE-2023-32681)
- `pandas >= 2.3.0`
- `pycoingecko >= 3.2.0`
- `tabulate >= 0.9.0`

### Automatische Sicherheitsprüfungen

Das Projekt verwendet:
- GitHub Dependabot für automatische Sicherheitsupdates
- `safety` für lokale Vulnerability-Scans

### Lokale Sicherheitsprüfung

```bash
# Virtual Environment aktivieren
source venv/bin/activate

# Sicherheitsscan durchführen
pip install safety
safety scan
```

## Sicherheitsprobleme melden

Falls Sie Sicherheitsprobleme entdecken:

1. **Erstellen Sie KEIN öffentliches Issue**
2. Kontaktieren Sie direkt: office@strength-coach.at
3. Beschreiben Sie das Problem detailliert
4. Warten Sie auf Bestätigung bevor Sie Details veröffentlichen

## Beste Praktiken

### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Umgebungsvariablen
- Keine API-Keys in den Code einbetten
- `CRYPTO_LANG` für Spracheinstellungen verwenden

### Updates
```bash
# Dependencies regelmäßig aktualisieren
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```
