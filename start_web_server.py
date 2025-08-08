#!/usr/bin/env python3
"""
🌐 Web Server Starter für Crypto AI
Autor: mad4cyber
Version: 1.0
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def start_server():
    """Starte Web Server"""
    print("🚀 Crypto AI Web Interface wird gestartet...")
    
    # Prüfe ob Virtual Environment existiert
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual Environment nicht gefunden!")
        print("💡 Führe zuerst aus: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    # Starte Flask Server
    try:
        print("🌐 Server startet auf http://localhost:5001")
        print("📊 Dashboard: http://localhost:5001/dashboard") 
        print("📈 Analyse: http://localhost:5001/analysis")
        print("\n⚠️  Drücke CTRL+C zum Beenden")
        print("=" * 50)
        
        # Browser öffnen nach kurzer Verzögerung
        import threading
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5001')
            except:
                pass
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Flask Server starten
        subprocess.run([
            "bash", "-c", 
            "source venv/bin/activate && python web_app.py"
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 Server beendet!")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Starten: {e}")
        return False

if __name__ == "__main__":
    start_server()