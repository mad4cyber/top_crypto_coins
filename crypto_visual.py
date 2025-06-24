#!/usr/bin/env python3
"""
🎨 Visuell verbesserte Kryptowährungs-Analyse v2.5
Autor: mad4cyber
Version: 2.5 - Visual Enhancement
"""

import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
from pycoingecko import CoinGeckoAPI

# Rich für bessere Terminal-Ausgabe
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich.layout import Layout
    from rich.tree import Tree
    from rich import box
    from rich.prompt import Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📦 Für optimale Darstellung installiere: pip install rich")

class VisualCryptoAnalyzer:
    """🎨 Visuell verbesserte Kryptowährungs-Analyse"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.cg = CoinGeckoAPI()
        self.cache = {}
        self.cache_ttl = 300
        self.language = os.environ.get('CRYPTO_LANG', 'de')
        
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
    
    def create_loading_animation(self, message: str):
        """🔄 Lade-Animation erstellen"""
        if not RICH_AVAILABLE:
            print(f"🔄 {message}")
            return None
            
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        )
    
    def get_crypto_data_with_animation(self, num: int = 10, currency: str = "eur") -> pd.DataFrame:
        """📊 Kryptodaten mit Animation laden"""
        cache_key = f"crypto_data_{num}_{currency}"
        
        # Cache prüfen
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            if RICH_AVAILABLE:
                self.console.print("⚡ [green]Verwende gecachte Daten[/green]")
            return cached_data
        
        # Mit Animation laden
        loading_msg = "Lade Kryptowährungsdaten..." if self.language == "de" else "Loading cryptocurrency data..."
        
        if RICH_AVAILABLE:
            with self.create_loading_animation(loading_msg) as progress:
                task = progress.add_task(loading_msg, total=100)
                
                try:
                    progress.update(task, advance=30)
                    coins = self.cg.get_coins_markets(
                        vs_currency=currency, 
                        order="market_cap", 
                        per_page=num,
                        page=1, 
                        sparkline=False, 
                        price_change_percentage='24h,7d,30d'
                    )
                    progress.update(task, advance=40)
                    
                    df = pd.DataFrame(coins)
                    progress.update(task, advance=20)
                    
                    # Spalten anpassen
                    df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 
                            'market_cap', 'price_change_percentage_24h_in_currency',
                            'price_change_percentage_7d_in_currency',
                            'price_change_percentage_30d_in_currency']]
                    
                    df.columns = ['id', 'symbol', 'name', 'rang', 'preis', 'marktkapitalisierung',
                                 '24h_änderung', '7d_änderung', '30d_änderung']
                    df['symbol'] = df['symbol'].str.upper()
                    progress.update(task, advance=10)
                    
                    self._set_cache(cache_key, df)
                    return df
                    
                except Exception as e:
                    self.console.print(f"❌ [red]Fehler: {e}[/red]")
                    return pd.DataFrame()
        else:
            # Fallback ohne Rich
            try:
                print(f"🔄 {loading_msg}")
                coins = self.cg.get_coins_markets(
                    vs_currency=currency, order="market_cap", per_page=num,
                    page=1, sparkline=False, price_change_percentage='24h,7d,30d'
                )
                df = pd.DataFrame(coins)
                df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 
                        'market_cap', 'price_change_percentage_24h_in_currency',
                        'price_change_percentage_7d_in_currency',
                        'price_change_percentage_30d_in_currency']]
                df.columns = ['id', 'symbol', 'name', 'rang', 'preis', 'marktkapitalisierung',
                             '24h_änderung', '7d_änderung', '30d_änderung']
                df['symbol'] = df['symbol'].str.upper()
                self._set_cache(cache_key, df)
                return df
            except Exception as e:
                print(f"❌ Fehler: {e}")
                return pd.DataFrame()
    
    def format_currency(self, value: float, currency: str = "eur") -> str:
        """💰 Währung formatieren"""
        symbols = {"eur": "€", "usd": "$", "gbp": "£", "chf": "CHF"}
        symbol = symbols.get(currency.lower(), currency.upper())
        
        if value >= 1e9:
            return f"{symbol}{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{symbol}{value/1e6:.2f}M"
        elif value >= 1000:
            return f"{symbol}{value:,.0f}"
        else:
            return f"{symbol}{value:.2f}"
    
    def get_change_color_and_icon(self, change: float) -> tuple:
        """🎨 Farbe und Icon für Preisänderung"""
        if change > 5:
            return "bright_green", "🚀"
        elif change > 0:
            return "green", "📈"
        elif change > -5:
            return "red", "📉"
        else:
            return "bright_red", "💥"
    
    def create_beautiful_table(self, df: pd.DataFrame, currency: str = "eur", compact: bool = False) -> Table:
        """🎨 Schöne Tabelle erstellen"""
        if not RICH_AVAILABLE:
            return None
            
        # Tabelle konfigurieren
        table = Table(
            title="🏆 Top Kryptowährungen",
            title_style="bold magenta",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            row_styles=["none", "dim"]
        )
        
        # Spalten hinzufügen
        table.add_column("🏅 Rang", justify="center", style="yellow", width=6)
        table.add_column("🪙 Symbol", justify="center", style="bold blue", width=8)
        table.add_column("📛 Name", justify="left", style="white", width=15)
        table.add_column("💰 Preis", justify="right", style="green", width=12)
        
        if not compact:
            table.add_column("📊 Marktkapitalisierung", justify="right", style="cyan", width=15)
            table.add_column("📈 24h", justify="center", width=8)
            table.add_column("📊 7d", justify="center", width=8)
            table.add_column("📉 30d", justify="center", width=8)
        else:
            table.add_column("📊 Marktkapitalisierung", justify="right", style="cyan", width=15)
            table.add_column("📈 24h", justify="center", width=10)
        
        # Daten hinzufügen
        for _, row in df.iterrows():
            rang = f"#{int(row['rang'])}"
            symbol = row['symbol']
            name = row['name'][:12] + "..." if len(row['name']) > 15 else row['name']
            preis = self.format_currency(row['preis'], currency)
            marktcap = self.format_currency(row['marktkapitalisierung'], currency)
            
            # 24h Änderung mit Farbe
            change_24h = row['24h_änderung']
            color_24h, icon_24h = self.get_change_color_and_icon(change_24h)
            change_24h_text = f"{icon_24h} {change_24h:+.1f}%"
            
            if compact:
                table.add_row(
                    rang, symbol, name, preis, marktcap,
                    f"[{color_24h}]{change_24h_text}[/{color_24h}]"
                )
            else:
                # 7d und 30d Änderungen
                change_7d = row['7d_änderung'] 
                change_30d = row['30d_änderung']
                
                color_7d, icon_7d = self.get_change_color_and_icon(change_7d)
                color_30d, icon_30d = self.get_change_color_and_icon(change_30d)
                
                change_7d_text = f"{icon_7d} {change_7d:+.1f}%"
                change_30d_text = f"{icon_30d} {change_30d:+.1f}%"
                
                table.add_row(
                    rang, symbol, name, preis, marktcap,
                    f"[{color_24h}]{change_24h_text}[/{color_24h}]",
                    f"[{color_7d}]{change_7d_text}[/{color_7d}]",
                    f"[{color_30d}]{change_30d_text}[/{color_30d}]"
                )
        
        return table
    
    def create_header_panel(self) -> Panel:
        """🎨 Header-Panel erstellen"""
        if not RICH_AVAILABLE:
            return None
            
        header_text = Text()
        header_text.append("🚀 KRYPTO-ANALYSE ", style="bold magenta")
        header_text.append("v2.5", style="bold cyan")
        header_text.append(" 🎨 Visual Edition", style="bold yellow")
        
        return Panel(
            Align.center(header_text),
            box=box.DOUBLE_EDGE,
            border_style="magenta",
            padding=(1, 0)
        )
    
    def create_footer_panel(self) -> Panel:
        """🎨 Footer-Panel erstellen"""
        if not RICH_AVAILABLE:
            return None
            
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        footer_text = Text()
        footer_text.append("📅 Daten abgerufen am: ", style="dim")
        footer_text.append(timestamp, style="bold green")
        footer_text.append(" | 🔄 Cache: 5 Min", style="dim")
        footer_text.append(" | 🌐 CoinGecko API", style="dim")
        
        return Panel(
            Align.center(footer_text),
            box=box.ROUNDED,
            border_style="green",
            padding=(0, 0)
        )
    
    def create_summary_panels(self, df: pd.DataFrame, currency: str = "eur") -> List[Panel]:
        """📊 Zusammenfassung-Panels erstellen"""
        if not RICH_AVAILABLE or df.empty:
            return []
        
        panels = []
        
        # Markt-Statistiken
        total_market_cap = df['marktkapitalisierung'].sum()
        avg_24h_change = df['24h_änderung'].mean()
        top_gainer = df.loc[df['24h_änderung'].idxmax()]
        top_loser = df.loc[df['24h_änderung'].idxmin()]
        
        # Markt-Panel
        market_text = Text()
        market_text.append("📊 MARKT-ÜBERSICHT\n\n", style="bold cyan")
        market_text.append(f"💰 Gesamt-Marktkapitalisierung: ", style="white")
        market_text.append(f"{self.format_currency(total_market_cap, currency)}\n", style="bold green")
        market_text.append(f"📈 Durchschnittliche 24h-Änderung: ", style="white")
        
        avg_color = "green" if avg_24h_change > 0 else "red"
        market_text.append(f"{avg_24h_change:+.2f}%", style=f"bold {avg_color}")
        
        market_panel = Panel(
            market_text,
            title="📊 Markt",
            border_style="cyan",
            box=box.ROUNDED
        )
        panels.append(market_panel)
        
        # Top Gewinner Panel
        gainer_text = Text()
        gainer_text.append("🚀 TOP GEWINNER\n\n", style="bold green")
        gainer_text.append(f"🪙 {top_gainer['symbol']}\n", style="bold white")
        gainer_text.append(f"📛 {top_gainer['name']}\n", style="white")
        gainer_text.append(f"📈 {top_gainer['24h_änderung']:+.2f}%", style="bold green")
        
        gainer_panel = Panel(
            gainer_text,
            title="🚀 Gewinner",
            border_style="green",
            box=box.ROUNDED
        )
        panels.append(gainer_panel)
        
        # Top Verlierer Panel
        loser_text = Text()
        loser_text.append("💥 TOP VERLIERER\n\n", style="bold red")
        loser_text.append(f"🪙 {top_loser['symbol']}\n", style="bold white")
        loser_text.append(f"📛 {top_loser['name']}\n", style="white")
        loser_text.append(f"📉 {top_loser['24h_änderung']:+.2f}%", style="bold red")
        
        loser_panel = Panel(
            loser_text,
            title="💥 Verlierer",
            border_style="red",
            box=box.ROUNDED
        )
        panels.append(loser_panel)
        
        return panels
    
    def display_crypto_analysis(self, num: int = 10, currency: str = "eur", compact: bool = False):
        """🎨 Vollständige Krypto-Analyse anzeigen"""
        if not RICH_AVAILABLE:
            print("⚠️  Rich nicht verfügbar. Installiere mit: pip install rich")
            return
        
        # Header anzeigen
        self.console.print(self.create_header_panel())
        self.console.print()
        
        # Daten laden
        df = self.get_crypto_data_with_animation(num, currency)
        
        if df.empty:
            self.console.print("❌ [red]Keine Daten verfügbar[/red]")
            return
        
        # Zusammenfassung anzeigen (nur bei nicht-kompakter Ansicht)
        if not compact:
            summary_panels = self.create_summary_panels(df, currency)
            if summary_panels:
                self.console.print(Columns(summary_panels))
                self.console.print()
        
        # Haupttabelle anzeigen
        table = self.create_beautiful_table(df, currency, compact)
        if table:
            self.console.print(table)
        
        self.console.print()
        
        # Footer anzeigen
        self.console.print(self.create_footer_panel())
    
    def create_interactive_menu(self):
        """🎯 Interaktives Menü"""
        if not RICH_AVAILABLE:
            print("📱 Interaktives Menü benötigt Rich-Bibliothek")
            return
        
        menu_text = Text()
        menu_text.append("🎯 INTERAKTIVES MENÜ\n\n", style="bold yellow")
        menu_text.append("1️⃣  Top 10 Kryptowährungen (Kompakt)\n", style="cyan")
        menu_text.append("2️⃣  Top 20 Kryptowährungen (Detailliert)\n", style="cyan")
        menu_text.append("3️⃣  USD-Preise anzeigen\n", style="cyan")
        menu_text.append("4️⃣  Portfolio anzeigen\n", style="cyan")
        menu_text.append("5️⃣  Beenden\n", style="red")
        
        menu_panel = Panel(
            menu_text,
            title="🎯 Optionen",
            border_style="yellow",
            box=box.DOUBLE_EDGE
        )
        
        self.console.print(menu_panel)
        
        while True:
            choice = self.console.input("\n🔢 [bold cyan]Wähle eine Option (1-5):[/bold cyan] ")
            
            if choice == "1":
                self.display_crypto_analysis(10, "eur", compact=True)
            elif choice == "2":
                self.display_crypto_analysis(20, "eur", compact=False)
            elif choice == "3":
                self.display_crypto_analysis(10, "usd", compact=False)
            elif choice == "4":
                self.console.print("📝 [yellow]Portfolio-Feature coming soon![/yellow]")
            elif choice == "5":
                self.console.print("👋 [green]Auf Wiedersehen![/green]")
                break
            else:
                self.console.print("❌ [red]Ungültige Auswahl![/red]")
            
            if choice in ["1", "2", "3", "4"]:
                continue_choice = Confirm.ask("\n🔄 Möchtest du fortfahren?")
                if not continue_choice:
                    break


def create_visual_cli():
    """🎨 Visual CLI erstellen"""
    parser = argparse.ArgumentParser(
        description="🎨 Visuell verbesserte Kryptowährungs-Analyse v2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎨 Visual Examples:
  %(prog)s                           # Interaktives Menü
  %(prog)s -n 10 --compact          # Top 10 kompakt
  %(prog)s -n 20                    # Top 20 detailliert
  %(prog)s --currency usd           # USD-Preise
        """
    )
    
    parser.add_argument("-n", "--num", type=int, default=10,
                       help="Anzahl der Kryptowährungen")
    parser.add_argument("-c", "--currency", type=str, default="eur",
                       help="Währung (eur, usd, gbp, chf)")
    parser.add_argument("--compact", action="store_true",
                       help="Kompakte Anzeige")
    parser.add_argument("--interactive", action="store_true",
                       help="Interaktives Menü")
    
    return parser


def main():
    """🎨 Hauptfunktion der visuellen Version"""
    parser = create_visual_cli()
    args = parser.parse_args()
    
    analyzer = VisualCryptoAnalyzer()
    
    # Interaktives Menü oder direkte Anzeige
    if args.interactive or (not any([args.num != 10, args.currency != "eur", args.compact])):
        analyzer.create_interactive_menu()
    else:
        analyzer.display_crypto_analysis(args.num, args.currency, args.compact)


if __name__ == "__main__":
    main()
