#!/usr/bin/env python3
"""
📊 Kryptowährungs-Analyse mit Charts und Grafiken v2.6
Autor: mad4cyber
Version: 2.6 - Charts Edition
"""

import argparse
import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from pycoingecko import CoinGeckoAPI

# Rich für Terminal-Ausgabe
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.columns import Columns
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Plotext für Terminal-Charts
try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False


class ChartCryptoAnalyzer:
    """📊 Krypto-Analyzer mit Charts und Grafiken"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.cg = CoinGeckoAPI()
        self.language = os.environ.get('CRYPTO_LANG', 'de')
    
    def create_ascii_chart(self, values: List[float], labels: List[str], title: str = "Chart") -> str:
        """📊 ASCII-Chart erstellen"""
        if not values:
            return "Keine Daten verfügbar"
        
        # Normalisierung für ASCII-Darstellung
        max_val = max(values)
        min_val = min(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Chart-Breite
        chart_width = 50
        chart_height = 10
        
        chart_lines = []
        chart_lines.append(f"📊 {title}")
        chart_lines.append("─" * (chart_width + 10))
        
        # Normalisierte Werte für Balken
        normalized = [(val - min_val) / range_val * chart_width for val in values]
        
        for i, (label, norm_val, orig_val) in enumerate(zip(labels, normalized, values)):
            bar = "█" * int(norm_val) + "░" * (chart_width - int(norm_val))
            color_indicator = "🟢" if orig_val > 0 else "🔴" if orig_val < 0 else "⚫"
            chart_lines.append(f"{label[:8]:8} │{bar}│ {color_indicator} {orig_val:+.1f}%")
        
        chart_lines.append("─" * (chart_width + 10))
        return "\\n".join(chart_lines)
    
    def create_terminal_chart(self, df: pd.DataFrame, chart_type: str = "bar") -> str:
        """📊 Terminal-Chart mit plotext erstellen"""
        if not PLOTEXT_AVAILABLE:
            # Fallback zu ASCII-Chart
            return self.create_ascii_chart(
                df['24h_änderung'].tolist(),
                df['symbol'].tolist(),
                "24h Preisänderungen"
            )
        
        # plotext Chart erstellen
        plt.clear_figure()
        
        if chart_type == "bar":
            symbols = df['symbol'].tolist()[:10]  # Top 10
            changes = df['24h_änderung'].tolist()[:10]
            
            # Farben basierend auf Werten
            colors = ['green' if x > 0 else 'red' for x in changes]
            
            plt.bar(symbols, changes, color=colors)
            plt.title("📊 24h Preisänderungen (%)")
            plt.xlabel("Kryptowährungen")
            plt.ylabel("Änderung (%)")
            
        elif chart_type == "scatter":
            market_caps = [x/1e9 for x in df['marktkapitalisierung'].tolist()[:10]]  # in Milliarden
            changes = df['24h_änderung'].tolist()[:10]
            
            plt.scatter(market_caps, changes)
            plt.title("📊 Marktkapitalisierung vs. 24h Änderung")
            plt.xlabel("Marktkapitalisierung (Milliarden €)")
            plt.ylabel("24h Änderung (%)")
        
        # Chart-Größe anpassen
        plt.plotsize(80, 20)
        
        # Als String zurückgeben
        return plt.build()
    
    def create_trend_visualization(self, df: pd.DataFrame) -> Panel:
        """📈 Trend-Visualisierung erstellen"""
        if not RICH_AVAILABLE:
            return None
        
        # Trend-Analyse
        bullish_count = len(df[df['24h_änderung'] > 0])
        bearish_count = len(df[df['24h_änderung'] < 0])
        total_count = len(df)
        
        bullish_pct = (bullish_count / total_count) * 100
        bearish_pct = (bearish_count / total_count) * 100
        
        # Visuelle Trend-Anzeige
        trend_text = Text()
        trend_text.append("📊 MARKT-TREND\\n\\n", style="bold yellow")
        
        # Bullish Bar
        bullish_bar_length = int(bullish_pct / 2)  # Max 50 Zeichen
        bearish_bar_length = int(bearish_pct / 2)
        
        trend_text.append("🟢 Bullish: ", style="green")
        trend_text.append("█" * bullish_bar_length, style="green")
        trend_text.append(f" {bullish_pct:.1f}% ({bullish_count} Coins)\\n", style="green")
        
        trend_text.append("🔴 Bearish: ", style="red")
        trend_text.append("█" * bearish_bar_length, style="red")
        trend_text.append(f" {bearish_pct:.1f}% ({bearish_count} Coins)\\n\\n", style="red")
        
        # Markt-Sentiment
        if bullish_pct > 60:
            sentiment = "🚀 Sehr Bullish"
            sentiment_color = "bright_green"
        elif bullish_pct > 50:
            sentiment = "📈 Bullish"
            sentiment_color = "green"
        elif bullish_pct < 40:
            sentiment = "📉 Bearish"
            sentiment_color = "red"
        else:
            sentiment = "⚖️ Neutral"
            sentiment_color = "yellow"
        
        trend_text.append("💭 Sentiment: ", style="white")
        trend_text.append(sentiment, style=f"bold {sentiment_color}")
        
        return Panel(
            trend_text,
            title="📈 Trend-Analyse",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def create_performance_grid(self, df: pd.DataFrame) -> Table:
        """🏆 Performance-Grid erstellen"""
        if not RICH_AVAILABLE:
            return None
        
        # Top Performer in verschiedenen Kategorien
        top_24h = df.nlargest(1, '24h_änderung').iloc[0]
        top_7d = df.nlargest(1, '7d_änderung').iloc[0]
        top_30d = df.nlargest(1, '30d_änderung').iloc[0]
        
        worst_24h = df.nsmallest(1, '24h_änderung').iloc[0]
        worst_7d = df.nsmallest(1, '7d_änderung').iloc[0]
        worst_30d = df.nsmallest(1, '30d_änderung').iloc[0]
        
        table = Table(
            title="🏆 Performance-Champions",
            title_style="bold gold1",
            box=box.DOUBLE_EDGE,
            border_style="gold1",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("📅 Zeitraum", justify="center", style="white", width=12)
        table.add_column("🏆 Bester", justify="left", style="green", width=20)
        table.add_column("📈 Performance", justify="right", style="bold green", width=12)
        table.add_column("💥 Schlechtester", justify="left", style="red", width=20)
        table.add_column("📉 Performance", justify="right", style="bold red", width=12)
        
        table.add_row(
            "24 Stunden",
            f"🪙 {top_24h['symbol']} ({top_24h['name'][:10]})",
            f"🚀 {top_24h['24h_änderung']:+.1f}%",
            f"🪙 {worst_24h['symbol']} ({worst_24h['name'][:10]})",
            f"💥 {worst_24h['24h_änderung']:+.1f}%"
        )
        
        table.add_row(
            "7 Tage",
            f"🪙 {top_7d['symbol']} ({top_7d['name'][:10]})",
            f"🚀 {top_7d['7d_änderung']:+.1f}%",
            f"🪙 {worst_7d['symbol']} ({worst_7d['name'][:10]})",
            f"💥 {worst_7d['7d_änderung']:+.1f}%"
        )
        
        table.add_row(
            "30 Tage",
            f"🪙 {top_30d['symbol']} ({top_30d['name'][:10]})",
            f"🚀 {top_30d['30d_änderung']:+.1f}%",
            f"🪙 {worst_30d['symbol']} ({worst_30d['name'][:10]})",
            f"💥 {worst_30d['30d_änderung']:+.1f}%"
        )
        
        return table
    
    def create_market_cap_visualization(self, df: pd.DataFrame) -> str:
        """📊 Marktkapitalisierung Visualisierung"""
        if not PLOTEXT_AVAILABLE:
            return "📊 Charts benötigen plotext: pip install plotext"
        
        plt.clear_figure()
        
        # Top 10 Marktkapitalisierungen
        symbols = df['symbol'].tolist()[:10]
        market_caps = [x/1e9 for x in df['marktkapitalisierung'].tolist()[:10]]  # in Milliarden
        
        # Pie Chart für Marktkapitalisierung
        plt.bar(symbols, market_caps, color=['gold', 'silver', '#CD7F32'] + ['blue'] * 7)
        plt.title("💰 Top 10 Marktkapitalisierungen (Milliarden €)")
        plt.xlabel("Kryptowährungen")
        plt.ylabel("Marktkapitalisierung (Milliarden €)")
        
        plt.plotsize(80, 15)
        return plt.build()
    
    def display_enhanced_analysis(self, num: int = 10, currency: str = "eur"):
        """🎨 Erweiterte Analyse mit Charts anzeigen"""
        if not RICH_AVAILABLE:
            print("⚠️ Rich nicht verfügbar. Installiere mit: pip install rich")
            return
        
        # Header
        header = Panel(
            Align.center(Text("📊 KRYPTO-CHARTS v2.6", style="bold magenta")),
            box=box.DOUBLE_EDGE,
            border_style="magenta"
        )
        self.console.print(header)
        self.console.print()
        
        # Daten laden mit Progress
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Lade Kryptowährungsdaten...", total=100)
            
            try:
                progress.update(task, advance=30)
                coins = self.cg.get_coins_markets(
                    vs_currency=currency, order="market_cap", per_page=num,
                    page=1, sparkline=False, price_change_percentage='24h,7d,30d'
                )
                progress.update(task, advance=70)
                
                df = pd.DataFrame(coins)
                df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 
                        'market_cap', 'price_change_percentage_24h_in_currency',
                        'price_change_percentage_7d_in_currency',
                        'price_change_percentage_30d_in_currency']]
                
                df.columns = ['id', 'symbol', 'name', 'rang', 'preis', 'marktkapitalisierung',
                             '24h_änderung', '7d_änderung', '30d_änderung']
                df['symbol'] = df['symbol'].str.upper()
                
            except Exception as e:
                self.console.print(f"❌ [red]Fehler: {e}[/red]")
                return
        
        # Charts und Visualisierungen
        self.console.print()
        
        # 1. Trend-Visualisierung
        trend_panel = self.create_trend_visualization(df)
        if trend_panel:
            self.console.print(trend_panel)
            self.console.print()
        
        # 2. Performance-Grid
        performance_table = self.create_performance_grid(df)
        if performance_table:
            self.console.print(performance_table)
            self.console.print()
        
        # 3. 24h Chart
        if PLOTEXT_AVAILABLE:
            chart_24h = self.create_terminal_chart(df, "bar")
            chart_panel = Panel(
                Text(chart_24h, style="white"),
                title="📊 24h Preisänderungen",
                border_style="cyan",
                box=box.ROUNDED
            )
            self.console.print(chart_panel)
            self.console.print()
            
            # 4. Marktkapitalisierung Chart
            market_cap_chart = self.create_market_cap_visualization(df)
            market_cap_panel = Panel(
                Text(market_cap_chart, style="white"),
                title="💰 Marktkapitalisierungen",
                border_style="gold1",
                box=box.ROUNDED
            )
            self.console.print(market_cap_panel)
        else:
            # Fallback ASCII Charts
            ascii_chart = self.create_ascii_chart(
                df['24h_änderung'].tolist()[:10],
                df['symbol'].tolist()[:10],
                "24h Preisänderungen"
            )
            chart_panel = Panel(
                Text(ascii_chart, style="white"),
                title="📊 24h Preisänderungen (ASCII)",
                border_style="cyan",
                box=box.ROUNDED
            )
            self.console.print(chart_panel)
        
        # Footer
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        footer = Panel(
            Align.center(Text(f"📅 Generiert am: {timestamp} | 📊 CoinGecko API", style="dim")),
            border_style="green",
            box=box.ROUNDED
        )
        self.console.print()
        self.console.print(footer)


def main():
    """📊 Hauptfunktion für Chart-Version"""
    parser = argparse.ArgumentParser(description="📊 Krypto-Charts v2.6")
    parser.add_argument("-n", "--num", type=int, default=10, help="Anzahl Kryptowährungen")
    parser.add_argument("-c", "--currency", type=str, default="eur", help="Währung")
    
    args = parser.parse_args()
    
    analyzer = ChartCryptoAnalyzer()
    analyzer.display_enhanced_analysis(args.num, args.currency)


if __name__ == "__main__":
    main()
