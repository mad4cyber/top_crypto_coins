#!/usr/bin/env python3
"""
🚀 Multi-Coin AI Analysis Dashboard
Autor: mad4cyber
Version: 1.0 - Enhanced Multi-Coin Edition

🎯 FEATURES:
- Multi-Coin AI-Prognosen parallel
- Performance-Vergleich zwischen Coins
- Portfolio-optimierte Empfehlungen
- Risk-Adjusted Returns
"""

import asyncio
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, TaskID
from rich import box

from ai_predictor import CryptoAIPredictor

class MultiCoinAIAnalysis:
    """🚀 Multi-Coin AI-Analyse Dashboard"""
    
    def __init__(self):
        self.console = Console()
        self.predictor = CryptoAIPredictor()
        
        # Top 15 Coins für Analyse
        self.coins = [
            'bitcoin',
            'ethereum', 
            'tether',
            'binancecoin',
            'solana',
            'ripple',
            'usd-coin',
            'staked-ether',
            'dogecoin',
            'cardano',
            'tron',
            'shiba-inu',
            'avalanche-2',
            'chainlink',
            'polygon'
        ]
    
    async def analyze_coin(self, coin_id: str) -> dict:
        """Analysiere einzelnen Coin"""
        try:
            result = self.predictor.predict_future_prices(coin_id)
            return result
        except Exception as e:
            return {
                'coin_id': coin_id,
                'error': str(e)
            }
    
    async def run_multi_analysis(self):
        """Führe Multi-Coin Analyse durch"""
        self.console.print(Panel.fit("🚀 Multi-Coin AI-Analyse Dashboard", style="bold blue"))
        
        # Progress Bar
        with Progress() as progress:
            task = progress.add_task("🔍 Analysiere Kryptowährungen...", total=len(self.coins))
            
            results = {}
            for coin in self.coins:
                progress.update(task, description=f"Analysiere {coin}...")
                result = await self.analyze_coin(coin)
                results[coin] = result
                progress.advance(task)
        
        # Erfolgreiche Analysen filtern
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        
        if failed_results:
            self.console.print(f"\n⚠️ [yellow]{len(failed_results)} Analysen fehlgeschlagen[/yellow]")
        
        if not valid_results:
            self.console.print("❌ [red]Keine erfolgreichen Analysen![/red]")
            return
        
        # Dashboard erstellen
        self.create_comparison_dashboard(valid_results)
        
        # Top-Empfehlungen
        self.create_recommendations(valid_results)
    
    def create_comparison_dashboard(self, results: dict):
        """Erstelle Vergleichs-Dashboard"""
        table = Table(title="📊 Multi-Coin AI-Prognose Vergleich", box=box.ROUNDED)
        table.add_column("🪙 Coin", style="cyan")
        table.add_column("💰 Aktuell", justify="right", style="white")
        table.add_column("🔮 Prognose 24h", justify="right", style="yellow") 
        table.add_column("📈 Änderung", justify="right")
        table.add_column("🎯 Konfidenz", justify="right", style="green")
        table.add_column("📊 Signal", justify="center")
        
        # Sortiere nach erwarteter Änderung
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].get('price_change_pct', 0), 
            reverse=True
        )
        
        for coin_id, result in sorted_results:
            current = result.get('current_price', 0)
            predicted = result.get('predicted_price', 0)
            change_pct = result.get('price_change_pct', 0)
            confidence = result.get('confidence', 0)
            
            # Signal bestimmen
            if change_pct > 5:
                signal = "🚀 STRONG BUY"
                signal_color = "bright_green"
            elif change_pct > 2:
                signal = "🟢 BUY"  
                signal_color = "green"
            elif change_pct > -2:
                signal = "⚪ HOLD"
                signal_color = "white"
            elif change_pct > -5:
                signal = "🟠 SELL"
                signal_color = "yellow"
            else:
                signal = "🔴 STRONG SELL"
                signal_color = "red"
            
            # Farbe für Änderung
            change_color = "green" if change_pct >= 0 else "red"
            
            table.add_row(
                coin_id.upper()[:8],
                f"€{current:.2f}",
                f"€{predicted:.2f}",
                f"[{change_color}]{change_pct:+.2f}%[/{change_color}]",
                f"{confidence:.1%}",
                f"[{signal_color}]{signal}[/{signal_color}]"
            )
        
        self.console.print(table)
    
    def create_recommendations(self, results: dict):
        """Erstelle Trading-Empfehlungen"""
        # Top Gewinner
        winners = [(k, v) for k, v in results.items() 
                  if v.get('price_change_pct', 0) > 0]
        winners.sort(key=lambda x: x[1]['price_change_pct'], reverse=True)
        
        # Top Verlierer  
        losers = [(k, v) for k, v in results.items()
                 if v.get('price_change_pct', 0) < 0]
        losers.sort(key=lambda x: x[1]['price_change_pct'])
        
        # Panels erstellen
        panels = []
        
        # Gewinner Panel
        if winners:
            winner_content = ""
            for i, (coin, data) in enumerate(winners[:3], 1):
                change = data['price_change_pct']
                conf = data['confidence']
                winner_content += f"{i}. {coin.upper()}: +{change:.2f}% (Konfidenz: {conf:.1%})\n"
            
            panels.append(Panel(
                winner_content.strip(),
                title="🚀 Top Gewinner (24h)",
                border_style="green"
            ))
        
        # Verlierer Panel
        if losers:
            loser_content = ""
            for i, (coin, data) in enumerate(losers[:3], 1):
                change = data['price_change_pct']
                conf = data['confidence']
                loser_content += f"{i}. {coin.upper()}: {change:.2f}% (Konfidenz: {conf:.1%})\n"
            
            panels.append(Panel(
                loser_content.strip(),
                title="📉 Größte Verluste (24h)",
                border_style="red"
            ))
        
        # Beste Konfidenz
        best_confidence = sorted(
            results.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )[:3]
        
        conf_content = ""
        for i, (coin, data) in enumerate(best_confidence, 1):
            conf = data['confidence']
            change = data['price_change_pct']
            conf_content += f"{i}. {coin.upper()}: {conf:.1%} ({change:+.2f}%)\n"
        
        panels.append(Panel(
            conf_content.strip(),
            title="🎯 Beste AI-Konfidenz",
            border_style="blue"
        ))
        
        # Portfolio-Empfehlung
        portfolio_weight = {}
        total_confidence = sum(r.get('confidence', 0) for r in results.values())
        
        for coin, data in results.items():
            conf = data.get('confidence', 0)
            change = data.get('price_change_pct', 0)
            
            # Gewichtung: Konfidenz * positive Änderung
            if change > 0:
                weight = (conf / total_confidence) * (1 + change/100)
                portfolio_weight[coin] = weight
        
        # Normalisiere Gewichtungen auf 100%
        total_weight = sum(portfolio_weight.values())
        if total_weight > 0:
            portfolio_weight = {k: (v/total_weight)*100 
                              for k, v in portfolio_weight.items()}
        
        # Sortiere nach Gewichtung
        portfolio_sorted = sorted(portfolio_weight.items(), 
                                key=lambda x: x[1], reverse=True)
        
        portfolio_content = ""
        for coin, weight in portfolio_sorted[:5]:
            portfolio_content += f"• {coin.upper()}: {weight:.1f}%\n"
        
        if portfolio_content:
            panels.append(Panel(
                portfolio_content.strip(),
                title="💼 AI-Portfolio Empfehlung",
                border_style="yellow"
            ))
        
        # Zeige Panels
        if len(panels) >= 2:
            columns = Columns(panels, equal=True)
            self.console.print("\n")
            self.console.print(columns)
        
        # Statistiken
        avg_change = sum(r.get('price_change_pct', 0) for r in results.values()) / len(results)
        avg_confidence = sum(r.get('confidence', 0) for r in results.values()) / len(results)
        
        stats_content = f"""
📊 Markt-Übersicht:
├─ Analysierte Coins: {len(results)}
├─ Durchschn. Änderung: {avg_change:+.2f}%
├─ Durchschn. Konfidenz: {avg_confidence:.1%}
└─ Positive Prognosen: {len(winners)} von {len(results)}
        """
        
        self.console.print(Panel(
            stats_content.strip(),
            title="📈 Markt-Statistiken",
            border_style="white"
        ))

async def main():
    """🚀 Hauptfunktion"""
    analyzer = MultiCoinAIAnalysis()
    
    try:
        await analyzer.run_multi_analysis()
    except KeyboardInterrupt:
        print("\n👋 Analyse abgebrochen!")

if __name__ == "__main__":
    asyncio.run(main())