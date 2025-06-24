import argparse
from pycoingecko import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate
import json
import os
from datetime import datetime
import time

# Verbessertes Argument-Parsing für bessere Klarheit und Benutzerfreundlichkeit
def parse_arguments():
    parser = argparse.ArgumentParser(description="Kryptowährungsdaten abrufen, filtern und verwalten.")
    parser.add_argument("-n", "--num", type=int, default=10, help="Anzahl der Top-Kryptowährungen zum Abrufen.")
    parser.add_argument("-c", "--currency", type=str, default="eur", help="Währung für die Anzeige der Werte (Standard: EUR).")
    parser.add_argument("-s", "--sort_by", type=str, default="market_cap",
                        choices=["market_cap", "gecko_desc", "gecko_asc", "mcap_desc", "mcap_asc", "price_asc",
                                 "price_desc", "volume_asc", "volume_desc"],
                        help="Sortierreihenfolge der Kryptowährungen.")
    parser.add_argument("--categories", type=str, nargs="+", help="Kryptowährungen nach Kategorie filtern (z.B. defi, nft, metaverse, layer1, layer2).")
    parser.add_argument("--min_market_cap", type=int, default=0, help="Mindestmarktkapitalisierung.")
    parser.add_argument("--max_market_cap", type=int, help="Maximale Marktkapitalisierung.")
    parser.add_argument("--exclude", type=str, nargs="+", help="Kryptowährungen nach ID ausschließen.")
    parser.add_argument("--include", type=str, nargs="+", help="Nur bestimmte Kryptowährungen nach ID einschließen.")
    parser.add_argument("--export_format", type=str, choices=["csv", "json", "excel"], help="Exportformat.")
    parser.add_argument("--price_alert", type=str, nargs="+",
                        help="Preisalarme setzen als 'symbol:zielpreis:above|below'.")
    parser.add_argument("--show_volume", action="store_true", help="Handelsvolumen anzeigen.")
    parser.add_argument("--show_supply", action="store_true", help="Umlaufmenge anzeigen.")
    parser.add_argument("--compact", action="store_true", help="Kompakte Anzeige mit weniger Spalten.")
    parser.add_argument("--lang", type=str, default="de", choices=["de", "en"], help="Sprache für die Ausgabe.")
    return parser.parse_args()


# Top-Kryptowährungen mit Fehlerbehandlung abrufen
def get_top_crypto(num, currency, sort_by, show_volume=False, show_supply=False, lang="de"):
    try:
        cg = CoinGeckoAPI()
        coins = cg.get_coins_markets(vs_currency=currency, order=sort_by, per_page=num,
                                     page=1, sparkline=False, price_change_percentage='24h,7d,30d')
    except Exception as e:
        error_msg = f"Fehler beim Abrufen der Daten von CoinGecko: {e}" if lang == "de" else f"Error fetching data from CoinGecko: {e}"
        print(error_msg)
        return pd.DataFrame()
    
    df = pd.DataFrame(coins)
    
    # Basis-Spalten
    base_columns = ['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 'market_cap',
                   'price_change_percentage_24h_in_currency', 'price_change_percentage_7d_in_currency',
                   'price_change_percentage_30d_in_currency']
    
    # Zusätzliche Spalten hinzufügen
    if show_volume:
        base_columns.append('total_volume')
    if show_supply:
        base_columns.extend(['circulating_supply', 'max_supply'])
    
    df = df[base_columns]
    
    # Deutsche Spaltennamen
    column_names = ['id', 'symbol', 'name', 'rang', 'preis', 'marktkapitalisierung', '24h_änderung', '7d_änderung', '30d_änderung']
    
    if show_volume:
        column_names.append('handelsvolumen')
    if show_supply:
        column_names.extend(['umlaufmenge', 'max_menge'])
    
    df.columns = column_names
    df['symbol'] = df['symbol'].str.upper()
    
    return df


# Fetch categories separately and filter accordingly
def filter_by_category(df, categories):
    if not categories:
        return df
    
    cg = CoinGeckoAPI()
    try:
        all_coins = cg.get_coins_list()  # Fetch all available coins
    except Exception as e:
        print(f"Error fetching categories: {e}")
        return df
    
    coin_categories = {coin['id']: coin.get('categories', []) for coin in all_coins}
    
    df['category'] = df['id'].map(coin_categories)
    df = df[df['category'].apply(lambda cat_list: any(cat.lower() in (c.lower() for c in cat_list) for cat in categories))]
    return df


# Verbesserte Filterlogik
def filter_crypto(df, min_market_cap=0, max_market_cap=None, exclude=None, include=None, lang="de"):
    initial_len = len(df)
    
    # Marktkapitalisierung filtern
    df = df[df['marktkapitalisierung'] >= min_market_cap]
    if max_market_cap:
        df = df[df['marktkapitalisierung'] <= max_market_cap]
    
    # Ausschließen/Einschließen
    if exclude:
        df = df[~df['id'].isin(exclude)]
    if include:
        df = df[df['id'].isin(include)]
    
    filter_msg = f"Gefiltert von {initial_len} auf {len(df)} Kryptowährungen." if lang == "de" else f"Filtered from {initial_len} to {len(df)} cryptocurrencies."
    print(filter_msg)
    return df

# Formatierung der Ausgabe verbessern
def format_crypto_data(df, currency="eur", compact=False):
    df_formatted = df.copy()
    
    # Preise formatieren
    currency_symbol = "€" if currency.lower() == "eur" else "$" if currency.lower() == "usd" else currency.upper()
    
    # Zahlen formatieren
    df_formatted['preis'] = df_formatted['preis'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notna(x) else "N/A")
    df_formatted['marktkapitalisierung'] = df_formatted['marktkapitalisierung'].apply(
        lambda x: f"{currency_symbol}{x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else f"{currency_symbol}{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{currency_symbol}{x:,.0f}" if pd.notna(x) else "N/A"
    )
    
    # Handelsvolumen formatieren (falls vorhanden)
    if 'handelsvolumen' in df_formatted.columns:
        df_formatted['handelsvolumen'] = df_formatted['handelsvolumen'].apply(
            lambda x: f"{currency_symbol}{x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else f"{currency_symbol}{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{currency_symbol}{x:,.0f}" if pd.notna(x) else "N/A"
        )
    
    # Umlaufmenge formatieren (falls vorhanden)
    if 'umlaufmenge' in df_formatted.columns:
        df_formatted['umlaufmenge'] = df_formatted['umlaufmenge'].apply(
            lambda x: f"{x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{x:,.0f}" if pd.notna(x) else "N/A"
        )
    
    if 'max_menge' in df_formatted.columns:
        df_formatted['max_menge'] = df_formatted['max_menge'].apply(
            lambda x: f"{x/1e9:.2f}B" if pd.notna(x) and x >= 1e9 else f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{x:,.0f}" if pd.notna(x) else "∞"
        )
    
    # Prozentuale Änderungen formatieren
    for col in ['24h_änderung', '7d_änderung', '30d_änderung']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )
    
    # Kompakte Anzeige
    if compact:
        columns_to_keep = ['rang', 'symbol', 'name', 'preis', 'marktkapitalisierung', '24h_änderung']
        df_formatted = df_formatted[[col for col in columns_to_keep if col in df_formatted.columns]]
    
    return df_formatted

# Export-Funktionalität
def export_data(df, export_format, filename_prefix="crypto_data"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == "csv":
        filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Daten exportiert nach: {filename}")
    
    elif export_format == "json":
        filename = f"{filename_prefix}_{timestamp}.json"
        df.to_json(filename, orient='records', indent=2)
        print(f"Daten exportiert nach: {filename}")
    
    elif export_format == "excel":
        filename = f"{filename_prefix}_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
        print(f"Daten exportiert nach: {filename}")

# Preisalarme
def price_alert(df, symbol, target_price, alert_type, lang="de"):
    symbol = symbol.upper()
    target_price = float(target_price)
    
    coin_data = df[df['symbol'] == symbol]
    
    if coin_data.empty:
        not_found_msg = f"Kryptowährung {symbol} nicht gefunden." if lang == "de" else f"Cryptocurrency {symbol} not found."
        print(not_found_msg)
        return
    
    # Preis direkt als Zahl verwenden (vor Formatierung)
    current_price = float(coin_data['preis'].iloc[0])
    coin_name = coin_data['name'].iloc[0]
    
    # Währungssymbol für Anzeige bestimmen
    currency_symbol = "€"  # Standard für EUR
    
    if alert_type.lower() == "above" and current_price > target_price:
        alert_msg = f"🚨 ALARM: {coin_name} ({symbol}) ist über {currency_symbol}{target_price:,.2f}! Aktueller Preis: {currency_symbol}{current_price:,.2f}" if lang == "de" else f"🚨 ALERT: {coin_name} ({symbol}) is above {currency_symbol}{target_price:,.2f}! Current price: {currency_symbol}{current_price:,.2f}"
        print(alert_msg)
    elif alert_type.lower() == "below" and current_price < target_price:
        alert_msg = f"🚨 ALARM: {coin_name} ({symbol}) ist unter {currency_symbol}{target_price:,.2f}! Aktueller Preis: {currency_symbol}{current_price:,.2f}" if lang == "de" else f"🚨 ALERT: {coin_name} ({symbol}) is below {currency_symbol}{target_price:,.2f}! Current price: {currency_symbol}{current_price:,.2f}"
        print(alert_msg)


# Hauptausführung
def main():
    args = parse_arguments()
    
    # Spracheinstellungen aus Umgebungsvariable berücksichtigen
    lang = os.environ.get('CRYPTO_LANG', args.lang)
    
    print(f"🔄 Lade Kryptowährungsdaten..." if lang == "de" else "🔄 Loading cryptocurrency data...")
    
    df_crypto = get_top_crypto(args.num, args.currency, args.sort_by, 
                              show_volume=args.show_volume, 
                              show_supply=args.show_supply, 
                              lang=lang)

    if df_crypto.empty:
        no_data_msg = "Keine Daten abgerufen. Beende Programm." if lang == "de" else "No data fetched. Exiting."
        print(no_data_msg)
        return
    
    # Kategorien filtern (falls implementiert)
    # df_crypto = filter_by_category(df_crypto, args.categories)
    
    # Kryptowährungen filtern
    df_crypto = filter_crypto(df_crypto, min_market_cap=args.min_market_cap,
                              max_market_cap=args.max_market_cap,
                              exclude=args.exclude, include=args.include,
                              lang=lang)
    
    # Preisalarme überprüfen (vor Formatierung)
    if args.price_alert:
        for alert in args.price_alert:
            if ':' in alert and len(alert.split(':')) == 3:
                symbol, target_price, alert_type = alert.split(':')
                price_alert(df_crypto, symbol, target_price, alert_type, lang)
            else:
                invalid_msg = "Ungültiges Alarm-Format. Verwende 'symbol:zielpreis:above|below'" if lang == "de" else "Invalid alert format. Use 'symbol:target_price:above|below'"
                print(invalid_msg)
    
    # Daten formatieren
    df_formatted = format_crypto_data(df_crypto, args.currency, args.compact)
    
    # Exportieren
    if args.export_format:
        export_data(df_crypto, args.export_format)  # Unformatierte Daten für Export
    
    # Tabelle anzeigen
    print("\n" + "="*80)
    title = f"Top {len(df_formatted)} Kryptowährungen" if lang == "de" else f"Top {len(df_formatted)} Cryptocurrencies"
    print(f"{title:^80}")
    print("="*80)
    
    print(tabulate(df_formatted, headers='keys', tablefmt='grid', showindex=False))
    
    # Zeitstempel hinzufügen
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    time_msg = f"\n📅 Daten abgerufen am: {timestamp}" if lang == "de" else f"\n📅 Data fetched at: {timestamp}"
    print(time_msg)


if __name__ == "__main__":
    main()
