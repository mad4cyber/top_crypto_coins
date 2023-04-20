import argparse
import time
from pycoingecko import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch and filter top cryptocurrencies.")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of top cryptocurrencies to fetch.")
    parser.add_argument("-c", "--currency", type=str, default="usd", help="Currency to display the values in.")
    parser.add_argument("-s", "--sort_by", type=str, default="market_cap", choices=["market_cap", "gecko_desc", "gecko_asc", "mcap_desc", "mcap_asc", "price_asc", "price_desc", "volume_asc", "volume_desc"], help="Sort order of the cryptocurrencies.")
    parser.add_argument("--categories", type=str, nargs="+", help="Filter cryptocurrencies by category.")
    parser.add_argument("--min_market_cap", type=int, default=0, help="Minimum market cap filter.")
    parser.add_argument("--max_market_cap", type=int, help="Maximum market cap filter.")
    parser.add_argument("--min_24h_change", type=float, help="Minimum 24-hour price change filter.")
    parser.add_argument("--min_7d_change", type=float, help="Minimum 7-day price change filter.")
    parser.add_argument("--min_30d_change", type=float, help="Minimum 30-day price change filter.")
    parser.add_argument("--exclude", type=str, nargs="+", help="Exclude specific cryptocurrencies by ID.")
    parser.add_argument("--include", type=str, nargs="+", help="Include only specific cryptocurrencies by ID.")
    parser.add_argument("--export_format", type=str, choices=["csv", "json", "excel"], help="Export format of the data.")
    parser.add_argument("--price_alert", type=str, nargs="+", help="Set price alerts in the format 'symbol:target_price:alert_type' (e.g. 'btc:50000:above').")
    parser.add_argument("--historical_data", type=str, nargs="+", help="Fetch historical data for specific cryptocurrencies by ID.")

    return parser.parse_args()

def get_top_crypto(num=10, currency='usd', sort_by='market_cap', categories=None):
    cg = CoinGeckoAPI()
    coins = cg.get_coins_markets(vs_currency=currency, order=sort_by, per_page=num, page=1, sparkline=False, price_change_percentage='24h,7d,30d')

    if categories:
        coins = [coin for coin in coins if any(category.lower() in (coin_category['name'].lower() for coin_category in coin['categories']) for category in categories)]

    df = pd.DataFrame(coins)
    df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 'market_cap', 'total_volume', 'price_change_percentage_24h', 'price_change_percentage_7d', 'price_change_percentage_30d']]
    df.columns = ['id', 'symbol', 'name', 'rank', 'price', 'market_cap', 'volume', '24h_change', '7d_change', '30d_change']
    df['symbol'] = df['symbol'].str.upper()

    return df

def filter_crypto(df, min_market_cap=0, max_market_cap=None, min_24h_change=None, min_7d_change=None, min_30d_change=None, exclude=[], include=[]):
    if max_market_cap is not None:
        df = df[df['market_cap'] <= max_market_cap]
    if min_24h_change is not None:
        df = df[df['24h_change'] >= min_24h_change]
    if min_7d_change is not None:
        df = df[df['7d_change'] >= min_7d_change]
    if min_30d_change is not None:
        df = df[df['30d_change'] >= min_30d_change]
    if exclude:
        df = df[~df['id'].isin(exclude)]
    if include:
        df = df[df['id'].isin(include)]

    return df[df['market_cap'] >= min_market_cap]

def export_data(df, export_format='csv', file_name='crypto_data'):
    if not os.path.exists('output'):
        os.makedirs('output')

    if export_format == 'csv':
        df.to_csv(f'output/{file_name}.csv', index=False)
    elif export_format == 'json':
        df.to_json(f'output/{file_name}.json', orient='records')
    elif export_format == 'excel':
        df.to_excel(f'output/{file_name}.xlsx', index=False)

def price_alert(df, symbol, target_price, alert_type='above'):
    coin = df[df['symbol'] == symbol.upper()]
    if not coin.empty:
        price = coin.iloc[0]['price']
        if (alert_type == 'above' and price >= target_price) or (alert_type == 'below' and price <= target_price):
            print(f"Price alert: {symbol.upper()} is {alert_type} {target_price}")
            
def get_historical_data(coin_id, currency='usd', days='30', interval='daily'):
    cg = CoinGeckoAPI()
    historical_data = cg.get_coin_market_chart(coin_id, vs_currency=currency, days=days, interval=interval)

    return pd.DataFrame(historical_data)
def main():
    args = parse_arguments()

    # Fetch top cryptocurrencies from CoinGecko
    top_crypto_df = get_top_crypto(num=args.num, currency=args.currency, sort_by=args.sort_by, categories=args.categories)

    # Filter cryptocurrencies based on market cap and price change
    filtered_crypto_df = filter_crypto(top_crypto_df, min_market_cap=args.min_market_cap, max_market_cap=args.max_market_cap,
                                       min_24h_change=args.min_24h_change, min_7d_change=args.min_7d_change, min_30d_change=args.min_30d_change,
                                       exclude=args.exclude, include=args.include)

    # Export data
    if args.export_format:
        export_data(filtered_crypto_df, export_format=args.export_format)

    # Check price alerts
    if args.price_alert:
        for alert in args.price_alert:
            symbol, target_price, alert_type = alert.split(':')
            price_alert(filtered_crypto_df, symbol, float(target_price), alert_type)

    # Get historical data
    if args.historical_data:
        for coin_id in args.historical_data:
            historical_data_df = get_historical_data(coin_id, currency=args.currency, days='30', interval='daily')
            print(historical_data_df)

    # Print the filtered cryptocurrencies
    print(tabulate(filtered_crypto_df, headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    main()
