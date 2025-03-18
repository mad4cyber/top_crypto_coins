import argparse
from pycoingecko import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate


# Improved argument parsing for better clarity and usability
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch, filter, and manage cryptocurrency data.")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of top cryptocurrencies to fetch.")
    parser.add_argument("-c", "--currency", type=str, default="usd", help="Currency to display the values in.")
    parser.add_argument("-s", "--sort_by", type=str, default="market_cap",
                        choices=["market_cap", "gecko_desc", "gecko_asc", "mcap_desc", "mcap_asc", "price_asc",
                                 "price_desc", "volume_asc", "volume_desc"],
                        help="Sort order of the cryptocurrencies.")
    parser.add_argument("--categories", type=str, nargs="+", help="Filter cryptocurrencies by category.")
    parser.add_argument("--min_market_cap", type=int, default=0, help="Minimum market cap filter.")
    parser.add_argument("--max_market_cap", type=int, help="Maximum market cap filter.")
    parser.add_argument("--exclude", type=str, nargs="+", help="Exclude cryptocurrencies by ID.")
    parser.add_argument("--include", type=str, nargs="+", help="Include only cryptocurrencies by ID.")
    parser.add_argument("--export_format", type=str, choices=["csv", "json", "excel"], help="Export format.")
    parser.add_argument("--price_alert", type=str, nargs="+",
                        help="Set price alerts as 'symbol:target_price:above|below'.")

    return parser.parse_args()


# Enhanced data retrieval with error handling and API call robustness
def get_top_crypto(num, currency, sort_by, categories=None):
    try:
        cg = CoinGeckoAPI()
        coins = cg.get_coins_markets(vs_currency=currency, order=sort_by, per_page=num,
                                     page=1, sparkline=False, price_change_percentage='24h,7d,30d')
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(coins)

    if categories:
        df = df[df['category'].str.lower().isin([cat.lower() for cat in categories])]

    df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 'market_cap',
             'price_change_percentage_24h_in_currency', 'price_change_percentage_7d_in_currency',
             'price_change_percentage_30d_in_currency']]

    df.columns = ['id', 'symbol', 'name', 'rank', 'price', 'market_cap', '24h_change', '7d_change', '30d_change']
    df['symbol'] = df['symbol'].str.upper()

    return df


# Streamlined filtering logic with better readability
def filter_crypto(df, min_market_cap=0, max_market_cap=None, exclude=None, include=None):
    initial_len = len(df)
    df = df[df['market_cap'] >= min_market_cap]
    if max_market_cap:
        df = df[df['market_cap'] <= max_market_cap]
    if exclude:
        df = df[~df['id'].isin(exclude)]
    if include:
        df = df[df['id'].isin(include)]
    print(f"Filtered from {initial_len} to {len(df)} cryptocurrencies.")

    return df


# More informative export messages and file naming conventions
def export_data(df, export_format):
    filename = f'crypto_data.{export_format}'
    try:
        if export_format == 'csv':
            df.to_csv(filename, index=False)
        elif export_format == 'json':
            df.to_json(filename, orient='records', indent=2)
        elif export_format == 'excel':
            df.to_excel(filename, index=False)
        print(f'Successfully exported data to {filename}.')
    except Exception as e:
        print(f"Error exporting data: {e}")


# Enhanced alert handling with better feedback and error checking
def price_alert(df, symbol, target_price, alert_type='above'):
    try:
        target_price = float(target_price)
        coin = df[df['symbol'] == symbol.upper()]
        if not coin.empty:
            price = coin.iloc[0]['price']
            if (alert_type == 'above' and price >= target_price) or (alert_type == 'below' and price <= target_price):
                print(f"[ALERT] {symbol.upper()} price is {price}, which is {alert_type} {target_price}")
    except ValueError:
        print(f"Invalid target price provided for alert on {symbol.upper()}.")


# Main execution flow clearly defined
def main():
    args = parse_arguments()

    df_crypto = get_top_crypto(args.num, args.currency, args.sort_by, args.categories)

    if df_crypto.empty:
        print("No data fetched. Exiting.")
        return

    df_crypto = filter_crypto(df_crypto, min_market_cap=args.min_market_cap,
                              max_market_cap=args.max_market_cap,
                              exclude=args.exclude, include=args.include)

    if args.export_format:
        export_data(df_crypto, args.export_format)

    if args.price_alert:
        for alert in args.price_alert:
            if ':' in alert:
                symbol, target_price, alert_type = alert.split(':')
                price_alert(df_crypto, symbol, target_price, alert_type)
            else:
                print("Invalid alert format. Use 'symbol:target_price:above|below'")

    print(tabulate(df_crypto, headers='keys', tablefmt='psql', showindex=False))


if __name__ == "__main__":
    main()
