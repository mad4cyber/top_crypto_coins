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
    parser.add_argument("--categories", type=str, nargs="+", help="Filter cryptocurrencies by category (e.g., defi, nft, metaverse, layer1, layer2).")
    parser.add_argument("--min_market_cap", type=int, default=0, help="Minimum market cap filter.")
    parser.add_argument("--max_market_cap", type=int, help="Maximum market cap filter.")
    parser.add_argument("--exclude", type=str, nargs="+", help="Exclude cryptocurrencies by ID.")
    parser.add_argument("--include", type=str, nargs="+", help="Include only cryptocurrencies by ID.")
    parser.add_argument("--export_format", type=str, choices=["csv", "json", "excel"], help="Export format.")
    parser.add_argument("--price_alert", type=str, nargs="+",
                        help="Set price alerts as 'symbol:target_price:above|below'.")
    return parser.parse_args()


# Fetch top cryptocurrencies with error handling
def get_top_crypto(num, currency, sort_by):
    try:
        cg = CoinGeckoAPI()
        coins = cg.get_coins_markets(vs_currency=currency, order=sort_by, per_page=num,
                                     page=1, sparkline=False, price_change_percentage='24h,7d,30d')
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(coins)
    df = df[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 'market_cap',
             'price_change_percentage_24h_in_currency', 'price_change_percentage_7d_in_currency',
             'price_change_percentage_30d_in_currency']]
    
    df.columns = ['id', 'symbol', 'name', 'rank', 'price', 'market_cap', '24h_change', '7d_change', '30d_change']
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


# Streamlined filtering logic
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


# Main execution
def main():
    args = parse_arguments()
    df_crypto = get_top_crypto(args.num, args.currency, args.sort_by)

    if df_crypto.empty:
        print("No data fetched. Exiting.")
        return
    
    df_crypto = filter_by_category(df_crypto, args.categories)
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
