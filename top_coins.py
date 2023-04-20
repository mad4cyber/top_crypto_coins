import argparse
import time
from pycoingecko import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate

# Function to fetch top cryptocurrencies from CoinGecko
def get_top_crypto(num=10, currency='usd', sort_by='market_cap_desc', categories=['coins'], max_retries=3):
    cg = CoinGeckoAPI()
    retries = 0

    # Retry fetching data from the API
    while retries <= max_retries:
        try:
            top_coins = []
            for category in categories:
                top_coins += cg.get_coins_markets(vs_currency=currency, per_page=num, order=sort_by, category=category)
            break
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error fetching data from CoinGecko API: {e}")
                return None
            time.sleep(2)

    # Extract and format the required data
    coin_data = []
    for coin in top_coins:
        coin_data.append({
            'id': coin['id'],
            'symbol': coin['symbol'].upper(),
            'name': coin['name'],
            'market_cap': coin['market_cap'],
            'price': coin['current_price'],
            '24h_change': coin.get('price_change_percentage_24h', None),
            '7d_change': coin.get('price_change_percentage_7d', None),
            '30d_change': coin.get('price_change_percentage_30d', None),
            '24h_volume': coin['total_volume'],
            'market_dominance': coin['market_cap_change_percentage_24h'],
        })

    return pd.DataFrame(coin_data)

# Function to get the list of valid currencies
def get_valid_currencies():
    cg = CoinGeckoAPI()
    return cg.get_supported_vs_currencies()
# Function to parse command-line arguments
def parse_arguments():
    valid_currencies = get_valid_currencies()
    sorting_criteria = [
        'market_cap_desc', 'market_cap_asc', 'gecko_desc', 'gecko_asc',
        'volume_desc', 'volume_asc', 'id_desc', 'id_asc', 'price_desc', 'price_asc',
        'percent_change_24h_desc', 'percent_change_24h_asc', 'percent_change_7d_desc', 'percent_change_7d_asc',
        'percent_change_30d_desc', 'percent_change_30d_asc'
    ]

    parser = argparse.ArgumentParser(description='Fetch and filter top cryptocurrencies from CoinGecko.')

    parser.add_argument('-n', '--num', type=int, default=10, help='Number of top cryptocurrencies to fetch (default: 10)')
    parser.add_argument('-c', '--currency', choices=valid_currencies, default='usd', help="Fiat currency (default: 'usd')")
    parser.add_argument('-s', '--sort_by', choices=sorting_criteria, default='market_cap_desc', help="Sorting criteria (default: 'market_cap_desc')")
    parser.add_argument('-mm', '--min_market_cap', type=int, default=0, help='Minimum market cap value for filtering (default: 0)')
    parser.add_argument('-mx', '--max_market_cap', type=int, default=None, help='Maximum market cap value for filtering (default: None)')
    parser.add_argument('-p24', '--min_24h_change', type=float, default=None, help='Minimum percentage change in price over the last 24 hours for filtering (default: None)')
    parser.add_argument('-p7', '--min_7d_change', type=float, default=None, help='Minimum percentage change in price over the last 7 days for filtering (default: None)')
    parser.add_argument('-p30', '--min_30d_change', type=float, default=None, help='Minimum percentage change in price over the last 30 days for filtering (default: None)')
    parser.add_argument('-mc', '--categories', nargs='+', default=['coins'], help="Categories to include (default: ['coins'])")
    parser.add_argument('--exclude', nargs='+', default=[], help="List of coins to exclude")
    parser.add_argument('--include', nargs='+', default=[], help="List of coins to include")
    parser.add_argument('--table', action='store_true', help="Display output as table")

    return parser.parse_args()


# Main function
def main():
    args = parse_arguments()

    # Fetch top cryptocurrencies from CoinGecko
    top_crypto_df = get_top_crypto(num=args.num, currency=args.currency, sort_by=args.sort_by, categories=args.categories)

    # Filter cryptocurrencies based on market cap and price change
    filtered_crypto_df = filter_crypto(top_crypto_df, min_market_cap=args.min_market_cap, max_market_cap=args.max_market_cap,
                                       min_24h_change=args.min_24h_change, min_7d_change=args.min_7d_change, min_30d_change=args.min_30d_change,
                                       exclude=args.exclude, include=args.include)

    # Display output
    if args.table:
        print(tabulate(filtered_crypto_df, headers='keys', tablefmt='psql'))
    else:
        print(filtered_crypto_df.to_string(index=False))
