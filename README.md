The Python script allows you to filter and output the top cryptocurrencies from CoinGecko. Here are 10 examples of how the script can be used:

Default call:

The top 10 cryptocurrencies by market capitalization in USD are outputted: python top_coins.py

Output in a table: The top 20 cryptocurrencies by market capitalization in EUR are outputted and formatted in a table: python top_coins.py -n 20 -c eur --table

Filter by volume: The top 50 cryptocurrencies by volume in USD are outputted: python top_coins.py -n 50 -s volume_desc

Minimum market capitalization: The top 10 cryptocurrencies by market capitalization in USD with a minimum market capitalization of 1 billion USD are outputted: python top_coins.py -mm 1000000000

Maximum market capitalization: The top 30 cryptocurrencies by market capitalization in USD with a maximum market capitalization of 500 million USD are outputted and formatted in a table: python top_coins.py -n 30 -mx 500000000 --table

Filter by 24-hour price change: The top 15 cryptocurrencies by market capitalization in USD with a minimum price change of 5% in the last 24 hours are outputted: python top_coins.py -n 15 -p24 5

Filter by 7-day price change: The top 10 cryptocurrencies by market capitalization in USD with a minimum price change of 10% in the last 7 days are outputted and formatted in a table: python top_coins.py -n 10 -p7 10 --table

Filter by 30-day price change: The top 20 cryptocurrencies by market capitalization in USD with a minimum price change of 20% in the last 30 days are outputted and Bitcoin and Ethereum are excluded: python top_coins.py -n 20 -p30 20 --exclude bitcoin ethereum

Filter by NFT category: The top 25 NFT cryptocurrencies by market capitalization in USD are outputted and formatted in a table: python top_coins.py -n 25 -mc nft --table

Filter by DeFi category: The top 50 DeFi cryptocurrencies by market capitalization in USD with a minimum market capitalization of 500 million USD and a minimum price change of 5% in the last 7 days are outputted:

python top_coins.py -n 50 -mc defi -mm 500000000 -p7 5
