Das Python-Script ermöglicht es, die Top-Kryptowährungen von CoinGecko zu filtern und auszugeben. Hier sind 10 Beispiele, wie das Script verwendet werden kann:

Standardaufruf:


Die Top 10 Kryptowährungen nach Marktkapitalisierung in USD werden ausgegeben:
python top_coins.py



Ausgabe in einer Tabelle: Die Top 20 Kryptowährungen nach Marktkapitalisierung in EUR werden ausgegeben und in einer Tabelle formatiert:
python top_coins.py -n 20 -c eur --table


Filtern nach Volumen: Die Top 50 Kryptowährungen nach Volumen in USD werden ausgegeben:
python top_coins.py -n 50 -s volume_desc


Mindest-Marktkapitalisierung: Die Top 10 Kryptowährungen nach Marktkapitalisierung in USD mit einer Mindest-Marktkapitalisierung von 1 Milliarde USD werden ausgegeben:
python top_coins.py -mm 1000000000



Maximale Marktkapitalisierung: Die Top 30 Kryptowährungen nach Marktkapitalisierung in USD mit einer maximalen Marktkapitalisierung von 500 Millionen USD werden ausgegeben und in einer Tabelle formatiert:
python top_coins.py -n 30 -mx 500000000 --table


Filtern nach 24-Stunden-Preisänderung: Die Top 15 Kryptowährungen nach Marktkapitalisierung in USD mit einer Mindest-Preisänderung von 5% in den letzten 24 Stunden werden ausgegeben:
python top_coins.py -n 15 -p24 5


Filtern nach 7-Tage-Preisänderung: Die Top 10 Kryptowährungen nach Marktkapitalisierung in USD mit einer Mindest-Preisänderung von 10% in den letzten 7 Tagen werden ausgegeben und in einer Tabelle formatiert:
python top_coins.py -n 10 -p7 10 --table


Filtern nach 30-Tage-Preisänderung: Die Top 20 Kryptowährungen nach Marktkapitalisierung in USD mit einer Mindest-Preisänderung von 20% in den letzten 30 Tagen werden ausgegeben und Bitcoin und Ethereum werden ausgeschlossen:
python top_coins.py -n 20 -p30 20 --exclude bitcoin ethereum


Filtern nach NFT-Kategorie: Die Top 25 NFT-Kryptowährungen nach Marktkapitalisierung in USD werden ausgegeben und in einer Tabelle formatiert:
python top_coins.py -n 25 -mc nft --table

Filtern nach DeFi-Kategorie: Die Top 50 DeFi-Kryptowährungen nach Marktkapitalisierung in USD mit einer Mindest-Marktkapitalisierung von 500 Millionen USD und einer Mindest-Preisänderung von 5% in den letzten 7 Tagen werden ausgegeben:

python top_coins.py -n 50 -mc defi -mm 500000000 -p7 5
