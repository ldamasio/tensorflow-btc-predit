import pandas as pd

# Carregar o arquivo CSV
bitcoin_price_data = pd.read_csv('BTC_price.csv')

# Verificar o número de linhas no DataFrame
print("Número de linhas de dados:", len(bitcoin_price_data))