import yfinance as yf
import pandas as pd

# Função para obter dados históricos do preço do Bitcoin usando yfinance
def get_bitcoin_price_history(start_date, end_date):
    bitcoin_data = yf.download('BTC-USD', start=start_date, end=end_date)
    if bitcoin_data is not None and not bitcoin_data.empty:
        return bitcoin_data
    print("Erro ao obter os dados do preço do Bitcoin usando yfinance.")
    return None

# Definir a data de início e fim para obter os dados históricos
start_date = "2020-01-01"
end_date = "2024-01-01"

# Obter os dados históricos do preço do Bitcoin
bitcoin_price_data = get_bitcoin_price_history(start_date, end_date)

if bitcoin_price_data is not None:
    # Salvar os dados em um arquivo CSV
    bitcoin_price_data.to_csv('BTC_price.csv')
    print("Arquivo BTC_price.csv gerado com sucesso!")
else:
    print("Não foi possível gerar o arquivo BTC_price.csv. Verifique sua conexão com a internet e tente novamente.")
