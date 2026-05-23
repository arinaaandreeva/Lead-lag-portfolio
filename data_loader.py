import pandas as pd
import yfinance as yf
import os
from config import SCRAPE_DATA, DATA_PATH, START_DATE, END_DATE, TICKERS_FILE

def load_tickers():
    df = pd.read_csv(TICKERS_FILE, sep="\t")

    # названия колонок к норм виду
    df = df.rename(columns={
        "Symbol": "ticker"
    })

    tickers = df["ticker"].astype(str).str.strip().tolist()
    tickers = tickers[:100]

    return tickers, df


def download_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

    returns_data = {}

    for ticker in tickers:
        df = pd.DataFrame({
            'close': data['Close'][ticker],
            'open': data['Open'][ticker],
            'adj_close': data['Adj Close'][ticker]
        }).dropna()

        df['close_to_close'] = df['adj_close'] / df['adj_close'].shift(1) - 1
        df['daytime'] = df['close'] / df['open'] - 1
        df['overnight'] = (1 + df['close_to_close']) / (1 + df['daytime']) - 1

        df['ticker'] = ticker
        returns_data[ticker] = df
        returns_data[ticker] = df[['open', 'close','close_to_close', 'daytime', 'overnight', 'ticker', 'volume']]

    all_returns = pd.concat(returns_data.values())
    return all_returns.reset_index().rename(columns={'index': 'date'})


def load_data(tickers):
    """
    Универсальная функция:
    - либо читает parquet
    - либо скачивает и сохраняет
    """

    if SCRAPE_DATA or not os.path.exists(DATA_PATH):
        print("Downloading data from yahoo")
        df = download_prices(tickers, START_DATE, END_DATE)
        df.to_parquet(DATA_PATH)
        print("Saved to parquet")

    else:
        print("Loading data from parquet...")
        df = pd.read_parquet(DATA_PATH)

    return df