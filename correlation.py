import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from data_loader import load_data
import config  # Импортируем наш конфиг

def calculate_correlations_fast(data, tickers, lookback_days=60, end_date=None):
    """
    Векторизованный быстрый расчет корреляций без циклов.
    Работает в сотни раз быстрее на матрицах 500х500.
    """
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    all_dates = sorted(data['date'].unique())

    if end_date is None:
        end_date = all_dates[-1]
    else:
        end_date = pd.to_datetime(end_date)

    if end_date not in all_dates:
        available_before = [d for d in all_dates if d <= end_date]
        if not available_before:
            raise ValueError(f"Нет данных до даты {end_date}")
        end_date = available_before[-1]

    end_idx = all_dates.index(end_date)
    start_idx = max(0, end_idx - lookback_days + 1)
    start_date = all_dates[start_idx]

    window_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Строим сводные таблицы (убеждаемся, что колонки упорядочены по списку tickers)
    overnight_pivot = window_data.pivot(index='date', columns='ticker', values='overnight').reindex(columns=tickers)
    daytime_pivot = window_data.pivot(index='date', columns='ticker', values='daytime').reindex(columns=tickers)

    # Overnight → Daytime
    combined_ol_dt = pd.concat([overnight_pivot, daytime_pivot], axis=1, keys=['OL', 'DT'])
    full_corr_matrix = combined_ol_dt.corr()
    corr_overnight_daytime = full_corr_matrix.loc['OL', 'DT'].to_numpy()

    # Daytime → Overnight
    daytime_shifted = daytime_pivot.shift(1)
    combined_dt_ol = pd.concat([daytime_shifted, overnight_pivot], axis=1, keys=['DT_shifted', 'OL'])
    full_corr_matrix_lag = combined_dt_ol.corr()
    corr_daytime_overnight = full_corr_matrix_lag.loc['DT_shifted', 'OL'].to_numpy()

    return corr_overnight_daytime, corr_daytime_overnight


def get_correlation_matrices(tickers, lookback_days=60):
    # Проверяем, лежат ли уже готовые файлы в папке data
    if os.path.exists(config.CORR_PATH_OL_DT) and os.path.exists(config.CORR_PATH_DT_OL):
        print("Загрузка готовых матриц корреляции из файлов...")
        corr_ol_dt = np.load(config.CORR_PATH_OL_DT)
        corr_dt_ol = np.load(config.CORR_PATH_DT_OL)
    else:
        returns_df = load_data(tickers)
        
        # Используем быструю версию. Если хотите старую с циклами — замените имя функции
        corr_ol_dt, corr_dt_ol = calculate_correlations_fast(
            returns_df, 
            tickers, 
            lookback_days=lookback_days
        )
        
        # Сохраняем на будущее, чтобы не считать заново
        os.makedirs("data", exist_ok=True)
        np.save(config.CORR_PATH_OL_DT, corr_ol_dt)
        np.save(config.CORR_PATH_DT_OL, corr_dt_ol)

    abs_corr_ol_dt = np.abs(corr_ol_dt)
    abs_corr_dt_ol = np.abs(corr_dt_ol)

    return abs_corr_ol_dt, abs_corr_dt_ol


if __name__ == "__main__":
    # Демонстрационный запуск
    # Возьмем топ-5 для теста
    test_tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    abs_ol_dt, abs_dt_ol = get_correlation_matrices(test_tickers, lookback_days=60)
    
    print("\n[В модуле] Overnight → Daytime (Топ-5):")
    print(pd.DataFrame(abs_ol_dt, index=test_tickers, columns=test_tickers).round(3))