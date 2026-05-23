import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from data_loader import load_tickers, load_data
from clustering import main as run_clustering 

def prepare_3week_targets(returns_df, horizon=3):
    df = returns_df.copy()

    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.to_period('W')
    df['weekday'] = df['Date'].dt.dayofweek

    # Weekly OHLC
    weekly = df.groupby(['ticker', 'week']).agg(
        monday_open=('open',
                     lambda x: x[df.loc[x.index, 'weekday'] == 0].iloc[0]
                     if any(df.loc[x.index, 'weekday'] == 0)
                     else np.nan),
        friday_close=('close',
                      lambda x: x[df.loc[x.index, 'weekday'] == 4].iloc[-1]
                      if any(df.loc[x.index, 'weekday'] == 4)
                      else np.nan),
        sector=('Sector', 'first'),
        industry=('Industry', 'first')
    ).reset_index()

    weekly = weekly.dropna()

    weekly['weekly_return'] = (
        weekly['friday_close'] / weekly['monday_open'] - 1
    )

    weekly['week_start_date'] = (
        weekly['week'].dt.start_time
    )

    weekly = weekly.sort_values(
        ['ticker', 'week_start_date']
    )

    # Считаем кумулятивную доходность через exp(sum(log(1+x)))
    weekly['future_3w_return'] = (
        weekly.groupby('ticker')['weekly_return']
        .transform(
            lambda x: np.exp(
                np.log1p(x).rolling(window=horizon).sum().shift(-horizon)
            ) - 1
        )
    )

    weekly['target'] = (
        weekly['future_3w_return'] > 0.03 #порог для определения роста/падения
    ).astype(int)

    return weekly


def add_lagger_features(weekly_df, lagger_ticker):
    d = weekly_df[
        weekly_df['ticker'] == lagger_ticker
    ].copy()
    d = d.sort_values('week_start_date')
    
    for lag in [1, 2, 3, 4, 5]:
        d[f'ret_lag{lag}'] = (
            d['weekly_return'].shift(lag)
        )
# оконные функции
    for w in [2, 4, 6]:
        d[f'ma{w}'] = (
            d['ret_lag1']
            .rolling(w)
            .mean()
        )
        d[f'std{w}'] = (
            d['ret_lag1']
            .rolling(w)
            .std()
        )

    # календарные признаки
    d['month'] = (
        d['week_start_date'].dt.month
    )

    d['week_of_year'] = (
        d['week_start_date']
        .dt.isocalendar()
        .week.astype(int)
    )

    return d


def prepare_leader_features(returns_df, leaders, top_n=20):
    df = returns_df[
        returns_df['ticker'].isin(leaders)
    ].copy()

    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.to_period('W')

    ticker_weekly = (
        df.groupby(['ticker', 'week'])['overnight']
        .agg([
            ('m', 'mean'),
            ('s', 'std'),
            ('p', lambda x: (x > 0).mean())
        ])
        .reset_index()
    )

    leader_rating = (
        ticker_weekly
        .groupby('ticker')['p']
        .mean()
        .nlargest(top_n)
        .index
    )

    top_df = ticker_weekly[
        ticker_weekly['ticker'].isin(leader_rating)
    ]

    top_stats = (
        top_df.groupby('week')['m']
        .agg([
            ('top_mean', 'mean'),
            ('top_std', 'std'),
            ('top_median', 'median')
        ])
        .reset_index()
    )

    leader_feats = (
        ticker_weekly.groupby('week')
        .agg(
            mean=('m', 'mean'),
            median=('m', 'median'),
            std=('m', 'std'),
            pos_ratio=('p', 'mean')
        )
        .reset_index()
    )

    leader_feats = leader_feats.merge(
        top_stats,
        on='week',
        how='left'
    )

    leader_feats['week_start_date'] = (
        leader_feats['week'].dt.start_time
    )

    leader_feats = leader_feats.sort_values(
        'week_start_date'
    )

    leader_feats['consistency'] = (
        leader_feats['mean']
        / (leader_feats['std'] + 1e-6)
    )

    cols = [
        'mean', 'median', 'std', 'pos_ratio',
        'top_mean', 'top_std', 'top_median', 'consistency'
    ]

    for col in cols:
        for w in [2, 4]:
            leader_feats[f'l_{col}_ma{w}'] = (
                leader_feats[col]
                .rolling(w)
                .mean()
            )

    shift_cols = [
        c for c in leader_feats.columns
        if c not in ['week', 'week_start_date']
    ]

    for col in shift_cols:
        leader_feats[f'{col}_lag1'] = (
            leader_feats[col].shift(1)
        )

    keep_cols = (
        ['week_start_date']
        + [c for c in leader_feats.columns if '_lag1' in c]
    )

    return leader_feats[keep_cols]

def create_full_dataset(returns_df, leaders, laggers, horizon=3, use_leader_features=True):
    # таргет
    weekly_all = prepare_3week_targets(returns_df, horizon=horizon)
    # сигналы лидеров
    if use_leader_features:
        leader_feats = prepare_leader_features(returns_df, leaders)

    all_data = []

    for lagger in laggers:
        lagger_data = add_lagger_features(weekly_all, lagger)

        # модлеь с или без лидеров
        if use_leader_features:
            merged = lagger_data.merge(
                leader_feats,
                on='week_start_date',
                how='left'
            )
        else:
            merged = lagger_data.copy()

        all_data.append(merged)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values(['week_start_date', 'ticker'])
    full_df = full_df.dropna()

    return full_df


def walk_forward_backtest(
    full_df, prices_df, params, cat_features,
    train_weeks=74, step_weeks=4, prob_threshold=0.65,
    top_k=3, allocation=0.05, commission=0.001, initial_capital=100_000
):
    """Формирование портфеля. В ПН покупка, в ПТ через 4 недели продажа"""
    data = full_df.copy()
    data['week_start_date'] = pd.to_datetime(data['week_start_date'])
    unique_weeks = sorted(data['week_start_date'].unique())

    forbidden = [
        'target', 'future_3w_return', 'weekly_return', 
        'monday_open', 'friday_close', 'week', 'week_start_date'
    ]

    feature_cols = [c for c in data.columns if c not in forbidden]

    prices_open = prices_df.pivot(index='Date', columns='ticker', values='open')
    prices_close = prices_df.pivot(index='Date', columns='ticker', values='close')
    
    prices_open.index = pd.to_datetime(prices_open.index)
    prices_close.index = pd.to_datetime(prices_close.index)

    cash = initial_capital
    active_positions = []
    trades = []
    history = []
    all_predictions = []
    start_idx = train_weeks

    while start_idx < len(unique_weeks):
        train_start = unique_weeks[start_idx - train_weeks]
        train_end = unique_weeks[start_idx]
        test_end_idx = min(start_idx + step_weeks, len(unique_weeks))
        test_weeks_list = unique_weeks[start_idx:test_end_idx]

        print(f"TRAIN: {train_start.date()} -> {(train_end - pd.Timedelta(days=7)).date()}")
        print(f"TEST:  {test_weeks_list[0].date()} -> {test_weeks_list[-1].date()}")

        train_df = data[
            (data['week_start_date'] >= train_start) & 
            (data['week_start_date'] < train_end)
        ].copy()

        test_df = data[
            data['week_start_date'].isin(test_weeks_list)
        ].copy()

        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']

        model = cb.CatBoostClassifier(
            **params,
            cat_features=cat_features,
            verbose=False
        )
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        print(f"AUC = {auc:.4f}")

        test_df['pred_proba'] = preds
        all_predictions.append(test_df)

        for week in test_weeks_list:
            print(f"\nWEEK: {week.date()}")
            to_close = []

            for pos in active_positions:
                if week >= pos['exit_week']:
                    friday = pos['exit_week'] + pd.Timedelta(days=4)
                    ticker = pos['ticker']

                    if friday in prices_close.index:
                        sell_price = prices_close.loc[friday, ticker]
                        if pd.notna(sell_price):
                            proceeds = pos['shares'] * sell_price
                            pnl = proceeds - pos['cost']
                            cash += proceeds * (1 - commission)

                            trades.append({
                                'ticker': ticker,
                                'entry_date': pos['entry_week'],
                                'exit_date': friday,
                                'pnl': pnl,
                                'return_pct': pnl / pos['cost']
                            })
                            print(f"SELL {ticker} | PnL=${pnl:.2f}")
                    to_close.append(pos)

            active_positions = [p for p in active_positions if p not in to_close]
            week_data = test_df[test_df['week_start_date'] == week].copy()
            
            signals = (
                week_data[week_data['pred_proba'] > prob_threshold]
                .sort_values('pred_proba', ascending=False)
            )

            current_tickers = [p['ticker'] for p in active_positions]
            signals = signals[~signals['ticker'].isin(current_tickers)].head(top_k)
            print(f"Signals: {len(signals)}")

            monday = week
            if monday not in prices_open.index:
                continue

            for _, row in signals.iterrows():
                ticker = row['ticker']
                buy_price = prices_open.loc[monday, ticker]

                if pd.isna(buy_price):
                    continue

                alloc_cash = cash * allocation
                shares = alloc_cash // buy_price

                if shares <= 0:
                    continue

                cost = shares * buy_price
                total_cost = cost * (1 + commission)

                if cash >= total_cost:
                    cash -= total_cost
                    exit_week = monday + pd.Timedelta(weeks=step_weeks - 1)
                    
                    active_positions.append({
                        'ticker': ticker,
                        'entry_week': monday,
                        'exit_week': exit_week,
                        'shares': shares,
                        'cost': cost
                    })
                    print(f"BUY {ticker} | P={row['pred_proba']:.3f}")

            portfolio_value = cash
            friday = week + pd.Timedelta(days=4)

            if friday in prices_close.index:
                for pos in active_positions:
                    ticker = pos['ticker']
                    px = prices_close.loc[friday, ticker]
                    if pd.notna(px):
                        portfolio_value += pos['shares'] * px

            history.append({
                'date': friday,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'n_positions': len(active_positions)
            })

            total_ret = (portfolio_value / initial_capital - 1) * 100
            print(f"Portfolio=${portfolio_value:,.0f} ({total_ret:+.2f}%)")

        start_idx += step_weeks

    history_df = pd.DataFrame(history)
    trades_df = pd.DataFrame(trades)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print("\n" + "=" * 70)
    print(f"FINAL VALUE: ${final_value:,.2f}")
    print(f"TOTAL RETURN: {total_return:+.2f}%")

    if not trades_df.empty:
        print(f"WINRATE: {(trades_df['pnl'] > 0).mean():.2%}")
        print(f"AVG RETURN: {trades_df['return_pct'].mean():.2%}")

    predictions_df['decile'] = pd.qcut(
        predictions_df['pred_proba'], 10, labels=False
    )

    ranking_eval = predictions_df.groupby('decile').agg(
        avg_future_return=('future_3w_return', 'mean'),
        winrate=('target', 'mean'),
        count=('target', 'count')
    )
    print(ranking_eval)

    return {
        'history': history_df,
        'trades': trades_df,
        'predictions': predictions_df,
        'ranking_eval': ranking_eval,
        'final_value': final_value
    }



if __name__ == "__main__":
    tickers, tickers_df = load_tickers()
    returns_df = load_data(tickers)

    if 'date' in returns_df.columns:
        returns_df = returns_df.rename(columns={'date': 'Date'})
    elif 'index' in returns_df.columns:
         returns_df = returns_df.rename(columns={'index': 'Date'})
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])

    cols_to_merge = ['ticker']
    if 'Sector' in tickers_df.columns: cols_to_merge.append('Sector')
    if 'Industry' in tickers_df.columns: cols_to_merge.append('Industry')
    
    returns_df = pd.merge(returns_df, tickers_df[cols_to_merge], on='ticker', how='left')
    returns_df['Sector'] = returns_df.get('Sector', pd.Series(dtype=str)).fillna('Unknown')
    returns_df['Industry'] = returns_df.get('Industry', pd.Series(dtype=str)).fillna('Unknown')

    leaders, laggers, leaders_herm, laggers_herm, leaders_bibl,laggers_bibl, leaders_svd, laggers_svd = run_clustering()

    try:
        full_df = create_full_dataset(
            returns_df=returns_df,
            leaders=leaders,
            laggers=laggers,
            horizon=4,
            use_leader_features=True
        )

        cat_features = ['ticker', 'Sector', 'Industry'] 

        params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42
        }

        results = walk_forward_backtest(
            full_df=full_df,
            prices_df=returns_df,
            params=params,
            cat_features=cat_features,
            step_weeks=5,
            prob_threshold=0.65
        )

        full_df_no_leaders = create_full_dataset(
            returns_df=returns_df,
            leaders=leaders,
            laggers=laggers,
            horizon=4,
            use_leader_features=False
        )

        results_no_leaders = walk_forward_backtest(
            full_df=full_df_no_leaders,
            prices_df=returns_df,
            params=params,
            cat_features=cat_features,
            step_weeks=5,
            prob_threshold=0.65
        )

        plt.figure(figsize=(12, 6))
        history_df = results['history']
        history_df_no_lead = results_no_leaders['history']

        plt.plot(
            history_df['date'],
            history_df['portfolio_value'],
            color='green',
            label='С сигналом от лидеров'
        )

        plt.plot(
            history_df_no_lead['date'],
            history_df_no_lead['portfolio_value'],
            color='blue',
            label='Без сигнала от лидеров'
        )

        plt.title('Сравнение доходности портфеля (Walk-Forward)')
        plt.xlabel('Дата')
        plt.ylabel('Доходность портфеля, $')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"ошибка: {e}")