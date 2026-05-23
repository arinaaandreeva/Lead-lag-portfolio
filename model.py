import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid


from data_loader import load_data, load_tickers
from clustering import main as get_clusters


def prepare_weekly_targets(returns_df, threshold=0.0):
    df = returns_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.to_period('W')
    df['weekday'] = df['Date'].dt.dayofweek

    weekly = df.groupby(['ticker', 'week']).agg(
        monday_open=('open', lambda x: x[df.loc[x.index, 'weekday'] == 0].iloc[0] if any(df.loc[x.index, 'weekday'] == 0) else np.nan),
        friday_close=('close', lambda x: x[df.loc[x.index, 'weekday'] == 4].iloc[-1] if any(df.loc[x.index, 'weekday'] == 4) else np.nan),
        sector=('Sector', 'first'),
        industry=('Industry', 'first')
    ).reset_index()

    weekly = weekly.dropna(subset=['monday_open', 'friday_close'])
    weekly['weekly_return'] = weekly['friday_close'] / weekly['monday_open'] - 1
    weekly['target'] = (weekly['weekly_return'] > threshold).astype(int)
    weekly['week_start_date'] = weekly['week'].dt.start_time

    return weekly


def add_lagger_features(weekly_df, lagger_ticker):
    d = weekly_df[weekly_df['ticker'] == lagger_ticker].copy()
    d = d.sort_values('week_start_date')

    d['ret_lag1'] = d['weekly_return'].shift(1)

    for lag in [2, 3, 4, 5]:
        d[f'ret_lag{lag}'] = d['weekly_return'].shift(lag)

    for w in [2, 4, 6]:
        d[f'ma{w}'] = d['ret_lag1'].rolling(w).mean()
        d[f'std{w}'] = d['ret_lag1'].rolling(w).std()

    d['month'] = d['week_start_date'].dt.month
    d['week_of_year'] = d['week_start_date'].dt.isocalendar().week.astype(int)

    return d


def prepare_leader_features(returns_df, leaders, top_n=20):
    df = returns_df[returns_df['ticker'].isin(leaders)].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.to_period('W')

    ticker_weekly = df.groupby(['ticker', 'week'])['overnight'].agg([
        ('m', 'mean'),
        ('s', 'std'),
        ('p', lambda x: (x > 0).mean())
    ]).reset_index()

    leader_rating = ticker_weekly.groupby('ticker')['p'].mean().nlargest(top_n).index
    top_df = ticker_weekly[ticker_weekly['ticker'].isin(leader_rating)]

    top_weekly_stats = top_df.groupby('week')['m'].agg([
        ('top_10_mean', 'mean'),
        ('top_10_std', 'std'),
        ('top_10_median', 'median')
    ]).reset_index()

    weekly_leader_feats = ticker_weekly.groupby('week').agg(
        mean=('m', 'mean'),
        median=('m', 'median'),
        std=('m', 'std'),
        pos_ratio=('p', 'mean'),
        max=('m', 'max'),
        min=('m', 'min')
    ).reset_index()

    weekly_leader_feats = weekly_leader_feats.merge(top_weekly_stats, on='week', how='left')
    weekly_leader_feats['week_start_date'] = weekly_leader_feats['week'].dt.start_time
    weekly_leader_feats = weekly_leader_feats.sort_values('week_start_date')

    weekly_leader_feats['consistency'] = weekly_leader_feats['mean'] / (weekly_leader_feats['std'] + 1e-6)

    target_cols = ['mean', 'std', 'pos_ratio', 'top_10_mean', 'top_10_std', 'top_10_median', 'consistency']
    for col in target_cols:
        for w in [2, 4]:
            weekly_leader_feats[f'l_{col}_ma{w}'] = weekly_leader_feats[col].rolling(w).mean()

    cols_to_shift = [c for c in weekly_leader_feats.columns if c not in ['week', 'week_start_date']]
    for col in cols_to_shift:
        weekly_leader_feats[f'{col}_lag1'] = weekly_leader_feats[col].shift(1)

    keep_cols = ['week_start_date'] + [c for c in weekly_leader_feats.columns if '_lag1' in c]
    return weekly_leader_feats[keep_cols]


def build_weekly_model_dataset(returns_df, leaders, laggers, threshold=0.0, test_weeks=4, val_weeks=4, train_weeks=33):
    weekly_all = prepare_weekly_targets(returns_df, threshold)
    leader_feats = prepare_leader_features(returns_df, leaders)

    all_data = []
    for lagger in laggers:
        lagger_data = add_lagger_features(weekly_all, lagger)
        merged = lagger_data.merge(leader_feats, on='week_start_date', how='left')
        all_data.append(merged)

    full_df = pd.concat(all_data, ignore_index=True)

    dates = full_df['week_start_date'].copy()
    sorted_idx = dates.argsort()
    dates_sorted = dates.iloc[sorted_idx].reset_index(drop=True)
    unique_weeks = dates_sorted.drop_duplicates().sort_values().reset_index(drop=True)

    test_start_week = unique_weeks.iloc[-test_weeks] if test_weeks > 0 else None
    val_start_week = unique_weeks.iloc[-(test_weeks + val_weeks)] if (test_weeks + val_weeks) > 0 else None
    train_start_week = val_start_week - pd.Timedelta(weeks=train_weeks)

    forbidden = ['target', 'weekly_return', 'monday_open', 'friday_close', 'week', 'week_start_date']
    feature_cols = [c for c in full_df.columns if c not in forbidden]

    train_mask = (full_df['week_start_date'] < val_start_week) & (full_df['week_start_date'] >= train_start_week)
    val_mask = (full_df['week_start_date'] >= val_start_week) & (full_df['week_start_date'] < test_start_week)
    test_mask = (full_df['week_start_date'] >= test_start_week)

    X_train = full_df[train_mask][feature_cols]
    y_train = full_df[train_mask]['target']
    
    X_val = full_df[val_mask][feature_cols]
    y_val = full_df[val_mask]['target']
    
    X_test = full_df[test_mask][feature_cols]
    y_test = full_df[test_mask]['target']

    forbidden_dates = ['weekly_return', 'monday_open', 'friday_close', 'week']
    feature_cols_dates = [c for c in full_df.columns if c not in forbidden_dates]
    
    X_train_dates = full_df[train_mask][feature_cols_dates]
    X_test_dates = full_df[test_mask][feature_cols_dates]

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_dates, X_test_dates


def train_and_evaluate(returns_df, leaders, laggers):  
    X_train, y_train, X_val, y_val, X_test, y_test, X_train_dates, X_test_dates = build_weekly_model_dataset(
        returns_df, leaders, laggers, threshold=0.03, test_weeks=12, val_weeks=4, train_weeks=74
    )

    cat_features = ['ticker', 'sector', 'industry']
    param_grid = {
        'iterations': [700, 1000],
        'learning_rate': [0.001, 0.005, 0.0005, 0.01, 0.03],
        'depth': [3, 5, 6, 7],
        'l2_leaf_reg': [3, 5, 7],
        'loss_function': ['Logloss'] 
    }

    best_score = -np.inf
    best_params = None
    best_model = None
    results = []

    for params in ParameterGrid(param_grid):
        params['iterations'] = int(params['iterations'])
        params['depth'] = int(params['depth'])

        model = cb.CatBoostClassifier(
            **params,
            cat_features=cat_features,
            early_stopping_rounds=50,
            random_seed=42,
            verbose=False
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        results.append({**params, 'roc_auc': score})

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    print("\nЛучшие параметры:", best_params)
    print(f"Лучший ROC-AUC на валидации: {best_score:.4f}")

    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"ROC-AUC на ТЕСТОВОЙ выборке: {test_roc_auc:.4f}")

    y_pred = (y_pred_proba_test > 0.50).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='.2f', cmap='Blues')
    plt.title("Confusion Matrix (Test Data)")
    plt.show()
    
    return best_model, X_test, y_test, X_test_dates


def main():
    tickers, _ = load_tickers()
    returns_df = load_data(tickers)

    (leaders, laggers,
     leaders_herm, laggers_herm,
     leaders_bibl, laggers_bibl,
     leaders_svd, laggers_svd) = get_clusters()
    
    leaders_to_use = leaders
    laggers_to_use = laggers
    
    print("\nОбучение модели")
    model, X_test, y_test, X_test_dates = train_and_evaluate(returns_df, leaders_to_use, laggers_to_use)


if __name__ == "__main__":
    main()