import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.metrics import roc_auc_score

from data_loader import load_tickers, load_data
from clustering import main as get_clusters
from catboost_model import prepare_weekly_targets, prepare_leader_features, add_lagger_features


def get_range_mask(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Возвращает булеву маску для фильтрации датафрейма по датам."""
    return (df['week_start_date'] >= start_dt) & (df['week_start_date'] < end_dt)


def run_train_window_experiment(X_train_dates: pd.DataFrame, params: dict, cat_features: list):
    """
    Эксперимент : Оценка зависимости ROC-AUC от размера обучающего окна.
    """
    TEST_HORIZON = 4   # 4 недели тест
    VAL_HORIZON = 4    # 4 недели валидация
    TEST_FOLDS = 6     # количество walk-forward фолдов
    WINDOW_SIZES = range(24, 90, 5)

    final_results = []
    available_weeks = sorted(X_train_dates['week_start_date'].unique())

    for weeks_back in WINDOW_SIZES:
        fold_aucs = []
        print(f"\nWindow = {weeks_back} weeks")

        for fold in range(TEST_FOLDS):
            # Берем конец истории и двигаемся назад
            test_end_idx = len(available_weeks) - fold * TEST_HORIZON
            if test_end_idx <= 0:
                continue

            test_end_dt = available_weeks[test_end_idx - 1] + pd.Timedelta(weeks=1)
            test_start_dt = test_end_dt - pd.Timedelta(weeks=TEST_HORIZON)

            val_end_dt = test_start_dt
            val_start_dt = val_end_dt - pd.Timedelta(weeks=VAL_HORIZON)

            train_end_dt = val_start_dt
            train_start_dt = train_end_dt - pd.Timedelta(weeks=weeks_back)

            train_mask = get_range_mask(X_train_dates, train_start_dt, train_end_dt)
            val_mask = get_range_mask(X_train_dates, val_start_dt, val_end_dt)
            test_mask = get_range_mask(X_train_dates, test_start_dt, test_end_dt)

            if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            X_train_sub = X_train_dates.loc[train_mask].drop(columns=['week_start_date', 'target'])
            y_train_sub = X_train_dates.loc[train_mask, 'target']
            
            X_val_sub = X_train_dates.loc[val_mask].drop(columns=['week_start_date', 'target'])
            y_val_sub = X_train_dates.loc[val_mask, 'target']

            X_test_sub = X_train_dates.loc[test_mask].drop(columns=['week_start_date', 'target'])
            y_test_sub = X_train_dates.loc[test_mask, 'target']

            model = cb.CatBoostClassifier(
                **params,
                cat_features=cat_features,
                verbose=False
            )

            model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=(X_val_sub, y_val_sub),
                early_stopping_rounds=50,
                verbose=False
            )
            preds = model.predict_proba(X_test_sub)[:, 1]
            auc = roc_auc_score(y_test_sub, preds)
            fold_aucs.append(auc)

        if len(fold_aucs) > 0:
            final_results.append({
                'weeks': weeks_back,
                'auc_mean': np.mean(fold_aucs),
                'auc_median': np.median(fold_aucs),
                'auc_std': np.std(fold_aucs),
                'n_folds': len(fold_aucs)
            })

            print(
                f"FINAL | "
                f"mean={np.mean(fold_aucs):.4f} | "
                f"median={np.median(fold_aucs):.4f} | "
                f"std={np.std(fold_aucs):.4f}"
            )

    results_df = pd.DataFrame(final_results).sort_values('weeks').reset_index(drop=True)
    print(results_df)

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['weeks'], results_df['auc_median'], marker='D', linestyle='--', color='green')
    # plt.plot(results_df['weeks'], results_df['auc_mean'], marker='D', linestyle='--', color='blue')
    plt.axhline(0.5, color='red', alpha=0.5)
    plt.title('Зависимость AUC от размера окна (кросс-валидация по неделям)')
    plt.xlabel('Недель в обучении')
    plt.ylabel('Медианный ROC-AUC')
    plt.grid(True)
    plt.show()
    
    return results_df


def run_test_horizon_experiment(X_train_dates: pd.DataFrame, params: dict, cat_features: list):
    """
    Эксперимент: Оценка зависимости ROC-AUC от размера тестового окна (горизонта прогнозирования).
    """
    FIXED_TRAIN_WINDOW = 74
    N_FOLDS = 6
    HORIZONS = range(1, 10)

    final_results = []
    all_fold_results = []
    available_weeks = sorted(X_train_dates['week_start_date'].unique())

    for horizon in HORIZONS:
        fold_aucs = []
        val_horizon = horizon 
        print(f"\nForecast Horizon = {horizon} weeks")

        for fold in range(N_FOLDS):
            test_end_idx = len(available_weeks) - fold * horizon
            if test_end_idx <= 0:
                continue

            test_end_dt = available_weeks[test_end_idx - 1] + pd.Timedelta(weeks=1)
            test_start_dt = test_end_dt - pd.Timedelta(weeks=horizon)

            val_end_dt = test_start_dt
            val_start_dt = val_end_dt - pd.Timedelta(weeks=val_horizon)

            train_end_dt = val_start_dt
            train_start_dt = train_end_dt - pd.Timedelta(weeks=FIXED_TRAIN_WINDOW)

            train_mask = get_range_mask(X_train_dates, train_start_dt, train_end_dt)
            val_mask = get_range_mask(X_train_dates, val_start_dt, val_end_dt)
            test_mask = get_range_mask(X_train_dates, test_start_dt, test_end_dt)

            if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            X_train_sub = X_train_dates.loc[train_mask].drop(columns=['week_start_date', 'target'])
            y_train_sub = X_train_dates.loc[train_mask, 'target']

            X_val_sub = X_train_dates.loc[val_mask].drop(columns=['week_start_date', 'target'])
            y_val_sub = X_train_dates.loc[val_mask, 'target']

            X_test_sub = X_train_dates.loc[test_mask].drop(columns=['week_start_date', 'target'])
            y_test_sub = X_train_dates.loc[test_mask, 'target']

            model = cb.CatBoostClassifier(
                **params,
                cat_features=cat_features,
                verbose=False
            )

            model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=(X_val_sub, y_val_sub),
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds = model.predict_proba(X_test_sub)[:, 1]
            auc = roc_auc_score(y_test_sub, preds)
            fold_aucs.append(auc)

            all_fold_results.append({
                'horizon': horizon,
                'fold': fold,
                'auc': auc,
                'train_start': train_start_dt,
                'train_end': train_end_dt,
                'test_start': test_start_dt,
                'test_end': test_end_dt
            })

            print(
                f"Fold {fold+1} | "
                f"AUC={auc:.4f} | "
                f"Test [{test_start_dt.date()} -> {test_end_dt.date()}]"
            )

        if len(fold_aucs) > 0:
            result = {
                'horizon': horizon,
                'auc_mean': np.mean(fold_aucs),
                'auc_median': np.median(fold_aucs),
                'auc_std': np.std(fold_aucs),
                'n_folds': len(fold_aucs)
            }
            final_results.append(result)

            print(
                f"mean={result['auc_mean']:.4f} | "
                f"median={result['auc_median']:.4f} | "
                f"std={result['auc_std']:.4f}"
            )

    horizons_df = pd.DataFrame(final_results).sort_values('horizon').reset_index(drop=True)
    folds_df = pd.DataFrame(all_fold_results)

    print(horizons_df)

    plt.figure(figsize=(10, 5))
    # plt.plot(horizons_df['horizon'], horizons_df['auc_median'], marker='D', linestyle='--', color='green')
    plt.plot(horizons_df['horizon'], horizons_df['auc_mean'], marker='D', linestyle='--', color='blue')
    plt.axhline(0.5, color='red', alpha=0.5)
    plt.title('Зависимость AUC от размера окна теста')
    plt.xlabel('Недель в тесте')
    plt.ylabel('Средний ROC-AUC')
    plt.grid(True)
    plt.show()
    
    return horizons_df, folds_df


if __name__ == "__main__":
    tickers, _ = load_tickers()
    returns_df = load_data(tickers)

    clusters = get_clusters()
    leaders = clusters[6]
    laggers = clusters[7]
    
    weekly_all = prepare_weekly_targets(returns_df, threshold=0.03)
    leader_feats = prepare_leader_features(returns_df, leaders)

    all_data = []
    for lagger in laggers:
        lagger_data = add_lagger_features(weekly_all, lagger)
        merged = lagger_data.merge(leader_feats, on='week_start_date', how='left')
        all_data.append(merged)

    full_df = pd.concat(all_data, ignore_index=True)
    
    forbidden = ['weekly_return', 'monday_open', 'friday_close', 'week']
    feature_cols = [c for c in full_df.columns if c not in forbidden]
    dataset_for_cv = full_df[feature_cols].dropna().sort_values('week_start_date')
    
    cat_features = ['ticker', 'sector', 'industry']
    
    # фиксируем параметры
    best_params = {
        'iterations': 1000,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'random_seed': 42
    }
    
    res_train_window = run_train_window_experiment(dataset_for_cv, best_params, cat_features)
    res_horizons, folds_data = run_test_horizon_experiment(dataset_for_cv, best_params, cat_features)