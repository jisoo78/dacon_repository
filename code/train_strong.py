import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path('/Users/choijisoo/Desktop/dacon')
DATA_DIR = BASE_DIR / 'data'
CODE_DIR = BASE_DIR / 'code'

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
LAYOUT_PATH = DATA_DIR / 'layout_info.csv'
OUT_PATH = CODE_DIR / 'submission_strong.csv'

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # scenario 내 타임 인덱스 (ID 정렬 기준)
    if 'scenario_id' in df.columns and 'ID' in df.columns:
        df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
        df['scenario_step'] = df.groupby('scenario_id').cumcount()

    # ratio / interaction features
    if {'order_inflow_15m', 'robot_active'}.issubset(df.columns):
        df['order_per_active_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1.0)
    if {'unique_sku_15m', 'order_inflow_15m'}.issubset(df.columns):
        df['sku_per_order_inflow'] = df['unique_sku_15m'] / (df['order_inflow_15m'] + 1.0)
    if {'charge_queue_length', 'charger_count'}.issubset(df.columns):
        df['charge_queue_per_charger'] = df['charge_queue_length'] / (df['charger_count'] + 1.0)
    if {'congestion_score', 'order_inflow_15m'}.issubset(df.columns):
        df['congestion_x_inflow'] = df['congestion_score'] * df['order_inflow_15m']
    if {'pack_utilization', 'loading_dock_util'}.issubset(df.columns):
        df['pack_dock_pressure'] = df['pack_utilization'] * df['loading_dock_util']
    if {'battery_mean', 'low_battery_ratio'}.issubset(df.columns):
        df['battery_risk_mix'] = (100 - df['battery_mean']) * df['low_battery_ratio']
    if {'robot_active', 'robot_idle', 'robot_charging'}.issubset(df.columns):
        total_robot_state = df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1.0
        df['active_ratio_state'] = df['robot_active'] / total_robot_state
        df['idle_ratio_state'] = df['robot_idle'] / total_robot_state
        df['charging_ratio_state'] = df['robot_charging'] / total_robot_state

    # cyclic encoding for hour/day
    if 'shift_hour' in df.columns:
        df['shift_hour_sin'] = np.sin(2 * np.pi * df['shift_hour'] / 24.0)
        df['shift_hour_cos'] = np.cos(2 * np.pi * df['shift_hour'] / 24.0)
    if 'day_of_week' in df.columns:
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    return df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    y = train_df[TARGET].copy()
    test_id = test_df['ID'].copy()

    groups = train_df['scenario_id'].copy()
    layout_key_train = train_df['layout_id'].copy()
    layout_key_test = test_df['layout_id'].copy()

    X_train = train_df.drop(columns=[TARGET, 'ID'])
    X_test = test_df.drop(columns=['ID'])

    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # missing indicators (all numeric cols with missing)
    miss_cols = [c for c in num_cols if X_train[c].isna().any()]
    for c in miss_cols:
        X_train[f'is_null_{c}'] = X_train[c].isna().astype(np.int8)
        X_test[f'is_null_{c}'] = X_test[c].isna().astype(np.int8)

    # layout-wise median imputation + global fallback
    for c in num_cols:
        grp_med = train_df.groupby('layout_id')[c].median()
        tr_fill = layout_key_train.map(grp_med)
        te_fill = layout_key_test.map(grp_med)
        global_med = train_df[c].median()

        X_train[c] = X_train[c].fillna(tr_fill).fillna(global_med)
        X_test[c] = X_test[c].fillna(te_fill).fillna(global_med)

    # categorical encoding: frequency + safe unknown handling
    for c in cat_cols:
        tr = X_train[c].fillna('missing').astype(str)
        te = X_test[c].fillna('missing').astype(str)
        freq = tr.value_counts(normalize=True)
        X_train[c] = tr.map(freq).fillna(0.0).astype(np.float32)
        X_test[c] = te.map(freq).fillna(0.0).astype(np.float32)

    # outlier clipping
    all_cols = X_train.columns.tolist()
    lower = X_train[all_cols].quantile(0.001)
    upper = X_train[all_cols].quantile(0.999)
    X_train[all_cols] = X_train[all_cols].clip(lower=lower, upper=upper, axis=1)
    X_test[all_cols] = X_test[all_cols].clip(lower=lower, upper=upper, axis=1)

    return X_train, X_test, y, groups, test_id


def train_lgb_ensemble(X_train, X_test, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)

    configs = [
        dict(
            objective='mae', metric='l1', n_estimators=3000, learning_rate=0.02,
            num_leaves=63, min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        ),
        dict(
            objective='mae', metric='l1', n_estimators=3000, learning_rate=0.02,
            num_leaves=127, min_child_samples=120, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=2.0,
        ),
        dict(
            objective='huber', metric='l1', n_estimators=2500, learning_rate=0.025,
            num_leaves=95, min_child_samples=100, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.5,
        ),
    ]
    seeds = [42, 2026]

    model_oofs = []
    model_tests = []
    model_scores = []

    for cfg_idx, cfg in enumerate(configs, 1):
        for seed in seeds:
            params = cfg.copy()
            params.update(dict(random_state=seed, n_jobs=-1))

            oof = np.zeros(len(X_train), dtype=np.float64)
            pred_test = np.zeros(len(X_test), dtype=np.float64)

            print(f'===== Model {cfg_idx} | seed {seed} =====')
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups), 1):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                model = LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric='l1',
                    callbacks=[early_stopping(200), log_evaluation(0)],
                )

                oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration_)
                pred_test += model.predict(X_test, num_iteration=model.best_iteration_) / N_SPLITS

                fold_mae = mean_absolute_error(y_va, oof[va_idx])
                print(f'  Fold {fold} MAE: {fold_mae:.5f} | best_iter: {model.best_iteration_}')

            score = mean_absolute_error(y, oof)
            print(f'  -> OOF MAE: {score:.5f}\n')

            model_oofs.append(oof)
            model_tests.append(pred_test)
            model_scores.append(score)

    # weighted blending (better score gets higher weight)
    scores = np.array(model_scores)
    weights = (1.0 / scores)
    weights = weights / weights.sum()

    blend_oof = np.zeros(len(X_train), dtype=np.float64)
    blend_test = np.zeros(len(X_test), dtype=np.float64)
    for w, oof, pt in zip(weights, model_oofs, model_tests):
        blend_oof += w * oof
        blend_test += w * pt

    blend_score = mean_absolute_error(y, blend_oof)
    print('===== Final Blend =====')
    print('Model scores:', [round(s, 5) for s in model_scores])
    print('Blend OOF MAE:', round(blend_score, 5))

    return blend_test, blend_score


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    layout = pd.read_csv(LAYOUT_PATH)

    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')

    train = add_features(train)
    test = add_features(test)

    X_train, X_test, y, groups, test_id = preprocess(train, test)
    print('Final features:', X_train.shape[1])

    pred_test, blend_score = train_lgb_ensemble(X_train, X_test, y, groups)

    sub = pd.DataFrame({
        'ID': test_id,
        TARGET: pred_test,
    })
    sub.to_csv(OUT_PATH, index=False)

    print(f'Saved: {OUT_PATH}')
    print(f'Final Blend OOF MAE: {blend_score:.5f}')
    print(sub.head().to_string(index=False))


if __name__ == '__main__':
    main()
