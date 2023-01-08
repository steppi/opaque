import argparse
import itertools as it

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.results import OpaqueResultsManager
from opaque.stats import NKLD
from opaque.utils import AnyMethodPipeline


def get_feature_array(df):
    return df[
        [
            'nu',
            'max_features',
            'log_num_entrez',
            'log_num_mesh',
            'sens_neg_set',
            'mean_spec',
            'std_spec',
        ]
    ].values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune parameters for beta-binomial models of"
        " sensitivity and specificity."
    )
    parser.add_argument('data_path')
    parser.add_argument('run_name')
    parser.add_argument('--coeff_prior_type_list', nargs='+', type=str)
    parser.add_argument('--coeff_prior_scale_list', nargs='+', type=float)
    parser.add_argument('--numpy_seed', type=int)
    parser.add_argument('--pymc_seed', type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.data_path, sep=',')
    df['log_num_entrez'] = np.log(df.num_entrez + 1)
    df['log_num_mesh'] = np.log(df.num_mesh + 1)

    if args.run_name not in OpaqueResultsManager.show_tables():
        OpaqueResultsManager.add_table(args.run_name)

    outer_splits = list(
        StratifiedGroupKFold(n_splits=5).split(
            df, df.joint_strat_label, groups=df.group
        )
    )

    for (
            i,
            (
                (outer_train_idx, outer_test_idx),
                prior_type,
                coeff_scale,
                target_type,
            ),
    ) in enumerate(
        it.product(
            outer_splits,
            args.coeff_prior_type_list,
            args.coeff_prior_scale_list,
            ["specificity", "sensitivity"],
        )
    ):
        df_outer_train = df.iloc[outer_train_idx, :]
        df_outer_test = df.iloc[outer_test_idx, :]

        if target_type == "specificity":
            df_outer_train = df_outer_train[df_outer_train.N_inlier > 0]
            y_outer_train = df_outer_train[
                ['N_inlier', 'K_inlier']
            ].values.astype(float)

            df_outer_test = df_outer_test[df_outer_test.N_inlier > 0]
            y_outer_test = df_outer_test[
                ['N_inlier', 'K_inlier']
            ].values.astype(float)

            strat_label = df_outer_train.spec_strat_label

        else:
            df_outer_train = df_outer_train[df_outer_train.N_outlier > 0]
            y_outer_train = df_outer_train[
                ['N_outlier', 'K_outlier']
            ].values.astype(float)

            df_outer_test = df_outer_test[df_outer_test.N_outlier > 0]
            y_outer_test = df_outer_test[
                ['N_outlier', 'K_outlier']
            ].values.astype(float)

            strat_label = df_outer_train.sens_strat_label
    
        X_outer_train = get_feature_array(df_outer_train)
        X_outer_test = get_feature_array(df_outer_test)

        inner_splits = StratifiedGroupKFold(n_splits=5).split(
            X_outer_train, strat_label, groups=df_outer_train.group
        )

        for j, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
            key = f"{target_type}:{prior_type}:{coeff_scale}:{i}:{j}"
            if OpaqueResultsManager.get(args.run_name, key) is not None:
                print(f"Results already computed for {key}")
                continue

            model = AnyMethodPipeline(
                [
                    ('scale', StandardScaler()),
                    (
                        'betabinom',
                        BetaBinomialRegressor(
                            coefficient_prior_type=prior_type,
                            coefficient_prior_scale=coeff_scale,
                            random_seed=args.pymc_seed,
                        ),
                    ),
                ]
            )
            model.fit(
                X_outer_train[inner_train_idx], y_outer_train[inner_train_idx]
            )
            total_samples = np.sum(y_outer_train[inner_train_idx][:, 0])
            total_successes = np.sum(y_outer_train[inner_train_idx][:, 1])
            # Smoothed using Bayesian estimate with uniform prior.
            p_baseline = (total_successes + 1) / (total_samples + 2)

            N = y_outer_train[inner_test_idx, 0]
            K_true = y_outer_train[inner_test_idx, 1]
            preds = model.predict(X_outer_train[inner_test_idx, :], N=N)
            K_pred = preds[:, 1]

            p_pred = K_pred / N
            # Population prevalence only estimated based on sample. Use
            # smoothing by assuming uniform prior.
            p_est = (K_true + 1) / (N + 2)

            nkld_score = NKLD(p_est, p_pred)
            baseline_nkld_score = NKLD(p_est, p_baseline)

            OpaqueResultsManager.insert(
                args.run_name,
                key,
                {
                    "nkld_score": nkld_score,
                    "baseline_score": baseline_nkld_score,
                    "hps": {"coeff_scale", "prior_type"},
                    "outer_split": i,
                    "inner_split": j,
                    "target_type": target_type,
                }
            )
