import argparse
import itertools as it
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.results import OpaqueResultsManager
from opaque.stats import NKLD
from opaque.utils import AnyMethodPipeline


def get_feature_array(df):
    """Pull out array of predictors from dataframe."""
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


def main(
        data_path,
        run_name,
        coeff_prior_type_list,
        coeff_prior_scale_list,
        pymc_seed=None,
        n_outer_splits=5,
        n_inner_splits=5,
):
    df = pd.read_csv(data_path, sep=',')

    # Generate log num training texts features, (smooth with +1 to avoid log 0).
    # Track separately if texts came from mesh annotations or entrez.
    df['log_num_entrez'] = np.log(df.num_entrez + 1)
    df['log_num_mesh'] = np.log(df.num_mesh + 1)

    if run_name not in OpaqueResultsManager.show_tables():
        OpaqueResultsManager.add_table(run_name)


    # We use nested cross validation. Tune hyperparameters on inner splits.
    # Test generalization error on outer splits.
    outer_splits = list(
        StratifiedGroupKFold(n_splits=n_outer_splits).split(
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
            coeff_prior_type_list,
            coeff_prior_scale_list,
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
            # For spec, Stratify loosely by number of inlier samples N_inlier.
            # max(3, ceil(log10(N_inlier + 1)))
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
            # For sens, Stratify loosely by number of inlier samples N_outlier.
            # max(3, ceil(log10(N_outlier + 1)))
            strat_label = df_outer_train.sens_strat_label
    
        X_outer_train = get_feature_array(df_outer_train)
        X_outer_test = get_feature_array(df_outer_test)

        inner_splits = StratifiedGroupKFold(n_splits=n_inner_splits).split(
            X_outer_train, strat_label, groups=df_outer_train.group
        )

        for j, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
            key = f"{target_type}:{prior_type}:{coeff_scale}:{i}:{j}"
            if OpaqueResultsManager.get(run_name, key) is not None:
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
                            random_seed=pymc_seed,
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

            # Predict prevalence for each case based on model.
            preds = model.predict(X_outer_train[inner_test_idx, :], N=N)
            # Betabinomial regression predicts number of successes. Turn this
            # into a prevalence estimate.
            K_pred = preds[:, 1]
            p_pred = K_pred / N

            # Need to estimate true population prevalence based on sample.
            # Smooth using Bayesian estimate with uniform prior.
            p_est = (K_true + 1) / (N + 2)

            # Compare predicted and estimated prevalences with
            # Normalied Kulback Leibler divergence, the most commonly used
            # metric for quantification learning. We try to control for the
            # bias due to varying sample sizes N by stratifying the CV splits
            # roughly by sample size.
            nkld_score = NKLD(p_est, p_pred)
            baseline_nkld_score = NKLD(p_est, p_baseline)

            # Save results. We don't even try to aggregate here. These will
            # be processed by another script.
            OpaqueResultsManager.insert(
                run_name,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    args = parser.parse_args()

    here = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(here, "adeft_betabinom_dataset_processed.csv")
    if not os.path.exists(data_path):
        print("Processed dataset has not been generated. First run "
              "make_datasplits.py.")

    # High entropy seed generated with snippet.
    # ---------------------------
    # from numpy.random import SeedSequence
    #
    # SeedSequence().entropy
    pymc_seed = 1612814232824194042486396718624000821

    coeff_prior_type_list = ["normal", "laplace"]
    coeff_prior_scale_list = np.exp2(np.arange(-4, 10))
        
    main(
        data_path,
        args.run_name,
        coeff_prior_type_list,
        coeff_prior_scale_list,
        pymc_seed=pymc_seed,
        n_outer_splits=5,
        n_inner_splits=5,
    )
