import argparse
import itertools as it
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.betabinomial_regression import DiagnosticTestPriorModel
from opaque.results import OpaqueResultsManager
from opaque.stats import NKLD
from opaque.utils import AnyMethodPipeline


parser = argparse.ArgumentParser()
parser.add_argument("run_name")
args = parser.parse_args()

here = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(here, "adeft_betabinom_dataset_processed.csv")

pymc_seed = 13893319457075495434352617086675957051
run_name = args.run_name

if run_name not in OpaqueResultsManager.show_tables():
    OpaqueResultsManager.add_table(run_name)


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


df = pd.read_csv(data_path, sep=",")

# Generate log num training texts features, (smooth with +1 to avoid log 0).
# Track separately if texts came from mesh annotations or entrez.
df['log_num_entrez'] = np.log(df.num_entrez + 1)
df['log_num_mesh'] = np.log(df.num_mesh + 1)

outer_splits = list(
    StratifiedGroupKFold(n_splits=5).split(
        df, df.joint_strat_label, groups=df.group
    )
)

coeff_prior_type_list = ["normal", "laplace"]
coeff_prior_scale_list = np.exp2(np.arange(-4, 10))

for (
        (i, (outer_train_idx, outer_test_idx)),
        prior_type,
        coeff_scale,
        target_type,
) in it.product(
    enumerate(outer_splits),
    coeff_prior_type_list,
    coeff_prior_scale_list,
    ["specificity", "sensitivity"],
):
    # Calling these "inner splits" lets us seamlessly use the
    # previous hp collection script.
    key = f"{target_type}:{prior_type}:{coeff_scale}:{0}:{i}"

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
    else:
        df_outer_train = df_outer_train[df_outer_train.N_outlier > 0]
        y_outer_train = df_outer_train[
            ['N_outlier', 'K_outlier']
        ].values.astype(float)

        df_outer_test = df_outer_test[df_outer_test.N_outlier > 0]
        y_outer_test = df_outer_test[
            ['N_outlier', 'K_outlier']
        ].values.astype(float)

    X_outer_train = get_feature_array(df_outer_train)
    X_outer_test = get_feature_array(df_outer_test)

    model = AnyMethodPipeline(
        [
            ("scale", StandardScaler()),
            (
                "betabinom",
                BetaBinomialRegressor(
                    coefficient_prior_type=prior_type,
                    coefficient_prior_scale=coeff_scale,
                    random_seed=pymc_seed,
                ),
            ),
        ]
    )
    model.fit(X_outer_train, y_outer_train)

    total_samples = np.sum(y_outer_train[:, 0])
    total_successes = np.sum(y_outer_train[:, 1])
    # Smoothed using Bayesian estimate with uniform prior.
    p_baseline = (total_successes + 1) / (total_samples + 2)

    N = y_outer_train[:, 0]
    K_true = y_outer_train[:, 1]

    # Predict prevalence for each case based on model.
    preds = model.predict(X_outer_train, N=N)
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
    # Calling these "inner splits" lets us seamlessly use the
    # previous hp collection script.
    OpaqueResultsManager.insert(
        run_name,
        key,
        {
            "nkld_score": nkld_score,
            "baseline_score": baseline_nkld_score,
            "hps": {"coeff_scale": coeff_scale, "prior_type": prior_type},
            "outer_split": 0,
            "inner_split": i,
            "target_type": target_type,
        }
    )
