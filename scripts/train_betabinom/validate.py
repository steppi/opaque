import argparse
import itertools as it
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.betabinomial_regression import DiagnosticTestPriorModel
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


parser = argparse.ArgumentParser()
parser.add_argument("run_name")
args = parser.parse_args()

here = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(here, "adeft_betabinom_dataset_processed.csv")
run_name = args.run_name


# High entropy seed generated with
# ---------------------------
# from numpy.random import SeedSequence
#
# SeedSequence().entropy
seed = 29574310898661272202790385091240407850
rng = np.random.default_rng(seed)

best_hps = pd.read_csv("best_hps_run1.csv", sep=",")
df = pd.read_csv(data_path, sep=',')

# Generate log num training texts features, (smooth with +1 to avoid log 0).
# Track separately if texts came from mesh annotations or entrez.
df['log_num_entrez'] = np.log(df.num_entrez + 1)
df['log_num_mesh'] = np.log(df.num_mesh + 1)


if run_name not in OpaqueResultsManager.show_tables():
    OpaqueResultsManager.add_table(run_name)


outer_splits = list(
    StratifiedGroupKFold(n_splits=5).split(
        df, df.joint_strat_label, groups=df.group
    )
)

results = {}

for (i, (outer_train_idx, outer_test_idx)) in enumerate(outer_splits):
    df_train = df.iloc[outer_train_idx, :]
    df_test = df.iloc[outer_test_idx, :]

    # Fit and evaluate specificity model
    df_train_spec = df_train[df_train.N_inlier > 0]
    df_test_spec = df_test[df_test.N_inlier > 0]

    y_train = df_train_spec[
        ['N_inlier', 'K_inlier']
    ].values.astype(float)
    y_test= df_test_spec[
        ['N_inlier', 'K_inlier']
    ].values.astype(float)

    X_train = get_feature_array(df_train_spec)
    X_test = get_feature_array(df_test_spec)

    prior_type, coeff_scale = best_hps[
        (best_hps.target_type == "specificity") &
        (best_hps.outer_split == i)
    ][["prior_type", "coeff_scale"]].values[0]

    spec_model = AnyMethodPipeline(
        [
            ('scale', StandardScaler()),
            (
                'betabinom',
                BetaBinomialRegressor(
                    coefficient_prior_type=prior_type,
                    coefficient_prior_scale=coeff_scale,
                    random_state=rng,
                ),
            ),
        ]
    )
    spec_model.fit(X_train, y_train)

    total_samples = np.sum(y_train[:, 0])
    total_successes = np.sum(y_train[:, 1])
    # Smoothed using Bayesian estimate with uniform prior.
    p_baseline = (total_successes + 1) / (total_samples + 2)

    N = y_test[:, 0]
    K_true = y_test[:, 1]
    
    # Predict prevalence for each case based on model.
    preds = spec_model.predict(X_test, N=N)
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

    results_spec = {
        "nkld_score": nkld_score,
        "baseline_nkld_score": baseline_nkld_score,
    }


    # Fit sensitivity model
    df_train_sens = df_train[df_train.N_outlier > 0]
    df_test_sens = df_test[df_test.N_outlier > 0]
    
    y_train = df_train_sens[
        ['N_outlier', 'K_outlier']
    ].values.astype(float)
    y_test = df_test_sens[
        ['N_outlier', 'K_outlier']
    ].values.astype(float)

    X_train = get_feature_array(df_train_sens)
    X_test = get_feature_array(df_test_sens)

    prior_type, coeff_scale = best_hps[
        (best_hps.target_type == "sensitivity") &
        (best_hps.outer_split == i)
    ][["prior_type", "coeff_scale"]].values[0]

    sens_model = AnyMethodPipeline(
        [
            ('scale', StandardScaler()),
            (
                'betabinom',
                BetaBinomialRegressor(
                    coefficient_prior_type=prior_type,
                    coefficient_prior_scale=coeff_scale,
                    random_state=rng,
                ),
            ),
        ]
    )
    sens_model.fit(X_train, y_train)
    total_samples = np.sum(y_train[:, 0])
    total_successes = np.sum(y_train[:, 1])
    # Smoothed using Bayesian estimate with uniform prior.
    p_baseline = (total_successes + 1) / (total_samples + 2)

    N = y_test[:, 0]
    K_true = y_test[:, 1]
    
    # Predict prevalence for each case based on model.
    preds = sens_model.predict(X_test, N=N)
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

    results_sens = {
        "nkld_score": nkld_score,
        "baseline_nkld_score": baseline_nkld_score,
    }

    diag_prior_model = DiagnosticTestPriorModel(sens_model, spec_model)

    df_test_joint = df_test[
        (df_test.N_outlier > 0) & (df_test.N_inlier > 0)
    ].copy()

    X_test = get_feature_array(df_test_joint)
    shape_params = diag_prior_model.batch_predict_shape_params(X_test)
    df_test_joint.loc[:,
        ["sens_alpha", "sens_beta", "spec_alpha", "spec_beta"]
    ] = shape_params

    results = {
        "split": i,
        "sens_model_metrics": results_sens,
        "spec_model_metrics": results_spec,
        "test_df": df_test_joint,
        "diag_prior_model": diag_prior_model.get_model_info(),
    }
    OpaqueResultsManager.insert(run_name, str(i), results)
