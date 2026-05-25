import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


from opaque.betabinomial_regression import BetaBinomialRegressor
from opaque.betabinomial_regression import DiagnosticTestPriorModel
from opaque.utils import AnyMethodPipeline


parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("best_hps_path")
args = parser.parse_args()

data_path = args.data_path


def get_feature_array(df):
    """Pull out array of predictors from dataframe."""
    return df[
        [
            'nu',
            'max_features',
            'log_num_entrez',
            'log_num_mesh',
            'log_num_db',
            'log_num_reader',
            'sens_neg_set',
            'mean_spec',
            'std_spec',
        ]
    ].values

seed = 140329980792857024078522205123704669829
rng = np.random.default_rng(seed)

df = pd.read_csv(data_path, sep=",")


df_spec = df[df.N_inlier > 0]
df_sens = df[df.N_outlier > 0]

best_hps = pd.read_csv(args.best_hps_path, sep=",")

prior_type, coeff_scale = best_hps[best_hps.target_type == "specificity"][
    ["prior_type", "coeff_scale"]
].values[0]

X_spec = get_feature_array(df_spec)
y_spec = df_spec[["N_inlier", "K_inlier"]].values.astype(np.int64)

X_sens = get_feature_array(df_sens)
y_sens = df_sens[["N_outlier", "K_outlier"]].values.astype(np.int64)


prior_type, coeff_scale = best_hps[best_hps.target_type == "specificity"][
    ["prior_type", "coeff_scale"]
].values[0]

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


prior_type, coeff_scale = best_hps[best_hps.target_type == "specificity"][
    ["prior_type", "coeff_scale"]
].values[0]


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


spec_model.fit(X_spec, y_spec)
sens_model.fit(X_sens, y_sens)


diag_prior_model = DiagnosticTestPriorModel(sens_model, spec_model)
model_info = diag_prior_model.get_model_info()

with open("diag_prior_model_distilled.json", "w") as f:
    json.dump(model_info, f)


with open("diag_prior_model_distilled.json") as f:
    model_info2 = json.load(f)

diag_prior_model2 = DiagnosticTestPriorModel.load(model_info2)


X = get_feature_array(df)

preds1 = diag_prior_model.batch_predict_shape_params(X)
preds2 = diag_prior_model2.batch_predict_shape_params(X)
