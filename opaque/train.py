from itertools import product
import json

import numpy as np
from sklearn.model_selection import KFold

from opaque.locations import BACKGROUND_DICTIONARY_PATH
from opaque.locations import DIAGNOSTIC_TEST_PRIOR_MODEL_PATH
from opaque.locations import NEGATIVE_SET_PATH
from opaque.nlp.featurize import BaselineTfidfVectorizer
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.ood.svm import LinearOneClassSVM

from opaque.betabinomial_regression import DiagnosticTestPriorModel


def train_anomaly_detector(
        agent_texts,
        train_texts,
        nu_vals,
        max_features_vals,
        n_folds=5,
        negative_texts=None,
        no_above=0.05,
        no_below=5,
        random_state=None,
        predict_shape_params=False,
        num_mesh_texts=None,
        num_entrez_texts=None,
):
    if negative_texts is None:
        with open(NEGATIVE_SET_PATH) as f:
            negative_texts = json.load(f)
    stats = {}
    for nu, max_features in product(nu_vals, max_features_vals):
        ad_model = GroundingAnomalyDetector(
            BaselineTfidfVectorizer(
                BACKGROUND_DICTIONARY_PATH,
                max_features_per_class=max_features,
                no_above=no_above,
                no_below=no_below,
                stop_words=agent_texts,
                smartirs="ntc",
            ),
            LinearOneClassSVM(nu=nu)
        )
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = kfold.split(train_texts)
        spec_list = []
        for train, test in splits:
            ad_model.fit(
                [
                    text for i, text in enumerate(train_texts) if i in train
                ]
            )
            preds_pos = ad_model.predict(
                [
                    text for i, text in enumerate(train_texts) if i in test
                ]
            ).flatten()
            spec_list.append(sum(preds_pos == 1.0) / len(preds_pos))
        preds_neg = ad_model.predict(negative_texts.values()).flatten()
        sens = sum(preds_neg == -1.0) / len(preds_neg)
        mean_spec = np.mean(spec_list)
        std_spec = np.std(spec_list)
        J = sens + mean_spec - 1
        stats[(nu, max_features)] = (
            sens, sum(preds_neg == 1.0), mean_spec, std_spec, J
        )
    # Choose values of nu and max features that maximize J
    best_params = max(stats.items(), key=lambda x: x[1][4])[0]
    best_nu, best_max_features = best_params
    ad_model = GroundingAnomalyDetector(
        BaselineTfidfVectorizer(
            BACKGROUND_DICTIONARY_PATH,
            max_features_per_class=best_max_features,
            no_above=no_above,
            no_below=no_below,
            stop_words=agent_texts,
            smartirs="ntc",
        ),
        LinearOneClassSVM(nu=best_nu)
    )
    ad_model.fit(train_texts)
    features = None
    if (
            num_mesh_texts is not None and
            num_entrez_texts is not None and
            isinstance(num_mesh_texts, int) and
            isinstance(num_entrez_texts, int)
    ):
        log_num_mesh = np.log(num_mesh_texts + 1)
        log_num_entrez = np.log(num_entrez_texts + 1)
        best_params = (best_nu, best_max_features)
        sens_neg_set, _, mean_spec, std_spec, _ = stats[best_params]
        features = [
            best_nu,
            best_max_features,
            log_num_entrez,
            log_num_mesh,
            sens_neg_set,
            mean_spec,
            std_spec,
        ]
    shape_params = None
    if predict_shape_params and features is not None:
        prior_model = DiagnosticTestPriorModel.load(
            DIAGNOSTIC_TEST_PRIOR_MODEL_PATH,
        )
        sp = prior_model.predict_shape_params(**features)
        shape_params = {
            "sens_alpha": sp.sens_alpha,
            "sens_beta": sp.sens_beta,
            "spec_alpha": sp.spec_alpha,
            "spec_beta": sp.spec_beta,
            "sens_alpha_var": sp.sens_alpha_var,
            "sens_beta_var": sp.sens_beta_var,
            "spec_alpha_var": sp.spec_alpha_var,
            "spec_beta_var": sp.spec_beta_var,
        }

    return {
        "model": ad_model.get_model_info(),
        "train_stats": stats,
        "best_params": {"nu": best_nu, "max_features": best_max_features},
        "num_training_texts": len(train_texts),
        "shape_params": shape_params,
        "features": features,
    }
