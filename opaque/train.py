from itertools import product
import json

import numpy as np
from sklearn.model_selection import KFold

from opaque.locations import BACKGROUND_DICTIONARY_PATH
from opaque.locations import NEGATIVE_SET_PATH
from opaque.nlp.featurize import BaselineTfidfVectorizer
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.ood.svm import LinearOneClassSVM


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
    return {
        "model": ad_model.get_model_info(),
        "stats": stats,
        "best_params": {"nu": best_nu, "max_features": best_max_features},
    }
