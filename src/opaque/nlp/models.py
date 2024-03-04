from sklearn.pipeline import Pipeline

from opaque.locations import BACKGROUND_DICTIONARY_PATH
from opaque.nlp.featurize import BaselineTfidfVectorizer
from opaque.ood.svm import LinearOneClassSVM


class GroundingAnomalyDetector:
    def __init__(self, featurizer, out_of_dist_classifier, memory=None):
        self.pipeline = Pipeline(
            [("featurize", featurizer), ("ood", out_of_dist_classifier)],
            memory=memory,
        )

    def fit(self, texts, **params):
        return self.pipeline.fit(texts, **params)

    def predict(self, texts, **params):
        return self.pipeline.predict(texts, **params)

    def feature_importances(self):
        ood = self.pipeline.named_steps["ood"]
        featurizer = self.pipeline.named_steps["featurize"]
        feature_names = featurizer.get_feature_names()
        scores = ood.feature_scores().tolist()[0]
        return sorted(zip(feature_names, scores), key=lambda x: -x[1])

    def get_model_info(self):
        return {
            "ood": self.pipeline.named_steps["ood"].get_model_info(),
            "featurize": self.pipeline.named_steps[
                "featurize"
            ].get_model_info(),
        }

    @classmethod
    def load_model_info(
            cls,
            model_info,
            featurizer_class=None,
            out_of_dist_classifier_class=None,
            featurizer_path=BACKGROUND_DICTIONARY_PATH,
    ):
        if featurizer_class is None:
            featurizer_class = BaselineTfidfVectorizer
        if out_of_dist_classifier_class is None:
            out_of_dist_classifier_class = LinearOneClassSVM
        featurizer = featurizer_class.load_model_info(
            model_info["featurize"], path=featurizer_path
        )
        out_of_dist = out_of_dist_classifier_class.load_model_info(
            model_info["ood"]
        )
        return GroundingAnomalyDetector(featurizer, out_of_dist)
