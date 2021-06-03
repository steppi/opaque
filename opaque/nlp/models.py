from sklearn.pipeline import Pipeline


class GroundingAnomalyDetector:
    def __init__(self, featurizer, out_of_dist_classifier, memory=None):
        self.pipeline = Pipeline(
            [("featurize", featurizer), ("ood", out_of_dist_classifier)],
            memory=memory
        )

    def fit(self, texts, **params):
        self.pipeline.fit(texts, **params)

    def feature_importances(self):
        ood = self.pipeline.named_steps["ood"]
        featurizer = self.pipeline.named_steps["featurize"]
        feature_names = featurizer.get_feature_names()
        scores = ood.feature_scores().toarray().tolist()[0]
        return sorted(zip(feature_names, scores), key=lambda x: -x[1])
