import os
import math
import logging
from collections import defaultdict
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logging.getLogger("gensim").setLevel("WARNING")


class BaselineTfidfVectorizer(BaseEstimator, TransformerMixin):
    """TfidfVectorizer with document frequencies from external corpus.

    Parameters
    ----------
    dict_path : str
        Path to pickle serialized gensim Dictionary
    max_features_per_class : Optional[int|float|callable]
        Maximum number of features taken per class in the training data.
        Default: None
    stop_words : Optional[list]
        List of stop words that are to be excluded as features.
    """

    def __init__(
            self,
            path,
            no_above=0.5,
            no_below=5,
            max_features_per_class=None,
            stop_words=None,
            **tfidf_params
    ):
        self.__tokenize = TfidfVectorizer().build_tokenizer()
        if stop_words is None:
            self.stop_words = []
        else:
            self.stop_words = [word.lower() for word in stop_words]
        self.no_above = no_above
        self.no_below = no_below
        self.tfidf_params = tfidf_params
        self.max_features_per_class = max_features_per_class

        self.path = os.path.realpath(os.path.expanduser(path))

    def fit(self, raw_documents, y=None):
        if y is None:
            texts = {
                "dummy": [self._preprocess(text) for text in raw_documents]
            }
        else:
            texts = defaultdict(list)
            for text, label in zip(raw_documents, y):
                texts[label].append(self._preprocess(text))
        good_tokens = set()
        # Load background dictionary trained on large corpus
        dictionary = Dictionary.load(self.path)
        # Filter out tokens by their frequency.
        dictionary.filter_extremes(
            no_above=self.no_above, no_below=self.no_below
        )
        baseline_tfidf_model = TfidfModel(
            dictionary=dictionary, **self.tfidf_params
        )
        for processed_texts in texts.values():
            local_dictionary = Dictionary(processed_texts)
            # Filter out tokens that aren't in the global dictionary
            local_dictionary.filter_tokens(
                good_ids=(
                    key
                    for key, value in local_dictionary.items()
                    if value in dictionary.token2id
                )
            )
            # Remove stopwords
            if self.stop_words:
                stop_ids = [
                    id_
                    for token, id_ in local_dictionary.token2id.items()
                    if token in self.stop_words
                ]
                local_dictionary.filter_tokens(bad_ids=stop_ids)
            if self.max_features_per_class is not None:
                mfpc = self.max_features_per_class
                if isinstance(mfpc, int):
                    max_features = mfpc
                elif isinstance(mfpc, float) and mfpc > 0:
                    max_features = math.floor(mfpc * len(processed_texts))
                elif callable(mfpc):
                    max_features = math.floor(mfpc(len(processed_texts)))
                else:
                    raise ValueError(
                        f"Invalid input for max_features_per_class: {mfpc}"
                    )
                # Keep only top features for the data in this class.
                local_dfs = {
                    token: local_dictionary.dfs[id_]
                    for token, id_ in local_dictionary.token2id.items()
                }
                baseline_idfs = {
                    token: baseline_tfidf_model.idfs[
                        dictionary.token2id[token]
                    ]
                    for token in local_dfs
                }
                local_df_global_idf = {
                    token: df / baseline_idfs[token]
                    for token, df in local_dfs.items()
                }
                local_df_global_idf = sorted(
                    local_df_global_idf.items(), key=lambda x: -x[1]
                )
                top_features = [
                    token
                    for token, _ in local_df_global_idf[
                            0: max_features
                    ]
                ]
                local_dictionary.filter_tokens(
                    good_ids=(
                        key
                        for key, value in local_dictionary.items()
                        if value in top_features
                    )
                )
            good_tokens.update(local_dictionary.token2id.keys())

        # Filter background dictionary to top features found in
        # training dictionary
        dictionary.filter_tokens(
            good_ids=(
                key
                for key, value in dictionary.items()
                if value in good_tokens
            )
        )
        model = TfidfModel(dictionary=dictionary, **self.tfidf_params)
        self.model_ = model
        self.tokens_ = good_tokens
        self.dictionary_ = dictionary
        return self

    def transform(self, raw_documents):
        check_is_fitted(self)
        processed_texts = [self._preprocess(text) for text in raw_documents]
        corpus = (self.dictionary_.doc2bow(text) for text in processed_texts)
        transformed_corpus = self.model_[corpus]
        X = corpus2csc(transformed_corpus, num_terms=len(self.dictionary_))
        return X.transpose()

    def get_feature_names(self):
        check_is_fitted(self)
        return [
            self.dictionary_.id2token[i] for i in range(len(self.dictionary_))
        ]

    @classmethod
    def load_model_info(cls, path, model_info):
        tokens = model_info["tokens"]
        tfidf = BaselineTfidfVectorizer(path)
        dictionary = Dictionary.load(path)
        dictionary.filter_tokens(
            good_ids=(
                key for key, value in dictionary.items() if value in tokens
            )
        )
        model = TfidfModel(dictionary=dictionary)
        tfidf.model_ = model
        tfidf.tokens_ = tokens
        tfidf.dictionary_ = dictionary
        if "max_features_per_class" in model_info:
            tfidf.max_features_per_class = model_info["max_features_per_class"]
        if "stop_words" in model_info:
            tfidf.stop_words = model_info["stop_words"]
        if "no_above" in model_info:
            tfidf.no_below = model_info["no_above"]
        if "no_below" in model_info:
            tfidf.no_below = model_info["no_below"]
        if "tfidf_model_info" in model_info:
            tfidf.tfidf_model_info = model_info["tfidf_model_info"]
        return tfidf

    def get_model_info(self):
        """Returns dictionary of info needed for reconstruction."""
        check_is_fitted(self)
        return {
            "tokens": list(self.tokens_),
            "stop_words": self.stop_words,
            "no_above": self.no_above,
            "no_below": self.no_below,
            "tfidf_params": self.tfidf_params,
            "max_features_per_class": self.max_features_per_class,
        }

    def _tokenize(self, text):
        """Wraps tokenizer of Scikit-learns TfidfVectorizer."""
        return self.__tokenize(text)

    def _preprocess(self, text):
        """Split text into lowercase tokens."""
        return [token.lower() for token in self._tokenize(text)]
