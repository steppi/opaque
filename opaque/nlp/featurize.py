from collections import defaultdict
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class BaselineTfidfVectorizer(BaseEstimator, TransformerMixin):
    """TfidfVectorizer with document frequencies from external corpus.

    Parameters
    ----------
    dict_path : str
        Path to pickle serialized gensim Dictionary
    max_features_per_class : Optional[int]
        Maximum number of features taken per class in the training data.
        Default: None
    stop_words : Optional[list]
        List of stop words that are to be excluded as features.
    """
    def __init__(
            self,
            dict_path,
            max_features_per_class=None,
            stop_words=None
    ):
        if stop_words is None:
            self.stop_words = []
        else:
            self.stop_words = stop_words
        self.dict_path = dict_path
        self.max_features_per_class = max_features_per_class
        self.__tokenize = TfidfVectorizer().build_tokenizer()

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
        dictionary = Dictionary.load(self.dict_path)
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
            # Keep only most frequent features per class
            if self.max_features_per_class is not None:
                local_dictionary.filter_extremes(
                    no_below=1,
                    no_above=1.0,
                    keep_n=self.max_features_per_class
                )
            good_tokens.update(local_dictionary.token2id.keys())

        # Filter background dictionary to top features found in
        # training dictionary
        dictionary.filter_tokens(
            good_ids=(
                key for key, value in dictionary.items()
                if value in good_tokens
            )
        )
        model = TfidfModel(dictionary=dictionary)
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
    def load(cls, dict_path, tokens):
        tfidf = BaselineTfidfVectorizer(dict_path)
        dictionary = Dictionary.load(dict_path)
        dictionary.filter_tokens(
            good_ids=(
                key for key, value in dictionary.items()
                if value in tokens
            )
        )
        model = TfidfModel(dictionary=dictionary)
        tfidf.model_ = model
        tfidf.tokens_ = tokens
        tfidf.dictionary_ = dictionary
        return tfidf

    def _tokenize(self, text):
        """Wraps tokenizer of Scikit-learns TfidfVectorizer."""
        return self.__tokenize(text)

    def _preprocess(self, text):
        """Split text into lowercase tokens."""
        return [token.lower() for token in self._tokenize(text)]
