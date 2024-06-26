"""Stores locations of resources used by opaque."""
import os
from appdirs import user_data_dir

from opaque import __version__

here = os.path.dirname(os.path.realpath(__file__))

OPAQUE_HOME = os.environ.get("OPAQUE_HOME")
if OPAQUE_HOME is None:
    OPAQUE_HOME = os.path.join(user_data_dir(), 'opaque')
BACKGROUND_DICTIONARY_PATH = os.path.join(
    OPAQUE_HOME, "background_dictionary.pkl"
)
NEGATIVE_SET_PATH = os.path.join(
    OPAQUE_HOME, "negative_set.json"
)
DIAGNOSTIC_TEST_PRIOR_MODEL_PATH = os.path.join(
    OPAQUE_HOME, "prior_model.pkl"
)
ADEFT_BETABINOM_DATASET_PATH = os.path.join(
    OPAQUE_HOME, "adeft_betabinom_dataset.csv"
)
S3_BUCKET = "adeft"
S3_KEY_PREFIX = f"opaque/{__version__}"
RESULTS_DB_PATH = os.path.join(OPAQUE_HOME, "opaque_results.db")
