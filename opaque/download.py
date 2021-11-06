import os
import wget

from opaque.locations import OPAQUE_HOME
from opaque.locations import S3_BUCKET_URL
from opaque.locations import BACKGROUND_DICTIONARY_PATH


def download_background_dictionary():
    wget.download(
        url=f"{S3_BUCKET_URL}/entrez_pubmed_dictionary.pkl",
        out=BACKGROUND_DICTIONARY_PATH
    )


if __name__ == "__main__":
    if not os.path.exists(OPAQUE_HOME):
        os.makedirs(OPAQUE_HOME)
    if not os.path.exists(BACKGROUND_DICTIONARY_PATH):
        download_background_dictionary()
