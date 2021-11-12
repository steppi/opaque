import argparse
import os
import subprocess
import wget

from opaque.locations import BACKGROUND_DICTIONARY_PATH
from opaque.locations import NEGATIVE_SET_PATH
from opaque.locations import OPAQUE_HOME
from opaque.locations import S3_BUCKET_URL


def download_background_dictionary():
    wget.download(
        url=f"{S3_BUCKET_URL}/{os.path.basename(BACKGROUND_DICTIONARY_PATH)}",
        out=BACKGROUND_DICTIONARY_PATH,
    )


def download_negative_set():
    compressed_path = NEGATIVE_SET_PATH + ".xz"
    wget.download(
        url=f"{S3_BUCKET_URL}/negative_set.json.xz",
        out=compressed_path,
    )
    subprocess.run(
        ["xz", "-v", "--decompress", "-1", compressed_path]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download resource files.")
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    force = args.force
    if not os.path.exists(OPAQUE_HOME):
        os.makedirs(OPAQUE_HOME)
    if force:
        if os.path.exists(BACKGROUND_DICTIONARY_PATH):
            os.remove(BACKGROUND_DICTIONARY_PATH)
        if os.path.exists(NEGATIVE_SET_PATH):
            os.remove(NEGATIVE_SET_PATH)
    if not os.path.exists(BACKGROUND_DICTIONARY_PATH):
        download_background_dictionary()
    if not os.path.exists(NEGATIVE_SET_PATH):
        download_negative_set()
