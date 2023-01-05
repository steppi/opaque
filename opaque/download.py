import argparse
import os
import subprocess
import wget

import opaque.locations as loc


def download_background_dictionary():
    wget.download(
        url=f"{loc.S3_BUCKET_URL}/"
        f"{os.path.basename(loc.BACKGROUND_DICTIONARY_PATH)}",
        out=loc.BACKGROUND_DICTIONARY_PATH,
    )


def download_negative_set():
    compressed_path = loc.NEGATIVE_SET_PATH + ".xz"
    wget.download(
        url=f"{loc.S3_BUCKET_URL}/"
        f"{os.path.basename(compressed_path)}",
        out=compressed_path,
    )
    subprocess.run(
        ["xz", "-v", "--decompress", "-1", compressed_path]
    )


def download_diagnostic_test_prior_model():
    wget.download(
        url=f"{loc.S3_BUCKET_URL}/"
        f"{os.path.basename(loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH)}",
        out=loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH,
    )


def download_adeft_betabinom_dataset():
    wget.download(
        url=f"{loc.S3_BUCKET_URL}/"
        f"{os.path.basename(loc.ADEFT_BETABINOM_DATASET_PATH)}",
        out=loc.ADEFT_BETABINOM_DATASET_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download resource files.")
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    force = args.force
    if not os.path.exists(loc.OPAQUE_HOME):
        os.makedirs(loc.OPAQUE_HOME)
    if force:
        for location in (
                loc.BACKGROUND_DICTIONARY_PATH,
                loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH,
                loc.NEGATIVE_SET_PATH,
        ):
            if os.path.exists(location):
                os.remove(location)
    if not os.path.exists(loc.BACKGROUND_DICTIONARY_PATH):
        download_background_dictionary()
    if not os.path.exists(loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH):
        download_diagnostic_test_prior_model()
    if not os.path.exists(loc.NEGATIVE_SET_PATH):
        download_negative_set()
    if not os.path.exists(loc.ADEFT_BETABINOM_DATASET_PATH):
        download_adeft_betabinom_dataset()
