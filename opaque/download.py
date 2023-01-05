import argparse
import boto3
import botocore
import logging
import os
import subprocess

import opaque.locations as loc


logger = logging.getLogger(__file__)


def download_background_dictionary():
    download_opaque_object(
        os.path.basename(loc.BACKGROUND_DICTIONARY_PATH),
        outpath=loc.BACKGROUND_DICTIONARY_PATH,
    )


def download_negative_set():
    compressed_path = loc.NEGATIVE_SET_PATH + ".xz"
    download_opaque_object(
        os.path.basename(compressed_path),
        outpath=compressed_path,
    )
    subprocess.run(
        ["xz", "-v", "--decompress", "-1", compressed_path]
    )


def download_diagnostic_test_prior_model():
    download_opaque_object(
        os.path.basename(loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH),
        outpath=loc.DIAGNOSTIC_TEST_PRIOR_MODEL_PATH
    )


def download_adeft_betabinom_dataset():
    download_opaque_object(
        os.path.basename(loc.ADEFT_BETABINOM_DATASET_PATH),
        outpath=loc.ADEFT_BETABINOM_DATASET_PATH
    )


def download_opaque_object(*args, outpath):
    logger.info(f"Downloading {'/'.join(args)}")
    return _anonymous_s3_download(loc.S3_BUCKET, _get_s3_key(*args), outpath)


def _get_s3_key(*args):
    return '/'.join((loc.S3_KEY_PREFIX, ) + args)


def _anonymous_s3_download(bucket, key, outpath):
    config = botocore.config.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client('s3', config=config, region_name='us-east-1')
    s3.download_file(bucket, key, outpath)


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
