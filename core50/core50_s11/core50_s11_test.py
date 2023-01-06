"""core50_s11 dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core50.core50_s11 import core50_s11


class Core50S11Test(tfds.testing.DatasetBuilderTestCase):
  """Tests for core50_s11 dataset."""
  # TODO(core50_s11):
  DATASET_CLASS = core50_s11.Core50S11
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
