"""core50_s2 dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core50.core50_s2 import core50_s2


class Core50S2Test(tfds.testing.DatasetBuilderTestCase):
  """Tests for core50_s2 dataset."""
  # TODO(core50_s2):
  DATASET_CLASS = core50_s2.Core50S2
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
