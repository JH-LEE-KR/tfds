"""not_mnist dataset."""

import tensorflow_datasets.public_api as tfds

# TODO(not_mnist): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(not_mnist): BibTeX citation
_CITATION = """
"""


class NotMnist(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for not_mnist dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  notMNIST_small.tar.gz files should be located at /home/jaeho/tensorflow_datasets/downloads/manual
  """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(not_mnist): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(28,28,1),encoding_format='png'),
            'label': tfds.features.ClassLabel(names=['0','1','2','3','4','5','6','7','8','9']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(not_mnist): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    archive_path = dl_manager.manual_dir / 'notMNIST.zip'
    extracted_path = dl_manager.extract(archive_path)
    # TODO(not_mnist): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(extracted_path / 'notMNIST/Train'),
        'test': self._generate_examples(extracted_path / 'notMNIST/Test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(not_mnist): Yields (key, example) tuples from the dataset
    for a in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
      p = path / a
      for f in p.glob('*.png'):
        import random
        img_id = random.getrandbits(512)
        yield img_id, {
            'image': f,
            'label': ord(a) - 65,
        }
