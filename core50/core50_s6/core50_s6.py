"""core50_s6 dataset."""

import tensorflow_datasets.public_api as tfds

# TODO(core50_s6): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(core50_s6): BibTeX citation
_CITATION = """
"""


class Core50S6(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for core50_s6 dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  core50_128x128.zip files should be located at /home/jaeho/tensorflow_datasets/downloads/manual
  """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(core50_s6): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(128,128, 3)),
            'label': tfds.features.ClassLabel(names=['o'+str(i) for i in range(1, 51)]),
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
    # TODO(core50_s6): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    archive_path = dl_manager.manual_dir / 'core50_128x128.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(core50_s6): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(extracted_path / 'core50_128x128/s6'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(core50_s6): Yields (key, example) tuples from the dataset
    label = ['o'+str(i) for i in range(1, 51)]
    for l in label:
      p = path / l
      for f in p.glob('*.png'):
        import random
        img_id = random.getrandbits(512)
        yield img_id, {
          'image': f,
          'label': l
        }
