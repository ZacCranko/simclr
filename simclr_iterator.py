# %%
import data_util
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
import functools

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, image_size, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=image_size,
      width=image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)


def build_input_fn(builder: tfds.core.DatasetBuilder, 
                   batch_size: int, 
                   train_mode: str = 'pretrain', 
                   is_training: bool = True,  
                   split: str = 'train',
                   cache_dataset: bool = True, image_size: int = 32):

  def _input_fn(input_context = None):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(is_training, image_size, is_pretrain=True, )
    preprocess_fn_finetune = get_preprocess_fn(is_training, image_size, is_pretrain=False,)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if is_training and train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    dataset = builder.as_dataset(
        split = split,
        shuffle_files = is_training,
        as_supervised = True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn

# %% Testing
dataset_builder = tfds.builder("cifar10")
mini_batch_size = 128
is_training = True
split = 'train'
input_fn = build_input_fn(dataset_builder, mini_batch_size, is_training, split = split)

batch = next(iter(input_fn()))
view_a, view_b = batch



# %%
