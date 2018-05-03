# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from datetime import datetime
import os.path
import random
import sys
import threading

from PIL import Image
import numpy as np
import json

import tensorflow as tf
from flickr8k import load_flickr8k_dataset
from coco import load_coco_dataset
from common import Vocabulary, ImageMetadata




tf.flags.DEFINE_string("graph_path", "/home/hillyess/ai/project-image-caption/faster_rcnn_resnet50_coco/exported_graphs/frozen_inference_graph.pb",
                       "Faster rcnn forzen graph.")

tf.flags.DEFINE_string('dataset', "coco",
                       "Must be flickr8k, flickr30k, or coco")
# coco path
tf.flags.DEFINE_string("train_image_dir", "/home/hillyess/ai/coco/images/train2014",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/home/hillyess/ai/coco/images/val2014",
                       "Validation image directory.")
tf.flags.DEFINE_string("train_captions_file", "/home/hillyess/ai/coco/annotations/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/home/hillyess/ai/coco/annotations/captions_val2014.json",
                       "Validation captions JSON file.")
# flickr8k path
tf.flags.DEFINE_string("image_dir", "/home/hillyess/ai/project-image-caption/Flickr8k/Flicker8k_Dataset",
                       "Directory containing the image files.")
tf.flags.DEFINE_string("text_path", "/home/hillyess/ai/project-image-caption/Flickr8k/Flickr8k_text",
                       "containing txt files about image caption annotations.")

tf.flags.DEFINE_string("output_dir", "/home/hillyess/coco_tfrecord", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 8,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 16,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Useless! Directly assigned in common.py. Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Useless! Directly assigned in common.py. Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")

tf.flags.DEFINE_string("word_counts_output_file", "/home/hillyess/coco_tfrecord/word_counts.txt",
                       "Output vocabulary file of word counts.")


tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

# A map of names to SSD feature extractors.
LOAD_DATASET_MAP = {
    'flickr30k': load_flickr8k_dataset,
    'flickr8k': load_flickr8k_dataset,
    'coco': load_coco_dataset,
}


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self,graph_path):
    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)


    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with detection_graph.as_default():
      # Create a single TensorFlow Session for all image decoding calls.
      with tf.Session(config=sess_config, graph=detection_graph) as self._sess:

        self._image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self._proposal_boxes = detection_graph.get_tensor_by_name('proposal_boxes:0')

        # self._detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # self._detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # self._num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        self._feature = detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')


  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
  
  def extract_faster_rcnn_feature(self, filename):
    try:
      image = Image.open(filename)
    except FileNotFoundError:
      return None

    try:
      image_np = self.load_image_into_numpy_array(image)
    except ValueError:
      return None


    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes,feat) = self._sess.run(
                        [self._proposal_boxes,self._feature],
                        feed_dict={self._image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(feat)

  def load_image_into_numpy_array(self, image):
      (im_width, im_height) = image.size

      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        bytes(v, encoding = "utf8") for v in value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _float_feature_list(values):
  """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(bytes(v, encoding = "utf8")) for v in values])


def _int64_list_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_list_feature(v) for v in values])

def _float_list_feature_list(values):
  """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_float_list_feature(v) for v in values])

def _bytes_list_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_list_feature(v) for v in values])

def fix_length_list(lista, fixed_length):
  if len(lista)>fixed_length:
    return lista[:fixed_length]
  elif len(lista)<fixed_length:
    i=0
    while len(lista)<fixed_length:
      lista.append(lista[i])
      i = i + 1
    return lista
  else:
    return lista


def _to_sequence_example(image, decoder, vocab):
  """Builds a SequenceExample proto for an image-caption pair.


  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

  Returns:
    A SequenceExample proto.
  """
  # code from tensorflow/models im2txt modified by zisang 20180418
  # with tf.gfile.FastGFile(image.filename, "rb") as f:
  #   encoded_image = f.read()

  # try:
  #   encoded_image = decoder.decode_jpeg(encoded_image)
  # except (tf.errors.InvalidArgumentError, AssertionError):
  #   print("Skipping file with invalid JPEG data: %s" % image.filename)
  #   return
  
  #
  #if not (len(image.captions) == 5):
  #  print(image.captions)
  #  pass
  image_captions = fix_length_list(image.captions, 5)
  assert len(image_captions) == 5
  try:
    bounding_box, feature_map = decoder.extract_faster_rcnn_feature(image.filename)
  except TypeError:
    return None
  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/filename": _bytes_feature(bytes(image.filename, encoding="utf8")),
      "image/data": _bytes_feature(feature_map.tostring()),
      "iamge/bounding_box": _bytes_feature(bounding_box.tostring())
  })

  img_captions_ids = []
  img_captions_mask = []
  for i in range(len(image_captions)):
    caption = image_captions[i]
    caption_ids = [vocab.word_to_id(word) for word in caption]
    caption_num_words = len(caption_ids)
    if caption_num_words > 21:
      caption_ids = caption_ids[:21]
      caption_num_words = 21

    caption_fix_len = np.zeros(21,dtype = np.int32)
    current_masks = np.zeros(21,dtype=np.float32)
    caption_fix_len[:caption_num_words] = np.array(caption_ids)
    current_masks[:caption_num_words] = 1.0

    img_captions_ids.append(caption_fix_len)
    img_captions_mask.append(current_masks)

  feature_lists = tf.train.FeatureLists(feature_list={
      "iamge/raw_caption":_bytes_feature_list(image.raw_captions),
      "image/caption": _bytes_list_feature_list(image_captions),
      "image/caption_ids": _int64_list_feature_list(img_captions_ids),
      "image/caption_mask": _float_list_feature_list(img_captions_mask)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
  """Processes and saves a subset of images as TFRecord files in one thread.

  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard:
      image = images[i]

      sequence_example = _to_sequence_example(image, decoder, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  """Processes a complete data set and saves it as a TFRecord.

  Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # # Break up each image into a separate entity for each caption.
  # images = [ImageMetadata(image.image_id, image.filename, [caption])
  #           for image in images for caption in image.captions]

  # # Shuffle the ordering of images. Make the randomization repeatable.
  # random.seed(12345)
  # random.shuffle(images)

  # Break the images into num_threads batches. Batch i is defined as
  # images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a utility for decoding JPEG images to run sanity checks.
  decoder = ImageDecoder(FLAGS.graph_path)

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(dataset,filename = 'word_counts.txt'):
  """Creates the vocabulary of word to word_id.

  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.

  Args:
    captions: A list of lists of strings.

  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")

  captions = [c for image in dataset for c in image.captions]

  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  vocab_file_path = os.path.join(FLAGS.output_dir, filename)
  with tf.gfile.FastGFile(vocab_file_path, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", vocab_file_path)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab

def _create_image_id_to_captions(dataset, filename):
  id_to_cap = {}
  for image in dataset:
    id_to_cap[image.image_id] = image.raw_captions

  file_path = os.path.join(FLAGS.output_dir, filename)
  fp = open(file_path, 'w')
  json.dump(id_to_cap, fp)
  fp.close()

def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")
  assert (FLAGS.dataset in LOAD_DATASET_MAP), (
      "Unknown dataset! Must be flickr8k, flickr30k, or coco")
  
  load_dataset = LOAD_DATASET_MAP[FLAGS.dataset]

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  train_dataset,val_dataset,test_dataset = load_dataset(FLAGS)

  # Create vocabulary from the training captions.
  vocab = _create_vocab(train_dataset)
  # Create image id to captions dict for evaluation
  _create_image_id_to_captions(train_dataset,filename='train_id_captions.json')
  _create_image_id_to_captions(val_dataset,filename='val_id_captions.json')
  _create_image_id_to_captions(test_dataset,filename='test_id_captions.json')

  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
  _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
  _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
  tf.app.run()

