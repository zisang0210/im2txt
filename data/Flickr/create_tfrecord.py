import hashlib
import io
import logging
import os
import random
import re
import json
import numpy as np
from PIL import Image

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags


flags.DEFINE_string('image_data_dir', '/home/hillyess/ai/project-image-caption/Flickr8k/Flickr8k_Dataset', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/hillyess/ai/project-image-caption/Flickr8k', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/home/hillyess/ai/project-image-caption/Flickr8k/Flickr8k_text/Flickr8k.token.txt',
                    'Path to label map')
FLAGS = flags.FLAGS

def openAnnotation(path):
    f = open(path, mode='r', encoding="utf-8")
    a = f.readlines()
    path_label_dict = []
    label = []
    for index, txt in enumerate(a):
        k = txt.split('\t')
        s = k[1].split('\n')
        imagename = k[0].split('#')
        imagename[0] = re.match(r".*\.jpg", imagename[0]).group(0)

        if imagename[1] == '0' and label:
            path_label_dict.append((imagename[0], label))
            label = []
        label.append(s[0])
    return path_label_dict



def dict_to_tf_example(label_map_dict):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    label_map_dict: A map from name of image to string labels .
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename = label_map_dict[0]
  img_path = os.path.join(FLAGS.image_data_dir, filename)

  try:
    with tf.gfile.GFile(img_path, 'rb') as fid:
     encoded_jpg = fid.read()
  except:
     logging.warning('Image Not Found %s', img_path)
     return None

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  (witdh, height) = image.size

  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  sentence_txt = label_map_dict[1]


  sentences = []
  f = open('dictionary.json', 'r')
  dictionary = f.read()
  dictionary = json.loads(dictionary)
  for index, _ in enumerate(sentence_txt):
      sentence = []
      for sen in sentence_txt[index].split(' '):
          try:
            sentence.append(dictionary[sen])
          except KeyError:
            sentence.append(dictionary['UNK'])
      sentences.append(sentence)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(witdh),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/score_0': dataset_util.int64_list_feature(sentences[0]),
      'image/score_1': dataset_util.int64_list_feature(sentences[1]),
      'image/score_2': dataset_util.int64_list_feature(sentences[2]),
      'image/score_3': dataset_util.int64_list_feature(sentences[3]),
      'image/score_4': dataset_util.int64_list_feature(sentences[4]),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8'))
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
  """


  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.warning('On image %d of %d', idx, len(examples))
    try:
      tf_example = dict_to_tf_example(label_map_dict[idx])
      writer.write(tf_example.SerializeToString())
    except AttributeError:
      logging.warning('Invalid example: %s, ignoring.', label_map_dict[idx][0])


  writer.close()


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  Annotation = openAnnotation(FLAGS.label_map_path)
  logging.warning('Reading from dataset.')
  random.seed(42)
  id_list = list(range(len(Annotation)))
  random.shuffle(id_list)
  num_examples = len(id_list)
  num_train = int(0.75 * num_examples)
  train_examples = id_list[:num_train]
  val_examples = id_list[num_train:]
  logging.warning('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'flickr_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'flickr_val.record')

  create_tf_record(train_output_path, Annotation, train_examples)
  create_tf_record(val_output_path, Annotation, val_examples)


if __name__ == '__main__':
  tf.app.run()






