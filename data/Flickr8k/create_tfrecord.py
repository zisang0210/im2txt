# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

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
flags.DEFINE_string('data_dir', '/home/hillyess/ai/project-image-caption/Flickr8k', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/hillyess/ai/project-image-caption', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/home/hillyess/ai/project-image-caption/Flickr8k/Flickr8k_text/Flickr8k.token.txt',
                    'Path to label map')
flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
FLAGS = flags.FLAGS

def openAnnotation(path):
    f = open(path, mode='r', encoding="utf-8")
    a = f.readlines()
    path_label=[]
    words = []
    for txt in a:
        k = txt.split('\t')
        s = k[1].split('\n')
        imagename = k[0].split('#')

        imagename[0] = re.match(r".*\.jpg", imagename[0]).group(0)
        label = s[0]

        path_label.append((imagename, label))
    return path_label



def dict_to_tf_example(label_map_dict,
                       image_subdirectory):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename = label_map_dict[0][0]
  importance = label_map_dict[0][1]
  img_path = os.path.join(FLAGS.data_dir, image_subdirectory, filename)

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
  sentence=[]

  f = open('dictionary.json', 'r')
  dictionary = f.read()
  dictionary = json.loads(dictionary)

  for sen in sentence_txt.split(' '):
      sentence.append(dictionary[sen])

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(witdh),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/score': dataset_util.bytes_feature(importance),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/class/sentence': dataset_util.int64_list_feature(sentence)
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
      tf_example = dict_to_tf_example(label_map_dict[idx], 'Flickr8k_Dataset')
      writer.write(tf_example.SerializeToString())
    except AttributeError:
      logging.warning('Invalid example: %s, ignoring.', label_map_dict[idx][0])


  writer.close()


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir

  Annotation = openAnnotation(FLAGS.label_map_path)



  logging.warning('Reading from Pet dataset.')
  image_dir = os.path.join(data_dir, 'Flickr8k_text')
  annotations_dir = os.path.join(data_dir, 'Flickr8k_text/')

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






