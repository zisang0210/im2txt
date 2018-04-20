import tensorflow as tf
import json

import os
from common import ImageMetadata, process_caption

def _load_coco_metadata(captions_file, image_dir):
  """Loads image metadata from a JSON file and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.

  Returns:
    A list of ImageMetadata.
  """
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the filenames.
  id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(caption)

  assert len(id_to_filename) == len(id_to_captions)
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    raw_captions = id_to_captions[image_id]
    captions = [process_caption(c) for c in raw_captions]
    image_metadata.append(ImageMetadata(image_id, filename, captions, raw_captions))
    num_captions += len(captions)
  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))

  return image_metadata

def load_coco_dataset(FLAGS):
  # Load image metadata from caption files.
  mscoco_train_dataset = _load_coco_metadata(FLAGS.train_captions_file,
                                                    FLAGS.train_image_dir)
  mscoco_val_dataset = _load_coco_metadata(FLAGS.val_captions_file,
                                                  FLAGS.val_image_dir)

  # Redistribute the MSCOCO data as follows:
  #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
  #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
  #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
  train_cutoff = int(0.85 * len(mscoco_val_dataset))
  val_cutoff = int(0.90 * len(mscoco_val_dataset))
  train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
  val_dataset = mscoco_val_dataset[train_cutoff:val_cutoff]
  test_dataset = mscoco_val_dataset[val_cutoff:]
  return train_dataset, val_dataset, test_dataset