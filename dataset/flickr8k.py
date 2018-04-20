import os
from common import ImageMetadata, process_caption

def _load_image_filename(filepath):
  with open(filepath,'rb') as f_images:
    imgs = f_images.read().decode().strip().split('\n') 
  return imgs

def _load_captions(captions_file):
  """Loads flickr8k captions.

  Args:
    captions_file: txt file containing caption annotations in 
    '<image file name>#<0-4> <caption>' format

  Returns:
    A dict of image filename to captions.
  """
  f_captions = open(captions_file, 'rb')
  captions = f_captions.read().decode().strip().split('\n')
  data = {}
  for row in captions:
    row = row.split("\t")
    row[0] = row[0][:len(row[0])-2]
    try:
      data[row[0]].append(row[1])
    except:
      data[row[0]] = [row[1]]
  f_captions.close()
  return data

def _to_image_meta_list(image_filenames, captions, image_dir,image_start_id = 0):
  """Process the captions and combine the data into a list of ImageMetadata.

  Args:
    image_filenames: From flickr_8k_train_dataset.txt or Flickr_8k.testImages.txt.
    captions: From Flickr8k.token.txt.
    image_dir: Flickr 8k image dir.
    image_start_id: set to 0 for train images and 8000 for test images

  Returns:
    A list of ImageMetadata.
  """

  image_metadata = []

  num_captions = 0
  image_id = image_start_id

  for base_filename in image_filenames:
    filename = os.path.join(image_dir, base_filename)
    raw_captions = captions[base_filename]
    image_captions = [process_caption(c) for c in raw_captions]
    image_metadata.append(ImageMetadata(image_id, filename, image_captions, raw_captions))
    
    image_id += 1
    num_captions += len(captions)

  return image_metadata

def load_flickr8k_metadata(text_path, image_dir):
  """Loads flickr8k images and captions.

  Args:
    text_path: containing txt files about image caption annotations
    image_dir: Directory containing the image files.

  Returns:
    train_dataset: A list of ImageMetadata.
    test_dataset: A list of ImageMetadata.
  """

  captions = _load_captions(os.path.join(text_path,'Flickr8k.token.txt'))
  train_imgs = _load_image_filename(os.path.join(text_path,'Flickr_8k.trainImages.txt'))
  test_imgs = _load_image_filename(os.path.join(text_path,'Flickr_8k.testImages.txt'))
  
  train_dataset = _to_image_meta_list(train_imgs, captions, image_dir, image_start_id = 0)
  print("Loaded train image metadata for %d images" %len(train_dataset))
  
  test_dataset = _to_image_meta_list(test_imgs, captions, image_dir, image_start_id = 8000)
  print("Loaded test image metadata for %d images" %len(test_dataset))

  return train_dataset, test_dataset

def load_flickr8k_dataset(FLAGS):
  flickr8k_train_dataset, flickr8k_test_dataset = \
      load_flickr8k_metadata(FLAGS.text_path,FLAGS.image_dir)

  # Redistribute the FLICKR8K data as follows:
  #   train_dataset = 100% of flickr8k_train_dataset + 85% of flickr8k_test_dataset.
  #   val_dataset = 5% of flickr8k_test_dataset (for validation during training).
  #   test_dataset = 10% of flickr8k_test_dataset (for final evaluation).
  train_cutoff = int(0.85 * len(flickr8k_test_dataset))
  val_cutoff = int(0.90 * len(flickr8k_test_dataset))

  train_dataset = flickr8k_train_dataset + flickr8k_test_dataset[0:train_cutoff]
  val_dataset = flickr8k_test_dataset[train_cutoff:val_cutoff]
  test_dataset = flickr8k_test_dataset[val_cutoff:]
  
  return train_dataset, val_dataset, test_dataset