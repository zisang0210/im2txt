from flickr8k import _load_image_filename,_load_captions

def test_load_image_filename():
  imgs = _load_image_filename('/home/zisang/Documents/code/data/Flicker8k/Flickr_8k.trainImages.txt')
  print(imgs[0:10])
  print(len(imgs))

def test_load_captions():
  captions = _load_captions('/home/zisang/Documents/code/data/Flicker8k/Flickr8k.token.txt')
  print(captions)

# test_load_image_filename()
test_load_captions()