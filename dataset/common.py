from collections import namedtuple
import nltk.tokenize

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions", "raw_captions"])


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id

def process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_caption = ["<S>"]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append("</S>")
  return tokenized_caption