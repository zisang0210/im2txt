import collections
import json
import tensorflow as tf
import os


flags = tf.app.flags
flags.DEFINE_string('token_path', '/home/hillyess/ai/project-image-caption/flickr30k_images/results_20130124.token', 'Root directory to raw dataset.')
flags.DEFINE_integer('vocabulary_size', 5000, 'Vocabulary size of dictionary')

token_path = flags.FLAGS.token_path
vocabulary_size = flags.FLAGS.vocabulary_size


def opendata(path):
    f = open(path, mode='r', encoding="utf-8")
    a = f.readlines()
    words = []
    for txt in a:
        k = txt.split('\t')
        s = k[1].split('\n')
        s = s[0].split(' ')
        for w in s:
            words.append(w)
    return words

vocabulary = opendata(token_path)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    if n_words:
        count.extend(collections.Counter(words).most_common(n_words - 1))
    else:
        count.extend(collections.Counter(words))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print('length of dictionary:', len(dictionary))

if os.path.exists('dictionary.json'):
    os.remove('dictionary.json')
fp = open('dictionary.json', 'a')
json.dump(dictionary, fp, ensure_ascii=False)
fp.close()

if os.path.exists('reverse_dictionary.json'):
    os.remove('reverse_dictionary.json')
fp = open('reverse_dictionary.json', 'a')
json.dump(reverse_dictionary, fp, ensure_ascii=False)
fp.close()





