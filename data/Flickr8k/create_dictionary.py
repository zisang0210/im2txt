import parser
import sys
import collections
import json

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

vocabulary = opendata('/home/hillyess/ai/project-image-caption/Flickr8k/Flickr8k_text/Flickr8k.token.txt')
vocabulary_size = 10000
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

fp = open('dictionary.json', 'a')
json.dump(dictionary, fp, ensure_ascii=False)
fp.close()
fp = open('reverse_dictionary.json', 'a')
json.dump(reverse_dictionary, fp, ensure_ascii=False)
fp.close()





