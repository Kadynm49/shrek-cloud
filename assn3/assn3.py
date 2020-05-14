import os
from collections import Counter

vocab = []

for filename in os.listdir('data/nonspam-train'):
    for line in open('data/nonspam-train/' + filename):
        for token in line.split():
            vocab.append(token)

for filename in os.listdir('data/spam-train'):
    for line in open('data/spam-train/' + filename):
        for token in line.split():
            vocab.append(token)

vocab_frequencies = Counter(vocab).most_common(2500)
# vocab = vocab_frequencies.keys()
doc_word_frequencies = [{} for i in range(700)]
doc_index = 0

for filename in os.listdir('data/nonspam-train'):
    for line in open('data/nonspam-train/' + filename):
        for token in line.split():
            if (vocab_frequencies[token] != 0):
                doc_word_frequencies[doc_index]

            
# print(vocabulary[2499])

