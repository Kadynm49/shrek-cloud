import os
from collections import Counter
from math import log
import json

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
vocab_frequencies_dict = {}
vocab_total_word_frequency = 0

for tuple in vocab_frequencies:
    vocab_frequencies_dict[tuple[0]] = tuple[1]
    vocab_total_word_frequency += tuple[1]

non_spam_word_frequencies = {}
spam_word_frequencies = {}
non_spam_word_probabilities = {}
spam_word_probabilities = {}

# Get frequencies for all non spam words that appear in vocab
for filename in os.listdir('data/nonspam-train'):
    for line in open('data/nonspam-train/' + filename):
        for token in line.split():
            if (token in vocab_frequencies_dict):
                if (token in non_spam_word_frequencies):
                    non_spam_word_frequencies[token] += 1
                else: 
                    non_spam_word_frequencies[token] = 1

# Get frequencies for all non spam words that appear in vocab
for filename in os.listdir('data/spam-train'):
    for line in open('data/spam-train/' + filename):
        for token in line.split():
            if (token in vocab_frequencies_dict):
                if (token in spam_word_frequencies):
                    spam_word_frequencies[token] += 1
                else: 
                    spam_word_frequencies[token] = 1

# Convert frequencies to probabilities in log space
for frequency in non_spam_word_frequencies:
    non_spam_word_probabilities[frequency] = log(non_spam_word_frequencies[frequency]/vocab_total_word_frequency)

for frequency in spam_word_frequencies:
    spam_word_probabilities[frequency] = log(spam_word_frequencies[frequency]/vocab_total_word_frequency)    

with open('nonspam-word-probabilites.json', 'w') as file:
    file.write(json.dumps(non_spam_word_probabilities))
    file.close()

with open('spam-word-probabilites.json', 'w') as file:
    file.write(json.dumps(spam_word_probabilities))
    file.close()

# with open('vocab-frequency.json', 'w') as file:
#     file.write(json.dumps(vocab_frequencies_dict))
#     file.close()

