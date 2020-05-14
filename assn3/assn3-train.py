import os
from collections import Counter
from math import log
import json

if __name__ == '__main__': 
    vocab = []

    # Create vocab by reading each token of training docs
    for filename in os.listdir('data/nonspam-train'):
        for line in open('data/nonspam-train/' + filename):
            for token in line.split():
                vocab.append(token)

    for filename in os.listdir('data/spam-train'):
        for line in open('data/spam-train/' + filename):
            for token in line.split():
                vocab.append(token)

    # Limit vocab to 2500 most common words
    vocab_frequencies = Counter(vocab).most_common(2500)
    vocab_frequencies_dict = {}
    vocab_total_word_frequency = 0

    # Create dictionary for vocabulary words and their frequencies
    for tuple in vocab_frequencies:
        vocab_frequencies_dict[tuple[0]] = tuple[1]
        vocab_total_word_frequency += tuple[1]

    nonspam_word_frequencies = {}
    spam_word_frequencies = {}
    nonspam_word_probabilities = {}
    spam_word_probabilities = {}
    spam_total_word_frequency = 0
    nonspam_total_word_frequency = 0

    # Get frequencies for all non spam words that appear in vocab
    for filename in os.listdir('data/nonspam-train'):
        for line in open('data/nonspam-train/' + filename):
            for token in line.split():
                if (token in vocab_frequencies_dict):
                    if (token in nonspam_word_frequencies):
                        nonspam_word_frequencies[token] += 1
                    else: 
                        nonspam_word_frequencies[token] = 1
                    nonspam_total_word_frequency += 1

    # Get frequencies for all non spam words that appear in vocab
    for filename in os.listdir('data/spam-train'):
        for line in open('data/spam-train/' + filename):
            for token in line.split():
                if (token in vocab_frequencies_dict):
                    if (token in spam_word_frequencies):
                        spam_word_frequencies[token] += 1
                    else: 
                        spam_word_frequencies[token] = 1
                    spam_total_word_frequency += 1

    # Add frequencies of 0 for the words that are in vocabulary but not in spam/nonspam documents
    for word in vocab_frequencies_dict:
        if word not in nonspam_word_frequencies:
            nonspam_word_frequencies[word] = 0
        if word not in spam_word_frequencies:
            spam_word_frequencies[word] = 0

    # Convert frequencies to probabilities in log space
    for frequency in nonspam_word_frequencies:
        nonspam_word_probabilities[frequency] = log((1 + nonspam_word_frequencies[frequency])/(nonspam_total_word_frequency + vocab_total_word_frequency))

    for frequency in spam_word_frequencies:
        spam_word_probabilities[frequency] = log((1 + spam_word_frequencies[frequency])/(spam_total_word_frequency + vocab_total_word_frequency))    

    # Save probability dictionaries and vocab in json files
    with open('nonspam-word-probabilities.json', 'w') as file:
        file.write(json.dumps(nonspam_word_probabilities))
        file.close()

    with open('spam-word-probabilities.json', 'w') as file:
        file.write(json.dumps(spam_word_probabilities))
        file.close()

    with open('vocab.json', 'w') as file:
        file.write(json.dumps(list(vocab_frequencies_dict.keys())))
        file.close()