# import pandas as pd
import string
import time
# import numpy as np
import re
import nltk
from nltk.util import ngrams
from collections import Counter
import copy
import random
from math import log, exp
# from IPython.display import clear_output

start = " <s> "
end = " </s> \n"
sentences = []
word_frequencies = {}

start_time = time.time()
# cleanedTrain.txt is a modified version of train.txt to have sentence markers 
# and <UNK> replacing words with a frequency lower than 3
with open("data/cleanedTrain.txt", "r") as file:
    for line in file:
        # sentences.append(start + line[:-1] + end)
        sentences.append(line[:-1])
        

text_list = " ".join(map(str, sentences))

for word in text_list.split():
    try:
        word_frequencies[word] += 1
    except:
        word_frequencies[word] = 1

# Report size of vocabulary
print("\nVocabulary size: " + str(len(word_frequencies)) + "\n")

# Get all tokens for ngrams
tokens = []
for word in text_list.split(): 
    tokens.append(word)

# Create counts of unsmoothed ngrams
unsmoothed_unigrams = Counter(ngrams(tokens, 1))
unsmoothed_bigrams = Counter(ngrams(tokens, 2))
unsmoothed_trigrams = Counter(ngrams(tokens, 3))

# Create counts of smoothed ngrams by copying unsmoothed and adding 1 to each count
smoothed_unigrams = copy.deepcopy(unsmoothed_unigrams)
smoothed_bigrams = copy.deepcopy(unsmoothed_bigrams)
smoothed_trigrams = copy.deepcopy(unsmoothed_trigrams)

for unigram in smoothed_unigrams:
    smoothed_unigrams[unigram] += 1
for bigram in smoothed_bigrams:
    smoothed_bigrams[bigram] += 1
for trigram in smoothed_trigrams:
    smoothed_trigrams[trigram] += 1

# Convert the ngram dictionaries to contain their probabilities instead of frequencies.
unigram_count = len(unsmoothed_unigrams)
bigram_count = len(unsmoothed_bigrams)
trigram_count = len(unsmoothed_trigrams)

for unigram in smoothed_unigrams:
    smoothed_unigrams[unigram] /= unigram_count
    unsmoothed_unigrams[unigram] /= unigram_count
for bigram in smoothed_bigrams:
    smoothed_bigrams[bigram] /= bigram_count
    unsmoothed_bigrams[bigram] /= bigram_count
for trigram in smoothed_trigrams:
    smoothed_trigrams[trigram] /= trigram_count
    unsmoothed_trigrams[trigram] /= trigram_count


# Generate Sentences using unsmoothed unigram model
print("Unsmoothed Unigram Sentences:")
for i in range(10):
    sentence_length = random.randrange(10, 20)
    unigrams = unsmoothed_unigrams.most_common(sentence_length)
    sentence = "<s> "
    for j in range(sentence_length):
        if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
            sentence += str(unigrams[j][0][0]) + " "
    sentence += " </s>"
    print(sentence)

# Generate Sentences using unsmoothed unigram model
print("\nSmoothed Unigram Sentences:")
for i in range(10):
    sentence_length = random.randrange(10, 20)
    unigrams = smoothed_unigrams.most_common(sentence_length)
    sentence = "<s> "
    for j in range(sentence_length):
        if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
            sentence += str(unigrams[j][0][0]) + " "
    sentence += " </s>"
    print(sentence)

# Create a list of sentence starts for unsmoothed bigrams
# STORE PROBABILITIES TOOOOOOOOOOOO
sentence_starters = []
for bigram in unsmoothed_bigrams.most_common():
    if bigram[0][0] == "<s>":
        sentence_starters.append(bigram[0][1])
    if len(sentence_starters) == 30:
        break

# Generate sentences using unsmoothed bigram model
print("\nUnsmoothed Bigram Sentences:")
for i in range(10):
    current_word = sentence_starters[random.randrange(0,20)]
    sentence = "<s> "
    word_count = 0
    while current_word != "</s>":
        if word_count == 20:
            break

        sentence += current_word + " "
        next_word = { "prob" : 0, "word": "" }

        # Find most likely word to come after current word
        for bigram in unsmoothed_bigrams:
            # Set most likely word if this bigram has the highest probability found so far (25% chance it will not set)
            if bigram[0] == current_word and unsmoothed_bigrams[bigram] > next_word["prob"] and random.randrange(1, 5) == 1:
                next_word["prob"] = unsmoothed_bigrams[bigram]
                next_word["word"] = bigram[1]

        current_word = next_word["word"]
        word_count += 1
    sentence += "</s>"
    print(sentence)

# Create a list of sentence starts for smoothed bigrams
sentence_starters = []
for bigram in smoothed_bigrams.most_common():
    if bigram[0][0] == "<s>":
        sentence_starters.append(bigram[0][1])
    if len(sentence_starters) == 30:
        break

# Generate sentences using smoothed bigram model
print("\nSmoothed Bigram Sentences:")
for i in range(10):
    current_word = sentence_starters[random.randrange(0,20)]
    sentence = "<s> "
    word_count = 0
    while current_word != "</s>":
        if word_count == 20:
            break

        sentence += current_word + " "
        next_word = { "prob" : 0, "word": "" }

        # Find most likely word to come after current word
        for bigram in smoothed_bigrams:
            # Set most likely word if this bigram has the highest probability found so far (25% chance it will not set)
            if bigram[0] == current_word and smoothed_bigrams[bigram] > next_word["prob"] and random.randrange(1, 5) > 1:
                next_word["prob"] = smoothed_bigrams[bigram]
                next_word["word"] = bigram[1]

        current_word = next_word["word"]
        word_count += 1
    sentence += "</s>"
    print(sentence)
