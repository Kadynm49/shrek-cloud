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
    sentence_p = 0
    for j in range(sentence_length):
        if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
            sentence += str(unigrams[j][0][0]) + " "
            sentence_p += log(unigrams[j][1])
    sentence += " </s>"
    print(sentence)
    print("Sentence Probability (Log Space): ", sentence_p, "\n")

# Generate Sentences using unsmoothed unigram model
print("\nSmoothed Unigram Sentences:")
for i in range(10):
    sentence_length = random.randrange(10, 20)
    unigrams = smoothed_unigrams.most_common(sentence_length)
    sentence = "<s> "
    sentence_p = 0
    for j in range(sentence_length):
        if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
            sentence += str(unigrams[j][0][0]) + " "
            sentence_p += log(unigrams[j][1])
    sentence += " </s>"
    print(sentence)
    print("Sentence Probability (Log Space): ", sentence_p, "\n")

# Create a list of sentence starts for unsmoothed bigrams
sentence_starters = []
for bigram in unsmoothed_bigrams.most_common():
    if bigram[0][0] == "<s>":
        sentence_starters.append((bigram[0][1], bigram[1]))
    if len(sentence_starters) == 30:
        break

# Generate sentences using unsmoothed bigram model
print("\nUnsmoothed Bigram Sentences:")
for i in range(10):
    current_word = sentence_starters[random.randrange(0,20)]
    sentence = "<s> "
    word_count = 0
    sentence_p = 0
    while current_word[0] != "</s>":
        if word_count == 20:
            break
        sentence += current_word[0] + " "
        if (current_word[1] != 0):
            sentence_p += log(current_word[1])
        next_word = { "prob" : 0, "word": "" }

        # Find most likely word to come after current word
        for bigram in unsmoothed_bigrams:
            # Set most likely word if this bigram has the highest probability found so far (25% chance it will not set)
            if bigram[0] == current_word[0] and unsmoothed_bigrams[bigram] > next_word["prob"] and random.randrange(1, 5) == 1:
                next_word["prob"] = unsmoothed_bigrams[bigram]
                next_word["word"] = bigram[1]

        current_word = (next_word["word"], next_word["prob"])
        word_count += 1
    sentence += "</s>"
    print(sentence)
    print("Sentence Probability (Log Space): ", sentence_p, "\n")
    

# Create a list of sentence starts for smoothed bigrams
sentence_starters = []
for bigram in smoothed_bigrams.most_common():
    if bigram[0][0] == "<s>":
        sentence_starters.append((bigram[0][1], bigram[1]))
    if len(sentence_starters) == 30:
        break

# Generate sentences using smoothed bigram model
print("\nSmoothed Bigram Sentences:")
for i in range(10):
    current_word = sentence_starters[random.randrange(0,20)]
    sentence = "<s> "
    word_count = 0
    sentence_p = 0
    while current_word[0] != "</s>":
        if word_count == 20:
            break

        sentence += current_word[0] + " "
        if (current_word[1] != 0):
            sentence_p += log(current_word[1])
        next_word = { "prob" : 0, "word": "" }

        # Find most likely word to come after current word
        for bigram in smoothed_bigrams:
            # Set most likely word if this bigram has the highest probability found so far (25% chance it will not set)
            if bigram[0] == current_word[0] and smoothed_bigrams[bigram] > next_word["prob"] and random.randrange(1, 5) > 1:
                next_word["prob"] = smoothed_bigrams[bigram]
                next_word["word"] = bigram[1]

        current_word = (next_word["word"], next_word["prob"])
        word_count += 1
    sentence += "</s>"
    print(sentence)
    print("Sentence Probability (Log Space): ", sentence_p, "\n")

#Trigram Model
sentence_starters = []
for trigram in unsmoothed_trigrams.most_common():
    if trigram[0][0] == "<s>":
        sentence_starters.append((trigram[0], trigram[1]))
    if len(sentence_starters) == 30:
        break
    
for i in range(10):
    current_word = sentence_starters[random.randrange(0,20)]
    current_word = (("<s>", "<s>", current_word[0][1]), current_word[1])
    sentence = "<s> "
    word_count = 0
    sentence_p = 0
    while current_word[0][2] != "</s>":
        if word_count == 20:
            break

        sentence += current_word[0][2] + " "
        sentence_p += log(current_word[1])
        next_word = { "prob" : 0, "word": "" }
        next_word_candidates = []
        # Find most likely word to come after current word
        for trigram in unsmoothed_trigrams:
            # Set most likely word if this trigram has the highest probatrility found so far (25% chance it will not set)
            # print(trigram)
            if trigram[1] == current_word[0][2] and trigram[0] == current_word[0][1] and unsmoothed_trigrams[trigram] > next_word["prob"] and len(next_word_candidates) <= 5:
                next_word_candidates.append({"prob": unsmoothed_trigrams[trigram], "word":trigram})

        next_word = next_word_candidates[random.randrange(len(next_word_candidates))]
        current_word = (next_word["word"], next_word["prob"])
        # print(current_word)

        word_count += 1
    sentence += "</s>"
    print(sentence)
    print("Sentence Probability (Log Space): ", sentence_p, "\n")
    
