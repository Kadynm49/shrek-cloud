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
print("The vocabulary size is: " + str(len(word_frequencies)))

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


# Generate Sentences using unigram model
print("Unsmoothed Unigram Sentences:\n")
# for i in range(10):
#     sentence_length = random.randrange(10, 20)
#     unigrams = unsmoothed_unigrams.most_common(sentence_length)
#     sentence = "<s> "
#     for j in range(sentence_length):
#         if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
#             sentence += str(unigrams[j][0][0]) + " "
#     sentence += " </s>"
#     print(sentence)

# print("\nSmoothed Unigram Sentences:")
# for i in range(10):
#     sentence_length = random.randrange(10, 20)
#     unigrams = smoothed_unigrams.most_common(sentence_length)
#     sentence = "<s> "
#     for j in range(sentence_length):
#         if (str(unigrams[j][0][0]) != "<s>" and str(unigrams[j][0][0]) != "</s>"):
#             sentence += str(unigrams[j][0][0]) + " "
#     sentence += " </s>"
    # print(sentence)
    
    
# Generate Sentences using Bigram model

# Create a list of sentence starts for unsmoothed bigrams
sentence_starters = []
for bigram in unsmoothed_bigrams:
    if(bigram[0] == "<s>"):
        sentence_starters.append(bigram[1])
print(sentence_starters)
# for i in range(10):
    
            
