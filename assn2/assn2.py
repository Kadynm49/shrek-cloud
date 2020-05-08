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
        sentences.append(start + line[:-1] + end)

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


print(unsmoothed_bigrams[('includes', 'charge')])
print(smoothed_bigrams[('includes', 'charge')])

# Generate Sentences using unigram model
for i in range(10):
    sentence_length = random.randrange(10, 20)
    unigrams = unsmoothed_unigrams.most_common(sentence_length)
    sentence = ""
    for j in range(sentence_length):
        sentence += str(unigrams[j][0][0]) + " "
    print(sentence)

# text = train_data['sentence_clean']
# text_list = " ".join(map(str, text))
# text_list[0:100]

# word_list = pd.DataFrame({'words':text.str.split(' ', expand = True).stack().unique()})   

# word_count_table = pd.DataFrame()
# for n,word in enumerate(word_list['words']):
#     # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
#     # e.g. 'ear' would be counted in each appearance of the word 'year'
#     word_count = len(re.findall(' ' + word + ' ', text_list))  
#     word_count_table = word_count_table.append(pd.DataFrame({'count':word_count}, index=[n]))
    
#     clear_output(wait=False)
#     print('Proportion of words completed:', np.round(n/len(word_list),4)*100,'%')

# word_list['count'] = word_count_table['count']
# # Remove the count for the start and end of sentence notation so 
# # that these do not inflate the other probabilities
# word_list['count'] = np.where(word_list['words'] == '<s' , 0,
#                      np.where(word_list['words'] == '/s>', 0,
#                      word_list['count']))


# word_list['prob'] = word_list['count']/sum(word_list['count'])
# print(word_list.head().to_string())

# ##next time:
# ##report size, add unknown tkn, bigram/unigram models


# end_time = time.time()  
# print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')
