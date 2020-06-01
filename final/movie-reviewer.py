# Get movie from user
print("What movie do you want reviewed?")
title = input()

import requests
import sys
import json
from bs4 import BeautifulSoup
from bs4 import NavigableString
import numpy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint

from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random

def main(): 
    # Find imdb title key for this movie using the omdb API
    omdb_key = "a542cac9"
    link = "http://www.omdbapi.com/?apikey=" + omdb_key
    t = "&t=" + title
    response = requests.get(link + t)
    if response.status_code != 200:
        print("The movie you entered is not in IMDB")
        exit()
    responseJson = json.loads(response.text)
    imdbID = responseJson["imdbID"]

    # Get imdb user reviews page
    URL = 'https://www.imdb.com/title/' + imdbID + '/reviews/'
    load_more_url = 'https://www.imdb.com/title/' + imdbID + '/reviews/_ajax?paginationKey='
    r = requests.get(url = URL)
    soup = BeautifulSoup(r.text, 'html.parser')

    # Remove br tags
    for e in soup.findAll('br'):
        e.extract()

    # Find load more button
    load_more_button = soup.find(class_ = "load-more-data")
    avg_score = 0

    for scale in soup.find_all(class_ = "point-scale"):
        score = scale.find_previous_sibling("span")
        avg_score += int(score.contents[0])

    # Get reviews by loading all pages of user reviews
    # IMDB uses a "Load More" button to retrieve 25 
    # reviews at a time, so we must find the paginationKey 
    # for each page to get the next 25 reviews and append 
    # those reviews to the list. Stop when there is no 
    # longer a "Load More" button
    reviews = []
    print ("Fetching all IMDB reviews\n")
    while (load_more_button != None):
        review_containers = soup.find_all("div", class_="show-more__control")
        for review_container in review_containers:
            if len(review_container.contents) > 0 and type(review_container.contents[0]) == NavigableString:
                review = review_container.contents[0]
                if len(review) > 1:
                    reviews.append(review)

        for scale in soup.find_all(class_ = "point-scale"):
            score = scale.find_previous_sibling("span")
            avg_score += int(score.contents[0])

        URL = load_more_url + load_more_button['data-key']
        r = requests.get(url = URL)
        soup = BeautifulSoup(r.text, 'html.parser')
        for e in soup.findAll('br'):
            e.extract()
        load_more_button = soup.find(class_="load-more-data")

    avg_score /= len(reviews)
    
    # Calculate average review length
    avg_review_len = 0
    for review in reviews:
        avg_review_len += len(review.split())
    avg_review_len //= len(reviews)

    # All reviews in one string
    reviews_string = " ".join(reviews)

    # Uncomment for tensorflow review (not working well and takes a long time to run)
    # processed_inputs_string  = tokenize_words(reviews_string)
    # create_review_with_tensorflow(processed_inputs, avg_score)

    create_review_with_markov_chains(reviews_string.split(), avg_review_len, avg_score)
    
    # Uncomment for n grams review
    # create_review_with_n_grams(reviews_string)

##########################################
# Markov chain generated text source 
# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])

# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def create_review_with_markov_chains(corpus, review_len, rating):
    import numpy as np
            
    pairs = make_pairs(corpus)

    word_dict = {}

    for word_1, word_2 in pairs:
        if word_1 in word_dict.keys():
            word_dict[word_1].append(word_2)
        else:
            word_dict[word_1] = [word_2]
    

    first_word = np.random.choice(corpus)

    while first_word.islower():
        first_word = np.random.choice(corpus)

    chain = [first_word]

    i = 0
    while True:
        next_word = np.random.choice(word_dict[chain[-1]])
        chain.append(next_word)
        if i >= review_len and (next_word[-1] == '.' or next_word[-1] == '!' or next_word[-1] == '?'):
            break
        i += 1

    review = ' '.join(chain)

    print("I rate this movie a %.1f/10" % rating)
    print(review)

##########################################
# N grams generated text source 
# script from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
def create_review_with_n_grams(reviews_string):
    from math import log
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance  
    for w1, w2, w3 in trigrams(reviews_string, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
    
    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] = log(model[w1_w2][w3]/total_count)
    
    # code courtesy of https://nlpforhackers.io/language-models/

    # starting words
    text = ["Shrek"]
    sentence_finished = False
    
    while not sentence_finished:
        # select a random probability threshold  
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True
        
        print (' '.join([t for t in text if t]))
        
##########################################
# Machine learning generated text source 
# https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

def create_review_with_tensorflow(processed_inputs, rating):
    chars = sorted(list(set(processed_inputs)))
    char_to_num = dict((c, i) for i, c in enumerate(chars))

    input_len = len(processed_inputs)
    vocab_len = len(chars)
    print ("Total number of characters:", input_len)
    print ("Total vocab:", vocab_len)

    seq_length = 100
    x_data = []
    y_data = []

    # loop through inputs, start at the beginning and go until we hit
    # the final character we can create a sequence out of
    for i in range(0, input_len - seq_length, 1):
        # Define input and output sequences
        # Input is the current character plus desired sequence length
        in_seq = processed_inputs[i:i + seq_length]

        # Out sequence is the initial character plus total sequence length
        out_seq = processed_inputs[i + seq_length]

        # We now convert list of characters to integers based on
        # previously and add the values to our lists
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])

    n_patterns = len(x_data)
    print ("Total Patterns:", n_patterns)

    X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)

    y = to_categorical(y_data)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))

    # If weight file doesn't exist
    # model.compile(loss='categorical_crossentropy', optimizer='adam')

    # filepath = title + "_model_weights_saved.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # desired_callbacks = [checkpoint]

    # model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)

    filename = title + "_model_weights_saved.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    num_to_char = dict((i, c) for i, c in enumerate(chars))

    start = numpy.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("Random Seed:")
    print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

    print("I rate this movie a %.1f/10" % rating)
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = num_to_char[index]
        seq_in = [num_to_char[value] for value in pattern]

        sys.stdout.write(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

if __name__ == '__main__': 
    main()