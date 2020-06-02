# Get movie from user
print("What movie or TV show do you want reviewed?")
title = input()

import requests
import sys
import json
from bs4 import BeautifulSoup
from bs4 import NavigableString
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams


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
    imdbID = ""
    if "imdbID" in responseJson:
        imdbID = responseJson["imdbID"]
    else: 
        print("Invalid title, please try a different movie or show.")
        exit()

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
    print ("\n==================================\nFetching reviews from IMDB\n==================================")
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

    print("I rate " + title + " %.1f/10\n" % avg_score)

    create_review_with_markov_chains(reviews_string.split(), avg_review_len, title)

    # create_review_with_tensorflow(reviews)

    create_review_with_n_grams(reviews_string.split())

##########################################
# Markov chain generated text source 
# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])

# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def create_review_with_markov_chains(corpus, review_len, title):
    print("\n==================================\nGenerating review with Markov chains\n")

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
        if (chain[-1] in word_dict): 
            next_word = np.random.choice(word_dict[chain[-1]])
        else: 
            next_word = np.random.choice(corpus)
        chain.append(next_word)
        if i >= review_len and (next_word[-1] == '.' or next_word[-1] == '!' or next_word[-1] == '?'):
            break
        i += 1

    review = ' '.join(chain)
    print(review + "\n")

##########################################
# N grams generated text source 
# script from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
def create_review_with_n_grams(corpus): 
    print("\n==================================\nGenerating review with trigrams\n")
    trigrams = list(ngrams(corpus, 3))
    unsmoothed_trigrams = Counter(ngrams(corpus, 3))
    
    trigram_count = len(unsmoothed_trigrams)

    for trigram in unsmoothed_trigrams:
        unsmoothed_trigrams[trigram] /= trigram_count
    
    randint = np.random.randint(trigram_count)
    
    while trigrams[randint][2].islower():
        randint = np.random.randint(trigram_count)

    first_word = (trigrams[randint], unsmoothed_trigrams[trigrams[randint]])

    review = ""
    # Generate sentences using unsmoothed trigram model
    for i in range(10):
        randint = np.random.randint(trigram_count)
        
        while trigrams[randint][2].islower():
            randint = np.random.randint(trigram_count)

        first_word = (trigrams[randint], unsmoothed_trigrams[trigrams[randint]])

        current_word = first_word
        sentence = ""
        word_count = 0
        while current_word[0][2][-1] != ".":
            if word_count == 20:
                break

            sentence += current_word[0][2] + " "
            next_word = { "prob" : 0, "word": "" }
            next_word_candidates = []
            # Find most likely word to come after current word
            for trigram in unsmoothed_trigrams:
                # Set most likely word if this trigram has the highest probability found so far
                if trigram[1] == current_word[0][2] and trigram[0] == current_word[0][1] and unsmoothed_trigrams[trigram] > next_word["prob"]:
                    next_word["prob"] = unsmoothed_trigrams[trigram]
                    next_word_candidates.append({"prob": unsmoothed_trigrams[trigram], "word":trigram})

            next_word_candidates = sorted(next_word_candidates, key = lambda i: i['prob']) 
            if (len(next_word_candidates) >= 5):
                next_word = next_word_candidates[random.randrange(0, 5)]
            else:
                next_word = next_word_candidates[random.randrange(0, len(next_word_candidates))]        
            current_word = (next_word["word"], next_word["prob"])

            word_count += 1
        sentence += current_word[0][2] + " "
        review += sentence
    
    print(review + "\n")

##########################################
# Neural network text generation 
# https://github.com/minimaxir/textgenrnn
def create_review_with_tensorflow(reviews):
    print("\n==================================\nGenerating review with a neural network\n")
    from textgenrnn import textgenrnn
    textgen = textgenrnn()
    textgen.train_on_texts(reviews, num_epochs=1)

if __name__ == '__main__': 
    main()