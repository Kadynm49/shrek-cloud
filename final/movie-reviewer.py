import requests
import sys
import json
from bs4 import BeautifulSoup
from bs4 import NavigableString
from bs4 import Tag
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random
this pr is for testing purposes

def main(): 
    while (True):
        # Get movie from user
        print("What movie or TV show do you want reviewed?")
        title = input()

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

        # Get review type from user
        print("Enter the type of review would you like to generate (space seperated): \"p\" - positive, \"n\" - negative, \"a\" - average")
        review_types = input().split()
        
        # Get desired models from user
        print("Enter the models would you like to run to generate the reviews (space seperated): \"1\" - Trigram, \"2\" - Markov Chain, \"3\" - Neural Network")
        models = input().split()
        
        # Get imdb user reviews page
        URL = "https://www.imdb.com/title/" + imdbID + "/reviews/"
        load_more_url = "https://www.imdb.com/title/" + imdbID + "/reviews/_ajax?paginationKey="
        r = requests.get(url = URL)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove br, ul, ol, li, a tags
        for e in soup.find_all("br"):
            e.extract()
        for e in soup.find_all("ul"):
            e.extract()
        for e in soup.find_all("ol"):
            e.extract()
        for e in soup.find_all("li"):
            e.extract()
        for e in soup.find_all("a"):
            e.extract()


        # Find load more button
        load_more_button = soup.find(class_ = "load-more-data")
        if ("data-key" not in load_more_button.attrs):
            print("IMDB has no user reviews for this. See for yourself:", URL)
            exit()

        reviews = []
        review_titles = []
        avg_score = 0

        negative_reviews = []
        negative_review_titles = []
        avg_negative_score = 0

        positive_reviews = []
        positive_review_titles = []
        avg_positive_score = 0

        # Get reviews by loading all pages of user reviews
        # IMDB uses a "Load More" button to retrieve 25 
        # reviews at a time, so we must find the paginationKey 
        # for each page to get the next 25 reviews and append 
        # those reviews to the list. Stop when there is no 
        # longer a "Load More" button
        
        print ("\n==================================\nFetching reviews from IMDB\n==================================")
        while (load_more_button != None):
            review_containers = soup.find_all("div", class_="lister-item-content")
            for review_container in review_containers:
                # Get review score
                score = None
                for content in review_container.contents:
                    if type(content) == Tag and content["class"][0] == "ipl-ratings-bar":
                        score = int(list(filter(lambda x: type(x) == Tag and x["class"][0] == "rating-other-user-rating", content.contents))[0].contents[3].contents[0])
                        if (score >= 5):
                            avg_positive_score += score
                        else:
                            avg_negative_score += score
                        avg_score += score

                # Get review and review"s title
                for content in review_container.contents:
                    if type(content) == Tag \
                        and content["class"][0] == "content" \
                        and len(content.contents) > 0 \
                        and len(content.contents[1].contents) > 0 \
                        and type(content.contents[1].contents[0]) == NavigableString: # failed on this line for Avengers: Endgame
                        
                        review = None
                        if len(content.contents[1].contents) == 1:
                            review = content.contents[1].contents[0]
                        else:
                            review = " ".join(content.contents[1].contents) 
                        
                        if (score is not None and score >= 5):
                            positive_reviews.append(review)
                        elif (score is not None):
                            negative_reviews.append(review)
                        reviews.append(review)

                    if type(content) == Tag and content["class"][0] == "title":
                        if score is not None and score >= 5:
                            positive_review_titles.append(content.contents[0])
                        elif score is not None:
                            negative_review_titles.append(content.contents[0])
                        review_titles.append(content.contents[0])

            # Get url for next 25 reviews and load them in
            URL = load_more_url + load_more_button["data-key"]
            r = requests.get(url = URL)
            soup = BeautifulSoup(r.text, "html.parser")

            # Remove unnecessary tags
            for e in soup.find_all("br"):
                e.extract()
            for e in soup.find_all("ul"):
                e.extract()
            for e in soup.find_all("ol"):
                e.extract()
            for e in soup.find_all("li"):
                e.extract()
            for e in soup.find_all("a"):
                e.extract()

            # Find next load more button
            load_more_button = soup.find(class_="load-more-data")

        avg_score /= len(reviews)
        avg_positive_score /= len(positive_reviews)
        avg_negative_score /= len(negative_reviews)
        
        # Calculate average review length
        avg_review_len = 0
        avg_positive_review_len = 0
        avg_negative_review_len = 0
        for review in reviews:
            review_len = len(review.split())
            avg_review_len += review_len

            if review in positive_reviews:
                avg_positive_review_len += review_len
            elif review in negative_reviews:
                avg_negative_review_len += review_len
        avg_review_len //= len(reviews)
        avg_positive_review_len //= len(positive_reviews)
        avg_negative_review_len //= len(negative_reviews)

        # All reviews in one string
        reviews_string = " ".join(reviews)
        positive_reviews_string = " ".join(positive_reviews)
        negative_reviews_string = " ".join(negative_reviews)

        for model in models:
            if (model == "1"):
                for review_type in review_types:
                    if review_type == "a":
                        print("==================================\nGenerating average review with trigram model\n")
                        create_review_with_trigrams(reviews_string.split(), avg_review_len, title, avg_score)
                    elif review_type == "p":
                        print("==================================\nGenerating positive review with trigram model\n")
                        create_review_with_trigrams(positive_reviews_string.split(), avg_positive_review_len, title, avg_positive_score)
                    elif review_type == "n":
                        print("==================================\nGenerating negative review with trigram model\n")
                        create_review_with_trigrams(negative_reviews_string.split(), avg_negative_review_len, title, avg_negative_score)
                    else:
                        print(review_type, "is not a valid option for review type.")
            elif (model == "2"):
                for review_type in review_types:
                    if review_type == "a":
                        print("==================================\nGenerating average review with markov model algorithm 1\n")
                        # create_review_with_markov_chains(reviews_string.split(), avg_review_len, title, avg_score)
                        # print("algorithm 2:")
                        create_review_with_markov_chains_2(reviews_string.split(), avg_review_len, title, avg_score)
                    elif review_type == "p":
                        print("==================================\nGenerating positive review with markov model algorithm 1\n")
                        # create_review_with_markov_chains(positive_reviews_string.split(), avg_positive_review_len, title, avg_positive_score)
                        # print("algorithm 2:")
                        create_review_with_markov_chains_2(positive_reviews_string.split(), avg_positive_review_len, title, avg_positive_score)
                    elif review_type == "n":
                        print("==================================\nGenerating negative review with markov model algorithm 1\n")
                        # create_review_with_markov_chains(negative_reviews_string.split(), avg_negative_review_len, title, avg_negative_score)
                        # print("algorithm 2:")
                        create_review_with_markov_chains_2(negative_reviews_string.split(), avg_negative_review_len, title, avg_negative_score)
                    else:
                        print(review_type, "is not a valid option for review type.")
            elif (model == "3"):
                for review_type in review_types:
                    if review_type == "a":
                        print("==================================\nGenerating average review with neural network model\n")
                        create_review_with_tensorflow(reviews, title, avg_score)
                    elif review_type == "p":
                        print("==================================\nGenerating positive review with neural network model\n")
                        create_review_with_tensorflow(positive_reviews, title, avg_positive_score)
                    elif review_type == "n":
                        print("==================================\nGenerating negative review with neural network model\n")
                        create_review_with_tensorflow(negative_reviews, title, avg_negative_score)
                    else:
                        print(review_type, "is not a valid option for review type.")
            else:
                print(model, "is not a valid option for language model.")

        print("\n")
        print("Review another movie or exit?")
        if (input() == "exit"):
            exit()  


##########################################
# Markov chain generated text source 
# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])

# Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
def create_review_with_markov_chains(corpus, review_len, title, avg_score):
    print("I rate " + title + " %.1f/10\n" % avg_score)

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
        if i >= review_len and (next_word[-1] == "." or next_word[-1] == "!" or next_word[-1] == "?"):
            break
        i += 1

    review = " ".join(chain)
    print(review + "\n")

##########################################
# Second markov chain method 
# Source: https://www.jeffcarp.com/posts/2019/markov-chain-python/
def create_review_with_markov_chains_2(corpus, review_len, title, avg_score):
    print("I rate " + title + " %.1f/10\n" % avg_score)
    # Create graph
    markov_graph = defaultdict(lambda: defaultdict(int))

    last_word = corpus[0]
    for word in corpus[1:]:
        markov_graph[last_word][word] += 1
        last_word = word
    
    print(' '.join(walk_graph(markov_graph, distance=review_len)), '\n')

# Source: https://www.jeffcarp.com/posts/2019/markov-chain-python/
def walk_graph(graph, distance=5, start_node=None):
    
    # If not given, pick a start node at random.
    if not start_node:
        start_node = random.choice(list(graph.keys()))
        while start_node.islower():
            start_node = np.random.choice(list(graph.keys()))
    
    weights = np.array(
        list(graph[start_node].values()),
        dtype=np.float64)
    # Normalize word counts to sum to 1.
    weights /= weights.sum()

    # Pick a destination using weighted distribution.
    choices = list(graph[start_node].keys())
    chosen_word = np.random.choice(choices, None, p=weights)
    
    if distance <= 0 and chosen_word[-1] == ".":
        return [chosen_word]
    
    return [chosen_word] + walk_graph(graph, distance=distance-1, start_node=chosen_word)

##########################################
# N grams generated text source 
# self made script from assn2
def create_review_with_trigrams(corpus, review_len, title, avg_score):
    print("I rate " + title + " %.1f/10\n" % avg_score)
    
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
    word_count = 0
    while True:
        randint = np.random.randint(trigram_count)
        
        while trigrams[randint][2].islower():
            randint = np.random.randint(trigram_count)

        first_word = (trigrams[randint], unsmoothed_trigrams[trigrams[randint]])

        current_word = first_word
        sentence = ""
        while current_word[0][2][-1] != "." and word_count <= (review_len * 2):
            sentence += current_word[0][2] + " "
            next_word = { "prob" : 0, "word": "" }
            next_word_candidates = []
            # Find most likely word to come after current word
            for trigram in unsmoothed_trigrams:
                # Set most likely word if this trigram has the highest probability found so far
                if trigram[1] == current_word[0][2] and trigram[0] == current_word[0][1] and unsmoothed_trigrams[trigram] > next_word["prob"]:
                    next_word["prob"] = unsmoothed_trigrams[trigram]
                    next_word_candidates.append({"prob": unsmoothed_trigrams[trigram], "word":trigram})

            next_word_candidates = sorted(next_word_candidates, key = lambda i: i["prob"]) 
            if (len(next_word_candidates) >= 3):
                next_word = next_word_candidates[random.randrange(0, 3)]
            else:
                next_word = next_word_candidates[random.randrange(0, len(next_word_candidates))]        
            current_word = (next_word["word"], next_word["prob"])

            word_count += 1
        sentence += current_word[0][2] + " "
        review += sentence
        
        if word_count > review_len:
            break
    
    print(review + "\n")

##########################################
# Neural network text generation 
# https://github.com/minimaxir/textgenrnn
def create_review_with_tensorflow(reviews, title, avg_score):
    from textgenrnn import textgenrnn
    textgen = textgenrnn()
    print("I rate " + title + " %.1f/10\n" % avg_score)
    textgen.train_on_texts(reviews, num_epochs=1)

if __name__ == "__main__": 
    main()
