import sys
from math import log
from collections import Counter

train = sys.argv[1]
test = sys.argv[2]
tag_counts = {}
word_tag_counts = {}
word_tag_probs = {'###' : 1}
most_common_word_tags = {'###' : '###'}
word_list = []
tag_list = []
most_common_tag = ''

# Read train data
for line in open('data/' + train[0:2] + '/' + train + '.txt'):
    split_line = line.strip().split('/')
    tag_list.append(split_line[1])
    if split_line[0] != '###':
        if split_line[1] in tag_counts:
            tag_counts[split_line[1]] += 1
        else: 
            tag_counts[split_line[1]] = 1

        if (split_line[0], split_line[1]) in word_tag_counts:
            word_tag_counts[(split_line[0], split_line[1])] += 1
        else:
            word_tag_counts[(split_line[0], split_line[1])] = 1

        if split_line[0] not in word_list:
            word_list.append(split_line[0])

most_common_tag = Counter(tag_list).most_common()[0][0]

# Get word tag pair probabilities
for word_tag in word_tag_counts:
    word_tag_probs[word_tag] = log(word_tag_counts[word_tag] / tag_counts[word_tag[1]])

# Create pairs of most common word tag pairs
for word in word_list:
    most_common = 0
    prob = 0
    for word_tag in word_tag_probs:
        if word_tag[0] == word:
            if most_common == 0:
                most_common = word_tag[1]
                prob = word_tag_probs[word_tag]
            elif word_tag_probs[word_tag] > prob:
                most_common = word_tag[1]
                prob = word_tag_probs[word_tag]
    most_common_word_tags[word] = most_common

correct = 0
line_count = 0
novel = 0
known = 0
novel_correct = 0
known_correct = 0

# Test using emission probabilities and transition probabilities
for line in open('data/' + test[0:2] + '/' + test + '.txt'):
    line_split = line.strip().split('/')
    line_count += 1
    if line_split[0] in most_common_word_tags:
        known += 1
        if most_common_word_tags[line_split[0]] == line_split[1]:
            correct += 1
            known_correct += 1
    else:
        novel += 1
        if line_split[1] == most_common_tag:
            correct += 1
            novel_correct += 1


print('Tagging accuracy (Viterbi decoding):', str((correct/line_count)*100) + '%', '(known:', str((known_correct/known)*100) + '%', 'novel:', str((novel_correct/novel)*100) + '%)')