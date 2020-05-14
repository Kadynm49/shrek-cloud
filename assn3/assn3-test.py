from math import exp
import json
import os

def classify(folder, vocab, nonspam_word_probabilites, spam_word_probabilities): 
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    true_negatives = 0

    for filename in os.listdir('data/' + folder + '-test'):
        spam_probability = 0
        nonspam_probability = 0

        for line in open('data/'+ folder +'-test/' + filename):
            for token in line.split():
                if token in vocab:
                    spam_probability += spam_word_probabilities[token]
                    nonspam_probability += nonspam_word_probabilites[token]

        if (spam_probability > nonspam_probability):
            if (folder == 'spam'):
                true_positives += 1
            else: 
                false_positives += 1
        else:
            if (folder == 'nonspam'):
                true_negatives += 1
            else: 
                false_negatives += 1
    
    return false_positives, true_positives, false_negatives, true_negatives

if __name__ == '__main__':
    nonspam_word_probabilites = json.loads(open('nonspam-word-probabilities.json').read())
    spam_word_probabilities = json.loads(open('spam-word-probabilities.json').read())
    vocab = json.loads(open('vocab.json').read())

    nonspam_false_positives, nonspam_true_positives, nonspam_false_negatives, nonspam_true_negatives = classify('nonspam', vocab, nonspam_word_probabilites, spam_word_probabilities)
    spam_false_positives, spam_true_positives, spam_false_negatives, spam_true_negatives = classify('spam', vocab, nonspam_word_probabilites, spam_word_probabilities)

    false_positives = nonspam_false_positives + spam_false_positives
    true_positives = nonspam_true_positives + spam_true_positives
    false_negatives = nonspam_false_negatives + spam_false_negatives
    true_negatives = nonspam_true_negatives + spam_true_negatives

    print('\n        |  correct  |  not correct')
    print('   spam |   ', true_positives, '   |     ', false_positives)
    print('nonspam |   ', true_negatives, '   |     ', false_negatives)
    print('\nPrecision: ' + str(true_positives/(true_positives + false_positives)))
    print('Recall: ' + str(true_positives/(true_positives + false_negatives)))