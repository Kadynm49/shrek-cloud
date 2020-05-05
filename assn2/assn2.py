import pandas as pd
import string
import time
import numpy as np
import re
from IPython.display import clear_output

start = "<s> "
end = "</s>\n"
i = 0
train_data = pd.DataFrame()

start_time = time.time()
with open("data/test.txt", "r") as file:
    for line in file:
        i = i + 1
        train_data = train_data.append(pd.DataFrame({'sentence_clean':start + line[:-1] + end}, index = [i]))

print(train_data.head().to_string())

text = train_data['sentence_clean']
text_list = " ".join(map(str, text))
text_list[0:100]

word_list = pd.DataFrame({'words':text.str.split(' ', expand = True).stack().unique()})   
   
word_count_table = pd.DataFrame()
for n,word in enumerate(word_list['words']):
    # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
    # e.g. 'ear' would be counted in each appearance of the word 'year'
    word_count = len(re.findall(' ' + word + ' ', text_list))  
    word_count_table = word_count_table.append(pd.DataFrame({'count':word_count}, index=[n]))
    
    clear_output(wait=False)
    print('Proportion of words completed:', np.round(n/len(word_list),4)*100,'%')

word_list['count'] = word_count_table['count']
# Remove the count for the start and end of sentence notation so 
# that these do not inflate the other probabilities
word_list['count'] = np.where(word_list['words'] == '<s' , 0,
                     np.where(word_list['words'] == '/s>', 0,
                     word_list['count']))


word_list['prob'] = word_list['count']/sum(word_list['count'])
print(word_list.head().to_string())




end_time = time.time()  
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')
