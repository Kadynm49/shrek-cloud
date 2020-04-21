import re
import unicodedata
from contractions import CONTRACTION_MAP
import nltk
from nltk.corpus import stopwords

# final_data taken from:  
# https://shrek.fandom.com/wiki/Shrek_(film)/Transcript

# get data for characters
def get_char_lines(char):    
    output = []          
    print('Getting lines for', char)    
    stopword_list = stopwords.words('english')    

    with open('final_data.txt', 'r') as f:
        for line in f:
            if re.findall(r'(^'+char+r'.*:.*)',line,re.IGNORECASE):
                # Clean line: remove symbols, lowercasing, expanding contractions, etc
                line = re.sub(r'.*:', '', line)
                line = re.sub('[\(\[].*?[\)\]]', ' ', line)
                line = re.sub('\\n', '', line)
                line = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                line = expand_contractions(line)
                line = line.lower()
                line = re.sub('[^a-zA-Z0-9\s]', '', line)

                # Remove stop words from line
                tokens = nltk.word_tokenize(line)    
                tokens = [token.strip() for token in tokens]    
                line = ' '.join([token for token in tokens if token not in stopword_list])

                output.append(line)
    f.close()
    print(char, 'has ', len(output), 'lines')
    return output

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                        flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

print(get_char_lines('fiona')[100])