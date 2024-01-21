from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)

print(dataset)

import random
random_choice = random.choice(dataset['train'])
#print(f'{random_choice["text"]}')

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize


text = []
for data in dataset['train']:
    print(data['text'])
    break
    #text.extend(sent_tokenize(data['text']))

#print(text)