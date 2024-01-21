import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from annoy import AnnoyIndex
import numpy as np
import torch

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
model = BertModel.from_pretrained("dslim/bert-base-NER")

vector_length = 25
doc_index = AnnoyIndex(vector_length, 'dot')

df = pd.read_csv("data/young_us_women_alive.csv")
df_out = pd.DataFrame(columns=['id', 'chunk'])

name_slugs = []
for _, row in df.iterrows():
	name_slug = row['slug']
	name_slugs.append(name_slug)

failed = []
id = 0

def get_article_text(name_slug):

	# fetch content
	response = requests.get(
		url="https://en.wikipedia.org/wiki/" + name_slug,
	)

	# see if successful
	if response.status_code != 200:
		failed.append(name_slug)
		return None
	
	# initialize the soup
	soup = BeautifulSoup(response.content, 'html.parser')

	# for progress tracking purposes
	title = soup.find(id="firstHeading").string

	# extract text from article
	article_body = soup.find(class_="mw-content-text")
	article_elements = soup.find_all('p')
	content = ""
	for piece in article_elements:
		content += re.sub('\[(.*?)\]', '', piece.get_text()).strip('\n')

	return title, content

def parse_content(title, content):
	global id

	# Occurs when upstream fault
	if content == None: return

	# progress indicator
	print(title)
	
	# split into sentences
	data = content.split('. ')
	
	# initialize starting params
	num_wordpieces = 0
	max_wordpieces = vector_length
	curr = torch.zeros(0)
	big_chunk = ""

	for i in range(len(data)):
		# tokenize data
		fact = title + ": " + str(data[i]) + '.'
		data[i] = fact
		chunk = data[i] 
		data[i] = tokenizer(data[i], return_tensors='pt')
		data[i]['input_ids'] = torch.squeeze(data[i]['input_ids'])
		sentence_len = len(data[i]['input_ids'])

		# add to previous if less than max_word_pieces
		if num_wordpieces + sentence_len <= max_wordpieces:
			curr = torch.cat((curr, data[i]['input_ids']))
			num_wordpieces += sentence_len
			big_chunk += (" " + chunk)
		# rare: super long sentences!
		elif sentence_len > max_wordpieces:

			# index previous chunk	
			if num_wordpieces > 0:
				curr = torch.cat((curr,torch.zeros(max_wordpieces - num_wordpieces)))
				doc_index.add_item(id, curr.tolist())	
				df_out.loc[len(df_out.index)] = [id, big_chunk]
				id += 1

			# index long sentence chunks
			long_sentence = data[i]['input_ids'].tolist() 
			while len(long_sentence) > max_wordpieces:
				doc_index.add_item(id, long_sentence[:max_wordpieces])
				df_out.loc[len(df_out.index)] = [id, chunk]
				id += 1
				
				long_sentence = long_sentence[max_wordpieces:]
			
			# index last chunk
			long_sentence += (max_wordpieces - len(long_sentence)) * [0]
			doc_index.add_item(id, long_sentence)
			df_out.loc[len(df_out.index)] = [id, chunk]
			id += 1

			# reset back to 0 for next iteration
			num_wordpieces = 0
			curr = torch.zeros(0)
			big_chunk = ""

		else:
			# index previous chunk
			curr = torch.cat((curr,torch.zeros(max_wordpieces - num_wordpieces)))
			doc_index.add_item(id, curr.tolist())	
			df_out.loc[len(df_out.index)] = [id, big_chunk]
			id += 1

			# start next chunk
			curr = data[i]['input_ids']
			num_wordpieces = sentence_len
			big_chunk = chunk
	
	# last chunk catch all
	if 0 < num_wordpieces < max_wordpieces:
		curr = torch.cat((curr,torch.zeros(max_wordpieces - num_wordpieces)))
		doc_index.add_item(id, curr.tolist())	
		df_out.loc[len(df_out.index)] = [id, big_chunk]
		id += 1

# kinda dumb to have this here but too tired to fix
for slug in name_slugs:
	title, content = get_article_text(slug)
	parse_content(title, content)

# save tree indexer
doc_index.build(5)
doc_index.save('data/wiki_index.ann')
df_out.to_csv('data/women_out.csv', index=False)

# clean up from dataset if not available on Wikipedia
for fail in failed:
    df = df.loc[df['slug'] != fail]

df.to_csv("data/young_us_women_alive.csv", index=False)