from annoy import AnnoyIndex # faiss not supported by pip :(
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import string

CHUNK_SIZE = 512

def remove_extra_spaces(text):
    return ' '.join(text.split())

def space_apart_punctuation(text):
    return ''.join([' ' + char + ' ' if char in string.punctuation else char for char in text])


bert_tiny = "prajjwal1/bert-small" # (L = 2, H = 128)
tokenizer = AutoTokenizer.from_pretrained(bert_tiny)
model = AutoModel.from_pretrained(bert_tiny)

bert_size = 512

doc_index = AnnoyIndex(bert_size, 'angular') # similarity score is dot product

bbc_df = pd.read_csv('data/bbc_news_list_uk.csv')

bbc_df_out = df = pd.DataFrame(columns=['id', 'chunk'])

id = 0
for i, row in bbc_df.iterrows():
            content = str(row["content"])

            space_apart = space_apart_punctuation(content)
            result = remove_extra_spaces(space_apart)
            split_data = result.split()

            chunk = ' '.join(split_data)
            chunk_vec = tokenizer(chunk)
            chunk_len = len(chunk_vec['input_ids'])

            if chunk_len < bert_size:
                zeroes = (bert_size - chunk_len) * [0]
                chunk_vec['input_ids'] += zeroes
            
            doc_index.add_item(id, chunk_vec['input_ids'][:bert_size])
            bbc_df_out.loc[len(bbc_df_out.index)] = [id, chunk]  
            id += 1

            '''
            rounds = len(split_data) // CHUNK_SIZE + 1

            for i in range(rounds):
                chunk = ' '.join(split_data[CHUNK_SIZE*i:CHUNK_SIZE*i+CHUNK_SIZE])
                chunk_vec = tokenizer(chunk)
                chunk_len = len(chunk_vec['input_ids'])

                zeroes = (bert_size - chunk_len) * [0]
                chunk_vec['input_ids'] += zeroes
                
                doc_index.add_item(id, chunk_vec['input_ids'][:bert_size])
                bbc_df_out.loc[len(bbc_df_out.index)] = [id, chunk]  
                id += 1
            '''
    #except:
        #print("Welp")
    

doc_index.build(1)
doc_index.save('data/wiki_index.ann')
bbc_df_out.to_csv('data/women_out.csv', index=False)



