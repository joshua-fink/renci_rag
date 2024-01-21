from annoy import AnnoyIndex
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

vector_length = 25
test = AnnoyIndex(vector_length, 'dot')

test.load('data/wiki_index.ann')

women_df = pd.read_csv('data/women_out.csv')
questions_df = pd.read_csv('data/questions.csv')

tokenizer = BertTokenizer.from_pretrained('dslim/bert-base-NER')

for _, row in questions_df.iterrows():
    q = row['question']

    q_t = torch.squeeze(tokenizer(q, return_tensors='pt')['input_ids'])
    q_t = torch.cat((q_t, torch.zeros(vector_length - len(q_t)))).tolist()

    outs = test.get_nns_by_vector(q_t, 5)
    print(q)
    print()
    for sim in outs:
        print(women_df.loc[women_df['id'] == sim, 'chunk'].iloc[0])

    print()
    print()
    print()
