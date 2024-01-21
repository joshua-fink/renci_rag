from annoy import AnnoyIndex
import pandas as pd

test = AnnoyIndex(512, 'angular')

test.load('data/doc_index.ann')

bbc_outs_df = pd.read_csv('data/bbc_out.csv')

outs = test.get_nns_by_item(578, 5)

print(bbc_outs_df.loc[bbc_outs_df['id'] == 1, 'chunk'].iloc[0])
print()

for sim in outs:
    print(bbc_outs_df.loc[bbc_outs_df['id'] == sim, 'chunk'].iloc[0])

