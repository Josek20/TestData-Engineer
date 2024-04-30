import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from main import data_pipeline, load_existing_embeddings

existing_embeddings_file = 'data/vector.npy'
existing_index_file = 'data/word_to_index.csv'
if not os.path.isfile(existing_embeddings_file):
    words_embeddings, word_to_index = load_existing_embeddings(save_to_embeddings=existing_embeddings_file)
else:
    words_embeddings = np.load(existing_embeddings_file)
    word_to_index_df = pd.read_csv(existing_index_file)
    word_to_index_df['word'] = word_to_index_df['word'].apply(lambda x: 'None' if pd.isna(x) else x)
    word_to_index = word_to_index_df.set_index('word').to_dict()['index']

phrase_manager = data_pipeline(words_embeddings, word_to_index)
