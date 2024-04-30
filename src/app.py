import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from main import pipeline_for_new_input, data_pipeline, load_existing_embeddings

app = Flask(__name__)

existing_embeddings_file = 'data/vector.npy'
existing_index_file = 'data/word_to_index.csv'
if not os.path.isfile(existing_embeddings_file):
    words_embeddings, word_to_index = load_existing_embeddings(save_to_embeddings=existing_embeddings_file)
else:
    words_embeddings = np.load(existing_embeddings_file)
    word_to_index_df = pd.read_csv(existing_index_file)
    word_to_index_df['word'] = word_to_index_df['word'].apply(lambda x: 'None' if pd.isna(x) else x)
    word_to_index = word_to_index_df.set_index('word').to_dict()['index']

phrase_manager = data_pipeline(words_embeddings, word_to_index, phrase_data_path='data/phrases.csv')


@app.route('/',  methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # Check if the request method is POST
        input_phrase = request.form.get('phrase')  # Use request.form to get form data
        if input_phrase:
            closest_phrase, l2_distance, cos_distance = pipeline_for_new_input(input_phrase, phrase_manager)
            return render_template('index.html', response=[closest_phrase, l2_distance, cos_distance])
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

