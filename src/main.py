import numpy as np
from gensim.models import KeyedVectors
import csv
import os
import pandas as pd
from numpy.linalg import norm
import Levenshtein

from utils import log_execution_time, handle_exceptions


def find_closest_key(target_string, dictionary):
    closest_key = None
    min_distance = float('inf')  # Initialize with a large value

    for key in dictionary.keys():
        distance = Levenshtein.distance(target_string, key)
        if distance < min_distance:
            min_distance = distance
            closest_key = key

    return dictionary[closest_key]


class PhraseManager:
    def __init__(self):
        self.all_phrases_embeddings = None
        self.embedding_size = 300
        self.all_words_embeddings = np.array([])
        self.word_to_emb_index = {}
        self.all_phrases_to_index = {}
        self.all_phrases_index_list = []
        self.all_phrases = []
        self.all_cosine_distances = None
        self.all_l2_distances = None

    def phrase_processor(self, phrase: list):
        word_index_list = []
        for word in phrase:
            word_index = self.word_to_emb_index.get(word)
            if word_index is None:
                word_index = find_closest_key(word, self.word_to_emb_index)
            word_index_list.append(word_index)
        return word_index_list

    def calculate_phrases_embeddings(self, phrases_embedding_path: str = 'data/phrases_embeddings.npy'):
        if os.path.isfile(phrases_embedding_path):
            self.all_phrases_embeddings = np.load(phrases_embedding_path)
        else:
            self.all_phrases_embeddings = np.zeros((len(self.all_phrases), self.embedding_size))
            for phrase, phrase_index in self.all_phrases_to_index.items():
                phrase_words_indexs = self.all_phrases_index_list[phrase_index]
                phrase_embedding = sum(self.all_words_embeddings[phrase_words_indexs])
                self.all_phrases_embeddings[phrase_index, :] = phrase_embedding
            np.save(phrases_embedding_path, self.all_phrases_embeddings)

    def calculate_distances(self, l2_distances_path: str = 'data/all_l2_distances.npy',
                            cosine_distances_path: str = 'data/all_cosine_distances.npy'):
        n = len(self.all_phrases)
        self.all_l2_distances = np.zeros((n, n))
        self.all_cosine_distances = np.zeros((n, n))
        for index1, phrase1 in enumerate(self.all_phrases):
            for index2, phrase2 in enumerate(self.all_phrases[index1 + 1:]):
                phrase1_embedding = self.all_phrases_embeddings[index1]
                phrase2_embedding = self.all_phrases_embeddings[index2]
                l2_distance = self.calculate_l2(phrase1_embedding, phrase2_embedding)
                self.all_l2_distances[index1, index2] = l2_distance
                cosine_distance = self.calculate_cos_distance(phrase1_embedding, phrase2_embedding)
                self.all_cosine_distances[index1, index2] = cosine_distance
        np.save(l2_distances_path, self.all_l2_distances)
        np.save(cosine_distances_path, self.all_cosine_distances)

    def calculate_l2(self, phrase1_embedding: np.array, phrase2_embedding: np.array):
        l2 = np.sqrt(np.sum((phrase1_embedding - phrase2_embedding) ** 2))
        return l2

    def calculate_cos_distance(self, phrase1_embedding: np.array, phrase2_embedding: np.array):
        cosine = np.dot(phrase1_embedding, phrase2_embedding)/(norm(phrase1_embedding)*norm(phrase2_embedding))
        return cosine


@handle_exceptions
@log_execution_time
def input_data(file_path: str, phrase_manager: PhraseManager):
    with open(file_path, 'r', encoding='ISO-8859-1') as fp:
        reader = csv.reader(fp, delimiter='\n')
        _ = next(reader)
        for phrase in reader:
            if phrase[0] not in phrase_manager.all_phrases:
                phrase_manager.all_phrases.append(phrase[0])
                phrase_manager.all_phrases_to_index[phrase[0]] = len(phrase_manager.all_phrases) - 1
                word_indexs = phrase_manager.phrase_processor(phrase[0].split(' '))
                phrase_manager.all_phrases_index_list.append(word_indexs)
    return phrase_manager


def data_pipeline(word_embeddings, word_to_index, phrase_data_path: str = '../data/phrases.csv'):
    phrase_manager = PhraseManager()
    phrase_manager.all_words_embeddings = word_embeddings
    phrase_manager.word_to_emb_index = word_to_index
    phrase_manager = input_data(phrase_data_path, phrase_manager)
    phrase_manager.calculate_phrases_embeddings()
    phrase_manager.calculate_distances()
    return phrase_manager


@handle_exceptions
@log_execution_time
def pipeline_for_new_input(phrase: str, phrase_manager: PhraseManager):
    word_index_list = phrase_manager.phrase_processor(phrase.split(' '))
    phrase_embedding = sum(phrase_manager.all_words_embeddings[word_index_list])

    best_phrase = None
    best_cos_distance = float('inf')
    best_l2_distance = float('inf')

    for index1, phrase1 in enumerate(phrase_manager.all_phrases):
        phrase1_embedding = phrase_manager.all_phrases_embeddings[index1]
        l2_distance = phrase_manager.calculate_l2(phrase1_embedding, phrase_embedding)
        cos_distance = phrase_manager.calculate_cos_distance(phrase1_embedding, phrase_embedding)
        if cos_distance < best_cos_distance or (cos_distance == best_cos_distance and l2_distance < best_l2_distance):
            best_phrase = phrase1
            best_cos_distance = cos_distance
            best_l2_distance = l2_distance
    return best_phrase, best_l2_distance, best_cos_distance


@handle_exceptions
@log_execution_time
def load_existing_embeddings(file_path: str = 'data/GoogleNews-vectors-negative300.bin.gz',
                             save_to_embeddings: str = 'data/vector.npy',save_to_index: str = 'data/word_to_index.csv', limit: int = 1_000_000):
    wv = KeyedVectors.load_word2vec_format(file_path, binary=True, limit=limit)
    words_embeddings = wv.vectors
    np.save(save_to_embeddings, words_embeddings)
    word_to_index = wv.key_to_index
    word_to_index_df = pd.DataFrame(list(wv.key_to_index.items()), columns=['word', 'index'])
    word_to_index_df.to_csv(save_to_index, index=False)
    return words_embeddings, word_to_index


if __name__ == '__main__':
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