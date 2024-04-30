import dataclasses
from dataclasses import dataclass
from time import time
import multiprocessing

import gensim
import numpy as np
from gensim.models import KeyedVectors
import csv
import os
import pandas as pd
from numpy.linalg import norm
import timeit
import Levenshtein


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

    def calculate_phrases_embeddings(self):
        if os.path.isfile('data/phrases_embeddings.npy'):
            self.all_phrases_embeddings = np.load('data/phrases_embeddings.npy')
        else:
            self.all_phrases_embeddings = np.zeros((len(self.all_phrases), self.embedding_size))
            for phrase, phrase_index in self.all_phrases_to_index.items():
                phrase_words_indexs = self.all_phrases_index_list[phrase_index]
                phrase_embedding = sum(self.all_words_embeddings[phrase_words_indexs])
                self.all_phrases_embeddings[phrase_index, :] = phrase_embedding
            np.save('data/phrases_embeddings.npy', self.all_phrases_embeddings)

    def calculate_distances(self):
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
        np.save('data/all_l2_distances.npy', self.all_l2_distances)
        np.save('data/all_cosine_distances.npy', self.all_cosine_distances)

    def calculate_l2(self, phrase1_embedding: np.array, phrase2_embedding: np.array):
        l2 = np.sqrt(np.sum((phrase1_embedding - phrase2_embedding) ** 2))
        return l2

    def calculate_cos_distance(self, phrase1_embedding: np.array, phrase2_embedding: np.array):
        cosine = np.dot(phrase1_embedding, phrase2_embedding)/(norm(phrase1_embedding)*norm(phrase2_embedding))
        return cosine


def input_data(file_path: str, phrase_manager: PhraseManager):
    with open(file_path, 'r', encoding='ISO-8859-1') as fp:
        reader = csv.reader(fp, delimiter=' ')
        _ = next(reader)
        for phrase in reader:
            if phrase not in phrase_manager.all_phrases:
                phrase_manager.all_phrases.append(phrase)
                phrase_manager.all_phrases_to_index[' '.join(phrase)] = len(phrase_manager.all_phrases) - 1
                word_indexs = phrase_manager.phrase_processor(phrase)
                phrase_manager.all_phrases_index_list.append(word_indexs)
    return phrase_manager


def data_pipeline(word_embeddings, word_to_index, phrase_data_path: str = 'data/phrases.csv'):
    phrase_manager = PhraseManager()
    phrase_manager.all_words_embeddings = word_embeddings
    phrase_manager.word_to_emb_index = word_to_index
    phrase_manager = input_data(phrase_data_path, phrase_manager)
    phrase_manager.calculate_phrases_embeddings()
    phrase_manager.calculate_distances()
    return phrase_manager


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
    pass