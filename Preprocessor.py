# Typing
from multiprocessing.sharedctypes import Value
from typing import Tuple

# Preprocessing
from string import punctuation
from collections import Counter
import os

# ML
import numpy as np

# constants


class Preprocessor:
    def load_data(data_file_path: str) -> list[str]:
        '''load data, return (reviews, labels)'''
        try:
            with open(data_file_path, 'r') as f:
                data = f.readlines()

        except IOError as e:
            print(f'ERROR: Preprocessor.load_data: {e}')
            return None

        return data

    def strip_punctuation(data_set: list[str]) -> Tuple[list[str], list[str]]:
        '''strip punctuation, return list of words and list of data'''
        all_data = list()

        for text in data_set:
            text = "".join([c for c in text if c not in punctuation])
            all_data.append(text)

        all_text = " ".join(all_data)
        all_words = all_text.split()

        return all_words, all_data

    def sorted_count(words: list[str]) -> dict[str, int]:
        '''get count of words, return dict of word:count'''
        count_words = Counter(words)
        total_words = len(words)
        sorted_count = count_words.most_common(total_words)

        return sorted_count

    def get_vocab(words: dict[str, int]) -> dict[str, int]:
        '''map words to int based on number of times they appear in words'''
        return {w: i+1 for i, (w, c) in enumerate(words)}

    def encode_data(data_set: list[str], vocab: dict[str, int]) -> list[int]:
        encoded_data = list()
        for entry in data_set:
            encoded_entry = list()
            for word in entry.split():
                if word not in vocab.keys():

                    # if word is not available in vocab_to_int put 0 in that place
                    encoded_entry.append(0)
                else:
                    encoded_entry.append(vocab[word])
            encoded_data.append(encoded_entry)

        return encoded_data

    def normalize_labels(labels: list[str], positive_str: str, negative_str: str) -> list[int]:
        '''Normalize labels, positive_str to 1, negative_str to 0, other to -1'''
        normalized_labels = list()
        for label in labels:
            stripped_label = label.strip().lower()
            if stripped_label == positive_str.strip().lower():
                normalized_labels.append(1)
            elif stripped_label == negative_str.strip().lower():
                normalized_labels.append(0)
            else:
                normalized_labels.append(-1)

        return normalized_labels

    def to_sentiment(rating: str) -> int:
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2
