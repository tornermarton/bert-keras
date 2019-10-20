from abc import ABC, abstractmethod
from typing import List, Union

from tensorflow.python.keras.utils import to_categorical


class Preprocessor(ABC):
    """The base class for all preprocessors used by the models defined in this package."""

    def __init__(self, max_sequence_length):
        self._max_seq_len = max_sequence_length

    @abstractmethod
    def fit(self, texts: List[str]):
        """Fit the preprocessor (e.g. tokenizer).

        :param texts: The sequences of words.
        :return: self
        """

        pass

    @abstractmethod
    def transform(self, texts: List[str]):
        """Transform the sequences of words into the format used as input at the neural network using the embedding.

        :param texts: The sequences of words.
        :return: The sequence(s) used as input at the NN.
        """

        pass

    def fit_transform(self, texts: List[str]):
        """Shortcut to call the fit and transform steps on the same set.

        :param texts: The sequences of words.
        :return: The output of the preprocessor's transform method.
        """

        return self.fit(texts=texts).transform(texts=texts)

    def fit_transform_classification(self, texts: List[str], labels: Union[List[str], List[int]]):
        """Simple shortcut to call the fit_transform method on the texts and the to_categorical method on the labels.

        :param texts: The sequences of words.
        :param labels: The corresponding labels.
        :return: The output of the fit_transform method and the one-hot encoded labels.
        """

        return self.fit_transform(texts=texts), to_categorical(y=labels)

    @abstractmethod
    def inverse_transform(self, sequences: List[str]):
        """Transform sequences of tokens back to sequences of words (sentences).

        :param sequences: The sequences of tokens.
        :return: The sequences of words
        """

        pass