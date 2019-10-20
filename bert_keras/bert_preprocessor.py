import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer
import numpy as np
from typing import List

from bert_keras.preprocessor import Preprocessor


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Source: https://github.com/google-research/bert_keras
    """

    def __init__(self, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.text_a = text_a
        self.text_b = text_b

    def __str__(self):
        return self.text_a + self.text_b


class BertPreprocessor(Preprocessor):
    """Preprocessor for BERT embedding.

    This class can be used to do all the work to create the inputs (and outputs) of a Neural Network using BERT
    as embedding. Currently only single sequence classification is supported.

    Source: https://github.com/google-research/bert_keras
    """

    def __init__(self,
                 pretrained_model_path: str,
                 **kwargs):

        super().__init__(**kwargs)

        info = hub.Module(spec=pretrained_model_path)(signature="tokenization_info", as_dict=True)

        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [
                    info["vocab_file"],
                    info["do_lower_case"]
                ]
            )

        # Create the tokenizer with the vocabulary of the pretrained model
        self._tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        basic_tokens = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        self._CLS_token = basic_tokens[0]
        self._SEP_token = basic_tokens[1]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _padding_sentence(self):
        """Return a zero length sentence to pad last batch.

        :return: Three sequences of zeros (tokens, masks, segment ids).
        """

        return [0] * self._max_seq_len, [0] * self._max_seq_len, [0] * self._max_seq_len

    def tokenize(self, text_a: str, text_b: str = None):
        """Convert sequence(s) of words into sequence(s) of tokens and also compute the masking- and segment ids.

        For further details please read BERT paper.

        :param text_a: First sequence
        :param text_b: Second sequence
        :return: The sequence of tokens, masks and segment ids.
        """

        input_ids = [0] * self._max_seq_len
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [0] * self._max_seq_len
        # The segment ids are 0 for text_a and 1 for text_b
        input_segment_ids = [0] * self._max_seq_len

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_len - 2:
                tokens_a = tokens_a[0:(self._max_seq_len - 2)]

        idx = 0
        input_ids[idx] = self._CLS_token
        idx += 1

        for element in self._tokenizer.convert_tokens_to_ids(tokens_a):
            input_ids[idx] = element
            input_mask[idx] = 1
            idx += 1

        if tokens_b:
            for element in self._tokenizer.convert_tokens_to_ids(tokens_b):
                input_ids[idx] = element
                input_mask[idx] = 1
                input_segment_ids[idx] = 1
                idx += 1

        input_ids[idx] = self._SEP_token

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        for i in range(idx + 1):
            input_mask[i] = 1

        # safety check
        assert len(input_ids) == self._max_seq_len
        assert len(input_mask) == self._max_seq_len
        assert len(input_segment_ids) == self._max_seq_len

        return input_ids, input_mask, input_segment_ids

    def fit(self, texts: List[str]) -> 'BertPreprocessor':
        """This function does nothing in case of BERT but must be implemented.

        :param texts: -
        :return: self
        """

        return self

    def transform(self, examples: List[InputExample]) -> list:
        """Transform sequences of words into sequences of tokens, masks and segment ids.

        Masks are used to separate valid and padding tokens. Here the segment ids are always one since the whole
        sequence belongs together.

        For further details please read BERT paper.

        :param texts: The sequences of texts.
        :return: The sequences of tokens, masks and segment ids.
        """

        input_ids, input_masks, segment_ids = [], [], []

        for i, example in enumerate(examples):
            input_id, input_mask, segment_id = self.tokenize(text_a=example.text_a, text_b=example.text_b)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        return [np.array(input_ids), np.array(input_masks), np.array(segment_ids)]

    def inverse_transform(self, sequences: np.ndarray):
        """Transform sequences of tokens back to sequences of words (sentences).

        :param sequences: The sequences of tokens.
        :return: The sequences of words
        """

        return self._tokenizer.convert_ids_to_tokens(sequences)
