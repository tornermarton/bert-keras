import tensorflow as tf
from typing import List
import numpy as np

from bert_keras import BertPreprocessor, BertLayer, InputExample


class BertEncoder(object):
    """This class can be used to calculate BERT embeddings for texts."""

    def __init__(self,
                 pretrained_model_path: str="https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1",
                 embedding_size: int = 1024,
                 max_sequence_length: int = 128
                 ):
        self._max_seq_len = max_sequence_length
        self._output_size = embedding_size
        self._pretrained_model_path = pretrained_model_path

        self._preprocessor = BertPreprocessor(
            pretrained_model_path=pretrained_model_path,
            max_sequence_length=max_sequence_length
        )

        self._model = self._construct_model()

    def _construct_model(self) -> tf.keras.Model:
        """Construct the model architecture using BERT as embedding.

        :return: Keras model with loaded weigths (output: BERT embedding)
        """

        in_id = tf.keras.layers.Input(shape=(self._max_seq_len,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self._max_seq_len,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self._max_seq_len,), name="segment_ids")

        inputs = [in_id, in_mask, in_segment]

        # Instantiate the custom Bert Layer
        outputs = BertLayer(
            pretrained_model_path=self._pretrained_model_path,
            output_size=self._output_size,
            n_layers_to_finetune=0,
            pooling=BertLayer.Pooling.ENCODER_OUT
        )(inputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def calculate_embeddings(self, examles: List[InputExample]) -> np.ndarray:
        bert_inputs = self._preprocessor.transform(examples=examles)

        return self._model.predict(x=bert_inputs)