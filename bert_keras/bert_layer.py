import tensorflow as tf
import tensorflow_hub as hub

from enum import Enum, auto


class BertLayer(tf.keras.layers.Layer):
    """Custom keras layer integrating Bert from tf-hub.

    Source: https://towardsdatascience.com/bert_keras-in-keras-with-tensorflow-hub-76bcbc9417b
    Bert: https://arxiv.org/pdf/1810.04805.pdf

    Search for pretrained models: https://tfhub.dev/
    (Here we use the actual best as default - Whole Word Masking, uncased version)

    Output size: each input token has vector representation which has e.g. 1024 elements.
    """

    class Pooling(Enum):
        """Supported pooling types. In other words: which layer output to use.

        FIRST: use the output of the classifier node - see BERT paper.
        REDUCE_MEAN: use the output of the last encoder layer, the shape is reduced: each token has a single number
        representation, take the mean of the corresponding outputs (e.g. output_size=1024) to get this number.
        ENCODER_OUT: use the full output of the last encoder layer - use BERT as word embedding
        (word embedding size = output_size, e.g. 1024)
        """

        FIRST = auto()
        REDUCE_MEAN = auto()
        ENCODER_OUT = auto()

    def __init__(self,
                 pretrained_model_path: str,
                 output_size: int,
                 pooling: Pooling,
                 n_layers_to_finetune: int = 0,
                 **kwargs):

        self._pretrained_model_path = pretrained_model_path
        # This should be set according to the used model (H-XXXX)
        self._output_size = output_size

        self._trainable = n_layers_to_finetune != 0
        self._n_layers_to_finetune = n_layers_to_finetune

        if type(pooling) is str and BertLayer.Pooling[pooling] in BertLayer.Pooling:
            self._pooling = BertLayer.Pooling[pooling]
        elif type(pooling) is BertLayer.Pooling and pooling in BertLayer.Pooling:
            self._pooling = pooling
        else:
            raise NameError(
                "Unsupported pooling type {}! Please use one from BertLayer.Pooling.".format(pooling)
            )

        self._bert_module = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        """Load the pretrained model and select which layers to train/fine-tune."""

        self._bert_module = hub.Module(
            spec=self._pretrained_model_path, trainable=self._trainable, name="{}_module".format(self.name)
        )

        # Determine which layers to train (add them to trainable_vars)
        trainable_vars = self._bert_module.variables
        if self._pooling == BertLayer.Pooling.FIRST:
            trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]
            trainable_layers = ["pooler/dense"]

        elif self._pooling == BertLayer.Pooling.REDUCE_MEAN or self._pooling == BertLayer.Pooling.ENCODER_OUT:
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                "Unsupported pooling type {}! Please use one from BertLayer.Pooling".format(self._pooling)
            )

        # Select how many layers to fine tune
        for i in range(self._n_layers_to_finetune):
            trainable_layers.append("encoder/layer_{}".format(str(23 - i)))

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self._bert_module.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        def mul_mask(x, m):
            """Use the mask tokens to mask out the output where the input is a padding token(give 0 output)."""

            return x * tf.expand_dims(input=m, axis=-1)

        def masked_reduce_mean(x, m):
            """Use the masking method and do the above described reduce mean operation."""

            return tf.reduce_sum(mul_mask(x=x, m=m), axis=1) / (
                    tf.reduce_sum(input_tensor=m, axis=1, keepdims=True) + 1e-10)

        inputs = [tf.keras.backend.cast(x=x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )

        if self._pooling == BertLayer.Pooling.FIRST:
            pooled = self._bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self._pooling == BertLayer.Pooling.REDUCE_MEAN:
            result = self._bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            input_mask = tf.cast(x=input_mask, dtype=tf.float32)
            pooled = masked_reduce_mean(x=result, m=input_mask)

        elif self._pooling == BertLayer.Pooling.ENCODER_OUT:
            result = self._bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            input_mask = tf.cast(x=input_mask, dtype=tf.float32)
            pooled = mul_mask(x=result, m=input_mask)
        else:
            raise NameError(
                "Unsupported pooling type {}! Please use one from BertLayer.Pooling".format(self._pooling)
            )

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size

    def get_config(self):
        """This method needs to be defined to save the layer config in a way it can be reloaded correctly."""

        config = super().get_config()
        config['pretrained_model_path'] = self._pretrained_model_path
        config['output_size'] = self._output_size
        config['pooling'] = self._pooling.name
        config['n_layers_to_finetune'] = self._n_layers_to_finetune
        return config
