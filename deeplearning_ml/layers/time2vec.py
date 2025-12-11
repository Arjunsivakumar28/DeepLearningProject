# Code from DeRieux et al.
import tensorflow as tf
from tensorflow import keras


class Time2Vec(keras.layers.Layer):
    def __init__(self, embed_dim: int, activation: str = 'sin', **kwargs):
        """
        Input:  (batch, seq, feat)
        Output: (batch, seq, feat * embed_dim)
        """
        super(Time2Vec, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.activation = activation.lower()

        if embed_dim <= 1:
            raise ValueError('embedding dimension must be greater than 1')

        if self.activation.startswith('sin'):
            self.activation_func = tf.sin
        elif self.activation.startswith('cos'):
            self.activation_func = tf.cos
        else:
            raise ValueError(
                f'unsupported periodic activation function "{activation}"'
            )

    def build(self, input_shape: tf.TensorShape):
        # input_shape: (batch, seq, feat)
        seq_len = input_shape[1]
        feat_dim = input_shape[2]

        if feat_dim is None:
            raise ValueError(
                f"Time2Vec requires a known feature dimension; got input_shape={input_shape}"
            )

        # Linear weights: broadcast over (batch, seq, feat)
        # Shape (1, 1, feat_dim) so they broadcast cleanly
        self.w_linear = self.add_weight(
            name='w_linear',
            shape=(1, 1, feat_dim),
            initializer='uniform',
            trainable=True,
        )
        self.b_linear = self.add_weight(
            name='b_linear',
            shape=(1, 1, feat_dim),
            initializer='uniform',
            trainable=True,
        )

        # Periodic weights over feature dim
        # x: (b, t, f), w_periodic: (f, k) -> inner: (b, t, f, k)
        self.w_periodic = self.add_weight(
            name='w_periodic',
            shape=(feat_dim, self.embed_dim - 1),  # (feat, embed_dim-1)
            initializer='uniform',
            trainable=True,
        )

        # Bias for periodic part, independent of seq_len so it can broadcast:
        # (1, 1, feat * (embed_dim-1))
        self.b_periodic = self.add_weight(
            name='b_periodic',
            shape=(1, 1, feat_dim * (self.embed_dim - 1)),
            initializer='uniform',
            trainable=True,
        )

        super(Time2Vec, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch, seq, feat)

        # Linear part: (1,1,feat) broadcast over (batch,seq,feat)
        embed_linear = self.w_linear * x + self.b_linear  # (batch, seq, feat)

        # Dynamic shape for safe reshaping
        dyn_shape = tf.shape(x)
        batch_size = dyn_shape[0]
        seq_len = dyn_shape[1]
        feat_dim = dyn_shape[2]

        # Periodic part:
        # x: (b, t, f), w_periodic: (f, k) -> inner: (b, t, f, k)
        inner = tf.einsum('btf,fk->btfk', x, self.w_periodic)

        # Reshape to (b, t, f * (embed_dim-1))
        inner = tf.reshape(
            inner,
            (batch_size, seq_len, feat_dim * (self.embed_dim - 1)),
        )

        # Add bias (broadcast over batch & time)
        inner = inner + self.b_periodic  # (b, t, f*(embed_dim-1))
        embed_periodic = self.activation_func(inner)

        # Concatenate linear and periodic along feature axis
        ret = tf.concat([embed_linear, embed_periodic], axis=-1)
        # ret: (batch, seq, feat * embed_dim)
        return ret

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(
            (input_shape[0], input_shape[1], input_shape[2] * self.embed_dim)
        )

    def get_config(self) -> dict:
        config = super(Time2Vec, self).get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'activation': self.activation,
        })
        return config


# Register custom layer for model loading
keras.utils.get_custom_objects()['Time2Vec'] = Time2Vec
