"""
References:
    * http://arxiv.org/abs/1605.07146
    * https://github.com/asmith26/wide_resnets_keras/blob/master/main.py
Majority of the code comes from here:
https://github.com/asmith26/wide_resnets_keras/blob/master/main.py
"""

# Imports
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import tensorflow as tf

WEIGHT_DECAY = 1e-6
INIT = "he_normal"
DEPTH = 28
WIDTH_MULT = 2


def projection_head(x, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = layers.Dense(
            hidden_dim,
            use_bias=False,
            name=f"projection_layer_{i}",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        )(x)
        x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
        x = layers.Activation("relu")(x)
    outputs = layers.Dense(hidden_dim, use_bias=False, name="projection_output")(x)
    return outputs


def prediction_head(x, hidden_dim=128, mx=4):
    """Constructs the prediction head."""
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = layers.Dense(
        hidden_dim // mx,
        use_bias=False,
        name=f"prediction_layer_0",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(
        hidden_dim,
        use_bias=False,
        name="prediction_output",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    return x


def get_network(hidden_dim=128, use_pred=False, return_before_head=True):
    


    trunk = tf.keras.applications.ResNet50V2(include_top=True,input_shape=(224,224,3))
    last_layer= trunk.get_layer("avg_pool")
    trunk_outputs = last_layer.output

    # Projections
    projection_outputs = projection_head(trunk_outputs, hidden_dim=hidden_dim)
    if return_before_head:
        model = tf.keras.Model(trunk.input, [trunk_outputs, projection_outputs])
    else:
        model = tf.keras.Model(trunk.input, projection_outputs)

    # Predictions
    if use_pred:
        prediction_outputs = prediction_head(projection_outputs)
        if return_before_head:
            model = tf.keras.Model(trunk.input, [projection_outputs, prediction_outputs])
        else:
            model = tf.keras.Model(trunk.input, prediction_outputs)

    return model
