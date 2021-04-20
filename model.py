import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0


def create_backbone_model(input_shape):
    # Very good efficiency/cost ratio for this model
    # I should test it without avg pooling, but time is short and results are
    # goot enough
    return EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            pooling='avg',
            input_shape=input_shape)


# Ok here is the meat of this project.
# Dropout is not really needed, but it's not hurting so it's fine.
# Using a dense layer w/o bias and relu activation as my embedder layer.
# The softmax layer is used only for training, as you'll see in the next 
# function.
# Two outputs are exposed: the embedder outputs, and the softmax output.
def create_head_model(input_shape, embedder_dim, num_classes):
    features_input = tf.keras.Input(input_shape)
    x = tf.keras.layers.Dropout(0.2)(features_input)
    embedder = tf.keras.layers.Dense(
            embedder_dim, 'relu', use_bias=False, name='embedder')(x)
    softmax = tf.keras.layers.Dense(
            num_classes, 'softmax', name='softmax')(embedder)
   
    model = tf.keras.models.Model(
            inputs=features_input, outputs=[embedder, softmax])
    return model


# For inference, we conect the two previous models:
# Backbone -> Head (the embedder)
# The only output exposed is the embedder output.
# In fact, this should be exposed to a third model, completely elliminating 
# loading the weights for the softmax layer... but time, no need for that 
# right now.
def load_inference_model(head_model_ckp, input_shape):
    backbone_model = create_backbone_model(input_shape)
    head_model = tf.keras.models.load_model(head_model_ckp, compile=False)
    out = head_model(backbone_model.output)

    inference_model = tf.keras.models.Model(
            inputs=backbone_model.input,
            outputs=out[0])

    return inference_model

