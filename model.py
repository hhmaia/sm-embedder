import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0

def create_backbone_model(input_shape):
    return EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            pooling='avg',
            input_shape=input_shape)


def create_head_model(input_shape, emb_dim, num_classes):
    features_input = tf.keras.Input(input_shape)
    x = tf.keras.layers.Dropout(0.2)(features_input)
    embedder = tf.keras.layers.Dense(
            emb_dim, 'relu', use_bias=False, name='embedder')(x)
    softmax = tf.keras.layers.Dense(
            num_classes, 'softmax', name='softmax')(embedder)
   
    model = tf.keras.models.Model(
            inputs=features_input, outputs=[embedder, softmax])
    return model


def load_inference_model(head_model_ckp, input_shape):
    backbone_model = create_backbone_model(input_shape)
    head_model = tf.keras.models.load_model(head_model_ckp, compile=False)
    out = head_model(backbone_model.output)

    inference_model = tf.keras.models.Model(
            inputs=backbone_model.input,
            outputs=out[0])

    return inference_model

