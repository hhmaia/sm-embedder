import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0


def create_backbone_model(input_shape):
    return EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            pooling='avg',
            input_shape=input_shape)


def create_head_model(input_shape, output_shape, num_classes):
    features_input = tf.keras.Input(input_shape)
    embeddings = tf.keras.layers.Dense(
            emb_dim, 'relu', use_bias=False, name='emb')(features_input)
    softmax = tf.keras.layers.Dense(
            num_classes, 'softmax', name='sm')(embeddings)
   
    model = tf.keras.models.Model(
            inputs=features_input, outputs=[embeddings, softmax])
    return model


def load_inference_model(head_model_ckp, input_shape):
    backbone_model = create_backbone_model(input_shape)
    head_model = tf.keras.models.load_model(head_model_ckp)

    head_model.inputs = backbone_model.outputs

    inference_model = tf.keras.models.Model(
            inputs=backbone_model.inputs,
            outputs=[head_model.outputs[0]])

    return inference_model
