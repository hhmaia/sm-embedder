import tensorflow as tf

from extract_features import feature_maps_dataset_from_tfrecord 
from centerloss import get_center_loss


input_shape = (300, 300, 3)
emb_dim = 20 
features_dim = 1280
features_dataset = 'build/features.tfrecord'
batch_size = 128
num_classes = 20






def _load_head(input_shape, output_shape):
    sequential = tf.keras.models.Sequential([
            tf.keras.layers.Dense(emb_dim, input_shape=input_shape),
    ])
    features = sequential.layers[0].input
    embeddings = sequential.layers[0].output
    labels = sequential.layers[-1].output
    model = tf.keras.models.Model(inputs=features, outputs=[labels, embeddings])
    return model


def prepare_features_dataset_for_training(featuremaps_tfrecord, batch_size):
    dataset = feature_maps_dataset_from_tfrecord(featuremaps_tfrecord)
    dataset = dataset.map(lambda ex: (ex['featuremap'], (ex['label'], ex['label'])))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


def train_model():
    dataset = prepare_features_dataset_for_training(features_dataset, batch_size)
    model = load_head([features_dim], emb_dim)
    center_loss = get_center_loss(1, num_classes, emb_dim)
    model.compile('adam', ['sparse_categorical_crossentropy', center_loss], ['accuracy']) 

    model.fit(dataset, batch_size=batch_size, epochs=100, steps_per_epoch=1000) 

