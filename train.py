import tensorflow as tf

from extract_features import feature_maps_dataset_from_tfrecord 
from centerloss import get_center_loss
from mlutils import create_lr_sched

input_shape = (300, 300, 3)
emb_dim = 10 
features_dim = 1280
train_featuremaps_record = 'build/train_features.tfrecord'
val_featuremaps_record = 'build/train_features.tfrecord'
batch_size = 128
num_classes = 20
epochs = 5 

def load_head(input_shape, output_shape, num_classes):
    features_input = tf.keras.Input(input_shape)
    embeddings = tf.keras.layers.Dense(
            emb_dim, 'relu', use_bias=False, name='emb')(features_input)
    softmax = tf.keras.layers.Dense(
            num_classes, 'softmax', name='sm')(embeddings)
   
    model = tf.keras.models.Model(
            inputs=features_input, outputs=[embeddings, softmax])
    return model


def prepare_features_dataset_for_training(featuremaps_tfrecord, batch_size, repeat=True):
    dataset = feature_maps_dataset_from_tfrecord(featuremaps_tfrecord)
    dataset = dataset.map(
            lambda ex: (ex['featuremap'], (ex['label'], ex['label'])))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def train_model():
    train_dataset = prepare_features_dataset_for_training(
            train_featuremaps_record, batch_size)
    val_dataset = prepare_features_dataset_for_training(
            val_featuremaps_record, batch_size)

    model = load_head([features_dim], emb_dim, num_classes)
    center_loss, centers = get_center_loss(0.8, num_classes, emb_dim)
    softmax_loss = tf.keras.losses.sparse_categorical_crossentropy

    model.compile('adam',
                  [center_loss, softmax_loss],
                  {'sm': 'accuracy', 'emb': 'mse'})

    tb_cb = tf.keras.callbacks.TensorBoard(
            'build/logs', 1, False)  

    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs, warmup=True), True)

    model.fit(train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=1000,
            callbacks=[tb_cb]) 

    return model, centers

model, centers = train_model()


with open('centers.tsv', 'w') as f:
    for center in centers.numpy():
        for w in center:
            f.write(str(w) + '\t')
        f.write('\n')
