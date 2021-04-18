import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from extract_features import load_model as load_backbone, \
        feature_maps_dataset_from_tfrecord


def export_model(input_shape):
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, 'softmax', input_shape=input_shape)
    ])
    return model

dataset = feature_maps_dataset_from_tfrecord('build/features.tfrecord')
dataset = dataset.map(lambda ex: (ex['featuremap'], ex['label']))
dataset = dataset.batch(128)
dataset = dataset.prefetch(512)
dataset = dataset.repeat()

model = export_model([1280])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
model.summary()

model.fit(dataset, batch_size=128, steps_per_epoch=100, epochs=10)

val_ds = feature_maps_dataset_from_tfrecord('build/valfeats.tfrecord')
val_ds = val_ds.map(lambda ex: (ex['featuremap'], ex['label']))
val_ds = val_ds.batch(16)
model.evaluate(val_ds)
