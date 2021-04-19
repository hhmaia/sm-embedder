import time
import tensorflow as tf

from extract_features import featuremaps_dataset_from_tfrecord 
from centerloss import get_center_loss
from mlutils import create_lr_sched
from model import create_head_model

input_shape = (300, 300, 3)
emb_dim = 10 
features_dim = 1280
train_featuremaps_record = 'build/train_featuremaps.tfrecord'
val_featuremaps_record = 'build/val_featuremaps.tfrecord'
batch_size = 128
num_classes = 20
epochs = 5 

head_model_ckp_path = 'build/models/head'
timestr = time.strftime("%Y%m%d-%H%M%S")
logs_path = 'build/logs/' + timestr

def prepare_features_dataset_for_training(featuremaps_tfrecord, batch_size, repeat=True):
    dataset = featuremaps_dataset_from_tfrecord(featuremaps_tfrecord)
    dataset = dataset.map(
            lambda ex: (ex['featuremap'], (ex['label'], ex['label'])))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def train_head(head_model_checkpoint=None):
    train_dataset = prepare_features_dataset_for_training(
            train_featuremaps_record, batch_size)
    val_dataset = prepare_features_dataset_for_training(
            val_featuremaps_record, batch_size, repeat=False)

    if head_model_checkpoint:
        model = tf.keras.models.load_model(head_model_checkpoint)
    else:
        model = create_head_model([features_dim], emb_dim, num_classes)

    center_loss, centers = get_center_loss(0.8, num_classes, emb_dim)
    softmax_loss = tf.keras.losses.sparse_categorical_crossentropy

    model.compile('adam',
                  [center_loss, softmax_loss],
                  {'sm': 'accuracy', 'emb': 'mse'})

    tb_cb = tf.keras.callbacks.TensorBoard(
            logs_path, 1, True)  

    # In the end, the model converges so quickly that we don't need a variable
    # learning rate
    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs, warmup=True), True)

    ckp_cb = tf.keras.callbacks.ModelCheckpoint(
            head_model_ckp_path,
            'sm_accuracy', 
            save_best_only=True)

    model.fit(train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=1000,
            validation_data=val_dataset,
            callbacks=[tb_cb, ckp_cb]) 

    return model, centers

head_model, centers = train_head()

with open('build/centers.tsv', 'w') as f:
    for center in centers.numpy():
        for w in center:
            f.write(str(w) + '\t')
        f.write('\n')
