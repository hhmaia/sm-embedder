import time
import tensorflow as tf

from extract_features import featuremaps_dataset_from_tfrecord 
from centerloss import get_center_loss
from mlutils import create_lr_sched
from model import create_head_model

# A LOT of boilerplate code

#TODO  Needs extraction? I don't think soo...
train_featuremaps_record = 'build/train_featuremaps.tfrecord'
val_featuremaps_record = 'build/val_featuremaps.tfrecord'
input_shape = (300, 300, 3)
featuremaps_dim = 1280
emb_dim = 32 
batch_size = 32 
num_classes = 20
epochs = 10 
centerloss_alphas = [1, 0.85, 0.8, 0.78, 0.7, 0.5]

# Directory logic will be explained @ README
head_ckp = 'build/checkpoints/head'
timestr = time.strftime("%Y%m%d-%H%M%S")
logs_path = 'build/logs/'# + timestr


# Mapping, batching, shuffling... bla bla bla
def prepare_features_dataset_for_training(record, batch_size, repeat=True):
    dataset = featuremaps_dataset_from_tfrecord(record)
    dataset = dataset.map(
            lambda ex: (ex['featuremap'], (ex['label'], ex['label'])))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def train_head(head_ckp=None):
    # Ugly, but need time
    try:
        model = tf.keras.models.load_model(head_ckp, compile=False)
    except:
        model = create_head_model([featuremaps_dim], emb_dim, num_classes)

    train_dataset = prepare_features_dataset_for_training(
            train_featuremaps_record, batch_size)
    val_dataset = prepare_features_dataset_for_training(
            val_featuremaps_record, batch_size, repeat=False)

    # Here, we get the parametrized center_loss function and softmax loss
    center_loss, centers = get_center_loss(
            centerloss_alphas[2], num_classes, emb_dim)
    softmax_loss = tf.keras.losses.sparse_categorical_crossentropy

    model.compile('adam',
                  [center_loss, softmax_loss],
                  {'sm': 'accuracy'})

    tb_cb = tf.keras.callbacks.TensorBoard(
            logs_path, 1, True)  

    # In the end, the model converges so quickly that we don't need a scheduler 
    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs, lr_end=1e-6, warmup=False), True)
    
    ckp_cb = tf.keras.callbacks.ModelCheckpoint(
            head_ckp,
            'sm_accuracy', 
            save_best_only=True)

    model.fit(train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=1000,
            validation_data=val_dataset,
            callbacks=[tb_cb, ckp_cb, lr_cb]) 

    return model, centers

head_model, centers = train_head(head_ckp)

# Exporting the centers. They will be useful for predictions and visualizations
with open('build/centers.tsv', 'w') as f:
    for center in centers.numpy():
        for w in center:
            f.write(str(w) + '\t')
        f.write('\n')

