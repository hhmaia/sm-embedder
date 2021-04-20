import time
import tensorflow as tf

from extract_features import featuremaps_dataset_from_tfrecord 
from centerloss import get_center_loss
from mlutils import create_lr_sched
from model import create_head_model


timestr = time.strftime("%Y%m%d-%H%M%S")
logs_dir = 'build/logs/'# + timestr


# Mapping, batching, shuffling... 
def featuremaps_dataset_for_training(record, batch_size, repeat=True):
    # load the pre calculated featuremaps
    dataset = featuremaps_dataset_from_tfrecord(record)
    # we put two labels because we have two losses
    dataset = dataset.map(
            lambda ex: (ex['featuremap'], (ex['label'], ex['label'])))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    # no need to repeat if validating
    if repeat:
        dataset = dataset.repeat()
    return dataset


# Here i'm using de default values to keep the code less verbose without
# losing flexibility if needed.
# Preety common defaults for batch size, steps, etc.
def train_head(
        head_checkpoint='build/checkpoints/head',
        train_record='build/train_featuremaps.tfrecord',
        val_record='build/val_featuremaps.tfrecord',
        num_classes=20,
        featuremaps_dim=1280,
        embedder_dim=32,
        centerloss_alpha=0.8,
        logs_dir='build/logs/',
        batch_size=32,
        epochs=10,
        steps=1000):

    # Load the model.
    try:
        model = tf.keras.models.load_model(head_checkpoint, compile=False)
        print(':: Head model checkpoint could not be loaded.')
    except:
        model = create_head_model([featuremaps_dim],
                                  embedder_dim,
                                  num_classes)
        print(':: Head model loaded successfully')

    train_dataset = featuremaps_dataset_for_training(train_record, batch_size)
    val_dataset = featuremaps_dataset_for_training(val_record, batch_size,False)

    # Here, we get the parametrized center_loss function and softmax loss
    # centerloss_alpha was experimented for [1,0.5] interval.
    # need to be careful with exploding/vanishing gradients here.
    # Its a pretty common issue when using centerloss.
    center_loss, centers = get_center_loss(
            centerloss_alpha, num_classes, embedder_dim)
    softmax_loss = tf.keras.losses.sparse_categorical_crossentropy

    model.compile(optimizer='adam',
                  loss=[center_loss, softmax_loss],
                  metrics={'softmax': 'accuracy'})

    tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir, histogram_freq=1, write_graph=True)  

    # Not really needed in this case... But it's nor hurting either.
    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs, lr_end=1e-6, warmup=False), True)
    
    ckp_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=head_checkpoint,
            monitor='softmax_accuracy', 
            save_best_only=True)

    model.fit(train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=1000,
            validation_data=val_dataset,
            callbacks=[tb_cb, ckp_cb, lr_cb]) 

    return model, centers

# For exporting the centers after training.
# They will be useful for predictions and visualizations
def export_centers(centers, path='build/centers.tsv'):
    with open(path, 'w') as f:
        for center in centers.numpy():
            for w in center:
                f.write(str(w) + '\t')
            f.write('\n')

if __name__ == '__main__':
    head_model, centers = train_head()
    export_centers(centers)

