import io
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.plugins import projector


def create_lr_sched(start_epoch, n_epochs, lr0=1e-3, lr_end=1e-9, warmup=True):
    """
    start_epoch: epoch where to start decaying
    n_epochs: total number of epochs
    lr0: initial learning rate
    lr_end: learning rate end value
    warmup: wheter to gradualy increase learning rate on the first few epochs

    return: learning rate scheduler function with given parameters.
    """

    def sched(epoch):
        exp_range = np.log10(lr0/lr_end) 
        epoch_ratio = (epoch - start_epoch)/(n_epochs - start_epoch)
        warmup_epochs = int(np.log10(lr0/lr_end)) 

        if warmup and epoch < warmup_epochs:
            lr = lr_end * (10 ** (epoch))
        elif epoch < start_epoch:
            lr = lr0
        else:
            lr = lr0 * 10**-(exp_range * epoch_ratio) 
        return lr

    return sched


def export_embeddings(embeddings, path):
    with open(path, 'w') as f: 
        text = '\n'.join(
               '\t'.join(str(v) for v in e)
               for e in embeddings)
        f.write(text)
     

def export_vocabulary(path, vocab_size, word_index):
    with open(path, 'w') as f:
        # padding
        f.writelines(['0\n'])
        words = list(word_index.keys())
        if '\n' in words:
            index = words.index('\n')
            words.remove('\n')
            words.insert(index, '\\n')
        f.write('\n'.join(words[:vocab_size]))


def plot_augmented_images_dataset_samples(
        dataset, fig_index=0, buffer_size=10000):

    dataset.shuffle(buffer_size)
    samples = dataset.take(8)

    i = 1
    fig = plt.figure(fig_index)
    for example in samples.as_numpy_iterator():
        subp = plt.subplot(2, 4, i) 
        subp.imshow(example['image'])
        subp.set_title(str(example['label']))
        i = i+1
    plt.show()
    

def plot_backbone_featuremap_samples(features, fig_index=0):
    fig = plt.figure(fig_index)
    for i in range(1, 7):
        subp = plt.subplot(2, 3, i) 
        subp.imshow(features[0,:,:,i])
    plt.show()
 

def plot_series(x, y, scale='log'):
    fig = plt.figure()
    sub = fig.add_subplot()
    sub.set_yscale(scale)
    sub.plot(x, y)
    plt.show()


def plot_hist(history, key, path, with_val=True, sufix=''):
    train_series = history.history[key]
    epochs = range(len(train_series))
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.plot(epochs, train_series, color='blue')

    if with_val:
        val_series = history.history['val_' + key]
        plt.plot(epochs, val_series, color='red')
        plt.legend(['training', 'validation'])
    else:
        plt.legend(['training'])

    return plt.show()


def export_projector_data(embeddings, meta_path, logs_path):
    embeddings_var = tf.Variable(embeddings, name='embeddings')
    checkpoint = tf.train.Checkpoint(embedding=embeddings_var)
    checkpoint.save(os.path.join(logs_path, 'embeddings.ckpt'))
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embeddings'
    embedding.metadata_path = meta_path 
    projector.visualize_embeddings(logs_path, config)

