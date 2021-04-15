'''
Augment images and save them as a tfrecord.

'''

import os
import argparse
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


_RANDOM_SEED = 983437


def iterator_from_directory(input_dir, batch_size=32, output_dir=None):
    image_generator = ImageDataGenerator(
            rotation_range=36,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=36,
            zoom_range=0.1,
            channel_shift_range=0.1,
            fill_mode='reflect',
            horizontal_flip=True,
            vertical_flip=True,
            dtype=tf.dtypes.uint16)

    iterator = image_generator.flow_from_directory(
            input_dir,
            target_size=(300,300),
            class_mode='sparse',
            batch_size=batch_size,
            shuffle=True,
            seed=_RANDOM_SEED)
            
    return iterator


def image_example(image_raw, label):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    feature = {
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(tf.io.encode_jpeg(image_raw).numpy())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(record_fname, iterator, n_batches):
    with tf.io.TFRecordWriter(record_fname) as writer:
        for batch_id in range(n_batches):
            print('\n:: Processing batch #{}...'.format(batch_id))
            batch = iterator.next()
            i = 1
            for image_raw, label in zip(*batch): 
                tf_example = image_example(image_raw, label)
                writer.write(tf_example.SerializeToString())
                print('\r  |_ file #{}.'.format(i), end='')
                i = i + 1


def dataset_from_tfrecord(record_path):
    def decode_example(example):
        schema = {
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image_raw': tf.io.FixedLenFeature([], dtype=tf.string)
        }
        example = tf.io.parse_single_example(example, schema)
        example['image'] = tf.io.decode_jpeg(example.pop('image_raw'))
        return example 
    
    dataset = tf.data.TFRecordDataset(record_path, num_parallel_reads=4)
    dataset = dataset.map(decode_example)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generates features from images directory")
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('-b', type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()

    it = iterator_from_directory(args.i, args.b)
    write_tfrecord(args.o, it, args.n)

