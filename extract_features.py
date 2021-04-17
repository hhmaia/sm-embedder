import argparse
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0 

from preprocessing import augmented_images_dataset_from_tfrecord


def prepare_dataset_for_inference(dataset, batch_size=128, prefetch_size=512):
    dataset = dataset.map(lambda ex: (ex['image'], ex['label']))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(512)
    return dataset


def load_model(input_shape):
    return EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            pooling='avg',
            input_shape=input_shape)


def featuremap_example(featuremap, label):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    feature = {
            'label': _int64_feature(label),
            'featuremap': _float_feature(featuremap)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_featuremaps_to_tfrecord(output_record, dataset, model):
    i = 0
    with tf.io.TFRecordWriter(output_record) as writer:
        dataset_iter = dataset.as_numpy_iterator()
        for images_batch, labels_batch in dataset:
            featuremaps_batch = model.predict(images_batch)
            for featuremap, label in zip(featuremaps_batch, labels_batch):
                example = featuremap_example(featuremap, label)
                writer.write(example.SerializeToString())
                i = i + 1
                print('\r  |_ #{} featuremaps processed.'.format(i), end='')


def feature_maps_dataset_from_tfrecord(record_path):
    def decode_example(example):
        schema = {
                'label': tf.io.FixedLenFeature([], tf.int64),
                'featuremap': tf.io.FixedLenFeature([1280], tf.float32)
        }
        example = tf.io.parse_single_example(example, schema)
        return example

    dataset = tf.data.TFRecordDataset(record_path, num_parallel_reads=4)
    dataset = dataset.map(decode_example)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', '--input_path', type=str, required=True,
            help='Raw images tfrecord path, with fields "images" and "labels"')
    parser.add_argument('-o', '--output_path', type=str, required=True,
            help="Path to the output tfrecord")
    parser.add_argument('-s', '--input_shape', type=tuple, default=(300, 300, 3), 
            help="Shape of the input images")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
            help="Batch size for inference")
    parser.add_argument('-p', '--prefetch_size', type=int, default=512,
            help="Prefetch parameter for the dataset")
    args = parser.parse_args()

    model = load_model(args.input_shape)
    ds = augmented_images_dataset_from_tfrecord(args.input_path)
    ds = prepare_dataset_for_inference(ds , args.batch_size, args.prefetch_size)
    write_featuremaps_to_tfrecord(args.output_path, ds, model)



