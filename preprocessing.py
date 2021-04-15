'''
Generate features and saves then as a tfrecord to accelerate training
'''

import os
import argparse
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


_RANDOM_SEED = 983437


'''
Proof of concept.
I only need to run this accasionally.
Don't use this script if you need performance.
It'll eat all your memory. You are warned! 
'''


def iterator_from_directory(input_dir, output_dir=None):
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
            rescale=255.,
            dtype=tf.dtypes.float32
            )

    iterator = image_generator.flow_from_directory(
            input_dir,
            target_size=(300,300),
            class_mode='sparse',
            batch_size=1024,
            shuffle=True,
            seed=_RANDOM_SEED,
            save_to_dir=output_dir,
            )
            
    return iterator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generates features from images directory")
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)
    args = parser.parse_args()

    it = iterator_from_directory(args.i, args.o)
    labels = []
    for batch in it: 
        for i in batch:
            labels.append(i[1])
            del(i)
        del(batch)
    print(labels)
