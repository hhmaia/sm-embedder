import os
from IPython.core.debugger import Tracer

from dataset_splitter import split_dataset 
from preprocessing import \
        iterator_from_directory, \
        write_images_tfrecord, \
        images_dataset_from_tfrecord
from extract_features import \
        load_model as load_backbone, \
        prepare_dataset_for_inference, \
        write_featuremaps_to_tfrecord, \
        featuremaps_dataset_from_tfrecord


dataset_dir = "/home/duo/datasets/desafio/"
build_dir = "build"
dataset_splits_dir = os.path.join(build_dir, "datasets")
input_images_target_size = (300, 300)
input_images_target_shape= (300, 300, 3)

# Create base directories for artefacts if they don't exist
if not os.path.isdir(build_dir):
    os.mkdir(build_dir)
if not os.path.isdir(dataset_splits_dir):
    os.mkdir(dataset_splits_dir)

# Splits the original dataset into 3 subdirectories for training, validation
# and testing, preserving the original dataset intact. 
split_dataset(dataset_dir, dataset_splits_dir, (.8, .1, .1), verbose=True)

# This first iterator yield augmented images that will be used for training.
train_it = iterator_from_directory(
        input_dir=os.path.join(dataset_splits_dir, 'train'),
        batch_size=64,
        augment=True,
        target_size=input_images_target_size)

# Writes a tfrecord containing the augmented images and their labels.
# I'm using tfrecord for performance and to facilitate dataset manipulation
# in the next steps.
write_images_tfrecord(
        record_fname=os.path.join(build_dir, "train.tfrecord"), 
        iterator=train_it,
        n_batches=200)

# Saving hashes from directory names to a file, to later recover class names 
with open(os.path.join(build_dir, 'labels.tsv'), 'w') as f:
    f.write('\n'.join(train_it.class_indices.keys())

# For validation and test datasets, we dont need augmentation. Also, we need
# to know how many files we are going to process so we don't repeat images. 
for subset in ['val', 'test']:
    dir_path = os.path.join(dataset_splits_dir, subset)

    # Recursively count files inside subset directories
    n_files = sum([len(files) for r, d, files in os.walk(dir_path)])

    # No augmentation
    subset_it = iterator_from_directory(
            input_dir=dir_path,
            batch_size=n_files,
            target_size=input_images_target_size)

    # Only one batch, with all files in it
    write_images_tfrecord(
            record_fname=os.path.join(build_dir, subset+'.tfrecord'),
            iterator=subset_it, 
            n_batches=1)

# Now here it's a trick to speed up training.
# The backbone is not going to be trainable.
# Knowing that, we can pre-compute all the outputs for the backbone (from now
# on we are going to call them featuremaps), store them on a tfrecord, 
# and later, feed them to the head of the model during training.
# This save A LOT of processor time.
# Doing it only for training and validation, because we wanna test the entire
# pipeline with the testing dataset.
# Go grab a coffe, it'll take some time.
for subset in ['train', 'val']:
    backbone_model = load_backbone(input_images_target_shape)
    images_record = os.path.join(build_dir, subset+'.tfrecord')
    featuremaps_record = os.path.join(build_dir, subset+'_featuremaps.tfrecord')

    dataset = images_dataset_from_tfrecord(images_record)  
    processed_dataset = prepare_dataset_for_inference(dataset) 

    # Compute the featuremaps and save them
    write_featuremaps_to_tfrecord(
        featuremaps_record, processed_dataset, backbone_model) 

