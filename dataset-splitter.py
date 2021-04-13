import os
import sys
import argparse
import random
import shutil


def split_subdir(input_dir, splits):
    '''
    Here some hashing and the module of the hash could have been used to do 
    the spliting, but as the number of files is small, we are going to do it
    the simple way.

    returns: a dictionary with the filenames for each subset
    '''
    fnames = os.listdir(input_dir)
    random.shuffle(fnames)
    total_nfiles = len(fnames)

    # how many files for each split?
    dir_nfiles = [int(total_nfiles*split) for split in splits]
    train_n, val_n, test_n = dir_nfiles 

    # if there are remaning files, put then in the training split
    train_n = train_n + (total_nfiles - sum(dir_nfiles))
    dirname_fnames = {
            'train': fnames[:train_n],
            'val': fnames[train_n:(train_n+val_n)],
            'test': fnames[-test_n:]}

    # verifies that no file is left behind
    assert(train_n == len(dirname_fnames['train']))
    assert(val_n == len(dirname_fnames['val']))
    assert(test_n == len(dirname_fnames['test']))
    assert(train_n + val_n + test_n == total_nfiles)

    return dirname_fnames


def _split_dataset(dataset_dir, out_dir, splits, verbose=False):
    '''
    Supose you have the directory dataset_dir with N subdirectories, one for
    each class, each of them with X examples.
    Creates a new directory out_dir with train, val and test subdirectories, 
    each one with the same N subdirectories, but with a fraction of the X
    examples. 

    Arguments:
        dataset_dir: path of the original dataset directory.
        out_dir: output directory, where the splits will be created.
        splits: list of 3 floats, representing the fraction of the split
        for, respectively, train, validation and test subsets.
    '''

    def validate_args():
        assert(os.path.isdir(dataset_dir))
        # each split needs to be in [0,1]
        for split in splits:
            assert(0. <= split <= 1.)
        # sum of the splits needs to be less or equal 1
        print(sum(splits))
        assert(sum(splits) <= 1.)


    validate_args()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        for subset in ['train', 'val', 'test']:
            p = os.path.join(out_dir, subset)
            os.mkdir(p)

    # dir names for all the classes
    subdir_names = os.listdir(dataset_dir)
    # iterates over all the classes
    for subdir in subdir_names:
        source_path = os.path.join(dataset_dir, subdir)
        subset_fnames = split_subdir(source_path, splits)
        # iterate over the subsets
        for subset, fnames in subset_fnames.items():
            for fname in fnames:
                source = os.path.join(source_path, fname)
                dest_path = os.path.join(out_dir, subset, subdir)
                if not os.path.isdir(dest_path):
                    os.mkdir(dest_path)
                dest = os.path.join(dest_path, fname)
                shutil.copy2(source, dest)
                if verbose:
                    print(source)
                    print(dest)
    return


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(
            description='Script for splitting the dataset into training,'
            'validation and testing subsets.\n\n'
            'Note that the sum of the splits needs to be <= 1')
    parser.add_argument('-d', '--dataset-dir', type=str, required=True,
            help='Path to the dataset dir')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
            help='Output directory for the subsets')
    parser.add_argument('-s', '--splits', nargs=3, type=float, required=True,
            metavar=('TRAIN', 'VALIDATION', 'TEST'),
            help='Size of the training split for training, validation and ' 
            'test subsets, each in the interval [0,1), and their sum '
            'also in [0,1).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='Turns on verbose output')

    args = parser.parse_args()

    _split_dataset(args.dataset_dir, args.output_dir, args.splits, args.verbose)
