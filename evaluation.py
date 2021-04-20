import numpy as np
import tensorflow as tf

from preprocessing import images_dataset_from_tfrecord
from extract_features import prepare_dataset_for_inference
from model import load_inference_model
from mlutils import export_projector_data


# Uses euclidian norm to get the nearest center to an embedding
# This center indicates the class for the embedding
# embeddings.shape == (n_samples, embedding_dim)
# centers.shape == (num_classes, embedding_dim)
# TODO: convert this to tensorflow code
def labels_from_embeddings(embeddings, centers):
    # Get squared distances from the centers to the embeddings,
    # This is acchieved with broadcast subtraction of the centers by the 
    # embeddings and squaring
    squared_diffs = [np.square(centers - e) for e in embeddings]
    # Sum over all the axis of an embedding and get the square root, for all 
    # embeddings to complete the calculation of the distance
    distances = np.sqrt(np.sum(squared_diffs, axis=-1))
    # Gets the index of the minimum values over the embeddings axis,
    # this will be the label for this embedding
    min_dist_labels = np.argmin(distances, axis=-1)
    return min_dist_labels 


def inference(head_checkpoint='build/checkpoints/head/',
              dataset_record='build/test.tfrecord', 
              input_shape=(300, 300, 3)):
    # Here we use a dataset containing raw images (numpy ndarray) and labels
    # for evaluation of the whole model.
    dataset = images_dataset_from_tfrecord(dataset_record)
    dataset = prepare_dataset_for_inference(dataset)
    model = load_inference_model(head_checkpoint, input_shape)
    return model.predict(dataset)


def export_embeddings_for_visualization(embeddings, filename, centers_filename):
    with open(filename, 'w') as f:
        # The top entries are the centers from created after training the head
        with open(centers_filename) as g:
            f.write(g.read())

        # After them, we put the inference computed embeddings
        for e in embeddings:
            for coord in e:
                f.write(str(coord) + '\t')
            f.write('\n') 


def export_metadata(filename, dataset_labels, num_classes):
    with open(filename, 'w') as f:
        # First the labels for the centers
        for i in range(num_classes):
            f.write('CENTER #' + str(i) + ' \n')

        for label in dataset_labels: 
            f.write(str(label) + '\n')


def get_results_summary(labels_pred, labels_true):
    labels_is_right = (labels_pred == labels_true)
    n_labels = len(labels_is_right)
    right_labels = np.count_nonzero(labels_is_right)
    wrong_labels = n_labels - right_labels
    values = (right_labels, wrong_labels, right_labels/n_labels, n_labels)
    summary = ('Right labels: {}\n'
               'Wrong labels: {}\n'
               'Ratio right to total: {}\n'
               'Total: {}').format(*values)
    return summary


if __name__ == '__main__':
    test_dataset_embeddings = inference() 

    # Loading the dataset again to get the labels
    labels_dataset = images_dataset_from_tfrecord('build/test.tfrecord') 
    labels_it = labels_dataset.map(lambda x: x['label']).as_numpy_iterator()
    real_labels = np.array(list(labels_it))

    # loading centers from tsv file
    centers = np.loadtxt('build/centers.tsv')
    infered_labels = labels_from_embeddings(test_dataset_embeddings, centers)

    summary = get_results_summary(infered_labels, real_labels)
    print(summary)

    # Export the centers and the computed embeddings to a single file for visuals 
    export_embeddings_for_visualization(
            embeddings=test_dataset_embeddings,
            filename='build/inference_embeddings.tsv',
            centers_filename='build/centers.tsv')

    export_metadata('build/metadata.tsv', real_labels, num_classes=20) 

    # Exporting embeddings data to tensorboard. I don't know why, but the labels
    # are not working... you can load them manually from "./build/metadata.tsv"
    embs_for_projector = np.loadtxt('build/inference_embeddings.tsv')
    export_projector_data(
            embs_for_projector,
            'build/metadata.tsv',
            'build/logs/') 
