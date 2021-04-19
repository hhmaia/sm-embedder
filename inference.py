import numpy as np
import tensorflow as tf

from preprocessing import images_dataset_from_tfrecord
from extract_features import prepare_dataset_for_inference
from model import load_inference_model
from mlutils import export_projector_data


head_ckp_path = 'build/checkpoints/head/'
record_path = 'build/test.tfrecord'

dataset = images_dataset_from_tfrecord(record_path)
dataset = prepare_dataset_for_inference(dataset)
model = load_inference_model(head_ckp_path, (300, 300, 3))

out = model.predict(dataset)
centers = np.loadtxt('build/centers.tsv')


with open('build/embeddings_output.tsv', 'w') as f:
    with open('build/centers.tsv') as g:
        f.write(g.read())

    for emb in out:
        for coord in emb:
            f.write(str(coord) + '\t')
        f.write('\n') 


dataset = images_dataset_from_tfrecord(record_path)

with open('build/metadata.tsv', 'w') as f:
    for i in range(20):
        f.write('CENTRO ' + str(i) + ' \n')

    for label in dataset.map(lambda x: x['label']).as_numpy_iterator():
        f.write(str(label) + '\n')


squared_diffs = [np.square(centers - emb) for emb in out]
s = np.sqrt(np.sum(squared_diffs, axis=-1))
infered_labels = np.argmin(s, axis=-1)
labels_it = dataset.map(lambda x: x['label']).as_numpy_iterator()
real_labels = np.array(list(labels_it))
print(infered_labels == real_labels)

embs_for_projector = np.loadtxt('build/embeddings_output.tsv')
export_projector_data(
        embs_for_projector,
        'build/metadata.tsv',
        'build/logs/validation') 
