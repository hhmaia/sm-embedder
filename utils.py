import matplotlib.pyplot as plt

def plot_augmented_images_dataset_samples(
        dataset, fig_index=0, buffer_size=10000):

    dataset.shuffle(buffer_size)
    samples = dataset.take(6)

    i = 1
    fig = plt.figure(fig_index)
    for example in samples.as_numpy_iterator():
        subp = plt.subplot(2, 3, i) 
        subp.imshow(example['image'])
        subp.set_title(str(example['label']))
        i = i+1
    plt.show()
    

def plot_featuremap_samples(features, fig_index=0):
    fig = plt.figure(fig_index)
    for i in range(1, 7):
        subp = plt.subplot(2, 3, i) 
        subp.imshow(features[0,:,:,i])
    plt.show()
 

