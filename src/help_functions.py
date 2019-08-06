import numpy as np
import tensorflow as tf
import pickle  # for saving python objects
import time  # for timing SPN learning duration
import os  # for reading and writing files and directories
import itertools  # for combining
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skimage.transform import resize
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
from spn.algorithms.Inference import likelihood  # likelihood inference

# Custom colormap
toy_cmap = ListedColormap([[1, 0.93, 0.03, 1.],
                           [1., 0.49, 0., 1.],
                           [0., 0.62, 0.89, 1.],
                           [0.9, 0, 0.49, 1.]])


def create_dir(dir_path, force_overwrite=False):
    if not force_overwrite:
        assert not os.path.isdir(dir_path), "Directory '%s' does exist already." % dir_path
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        print("Creation of the directory '%s' failed!" % dir_path)
        raise


def load_mnist(num_train_samples=60000, num_test_samples=10000, res=28, normalization=False):
    """Loads the MNIST digit image dataset via TensorFlow Keras."""
    # Fetch train (60.000 images) and test data (10.000 images)
    (train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = tf.keras.datasets.mnist.load_data()

    # Convert data type
    train_images_raw = np.asarray(train_images_raw, dtype=np.float32)
    test_images_raw = np.asarray(test_images_raw, dtype=np.float32)

    # Convert data by squaring the pixel intensities
    # train_images_raw = np.square(train_images_raw)
    # test_images_raw = np.square(test_images_raw)

    # Cut data
    train_labels = train_labels_raw[:num_train_samples]
    test_labels = test_labels_raw[:num_test_samples]
    train_images = train_images_raw[:num_train_samples]
    test_images = test_images_raw[:num_test_samples]

    # Normalize data
    if normalization:
        train_images_mean = np.mean(train_images, 0)
        train_images_std = np.std(train_images, 0)
        std_eps = 1e-7
        train_images = (train_images - train_images_mean) / (train_images_std + std_eps)
        test_images = (test_images - train_images_mean) / (train_images_std + std_eps)

    # Resize images
    if res != 28:
        train_images = np.array(list(map(lambda x: resize(x, (res, res)), train_images)))
        test_images = np.array(list(map(lambda x: resize(x, (res, res)), test_images)))

    # Reshape and convert data
    train_images = train_images.reshape(-1, res * res).astype('float32')
    test_images = test_images.reshape(-1, res * res).astype('float32')

    return (train_images, train_labels), (test_images, test_labels)


def generate_linear(num_train_samples=200, num_test_samples=32 * 32, noise=False):
    """Generates a linear 2D two-class problem dataset and returns it."""
    # Generate the dataset
    # Initialize two 2D point sets with num_train_samples and num_test_samples resp.
    train_samples = np.random.uniform(0.0, 128.0, (num_train_samples, 2))
    num_test_samples = np.sqrt(num_test_samples)
    test_samples = list(itertools.product(np.linspace(0.5, 127.5, num_test_samples),
                                          np.linspace(0.5, 127.5, num_test_samples)))

    # compute train and test labels
    labels = [[], []]

    for k, samples in enumerate((train_samples, test_samples)):
        for i in range(0, len(samples)):
            sample = samples[i]
            x = 8 * np.random.poisson()
            if sample[0] < 64:
                if sample[0] + x > 70 and k == 0 and noise:
                    labels[k] = np.append(labels[k], [1])
                else:
                    labels[k] = np.append(labels[k], [0])
            else:
                if sample[0] - x < 58 and k == 0 and noise:
                    labels[k] = np.append(labels[k], [0])
                else:
                    labels[k] = np.append(labels[k], [1])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def generate_non_linear(num_train_samples=200, num_test_samples=32 * 32, noise=False):
    """Generates a non-linear, 2D two-class problem dataset and returns it."""
    # Generate the dataset
    # Initialize two 2D point sets with num_train_samples and num_test_samples resp.
    train_samples = np.random.uniform(0.0, 128.0, (num_train_samples, 2))
    num_test_samples = np.sqrt(num_test_samples)
    test_samples = list(itertools.product(np.linspace(0.5, 127.5, num_test_samples),
                                          np.linspace(0.5, 127.5, num_test_samples)))

    # compute train and test labels
    labels = [[], []]

    for k, samples in enumerate((train_samples, test_samples)):
        for i in range(0, len(samples)):
            sample = samples[i]
            x = np.random.poisson()
            if 16 <= sample[0] <= 112 and 16 <= sample[1] <= 112:
                if sample[0] < 40 and sample[1] < 40:
                    if np.sqrt((40 - sample[0]) ** 2 + (40 - sample[1]) ** 2) <= 24:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [0])
                        else:
                            labels[k] = np.append(labels[k], [1])
                    else:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [1])
                        else:
                            labels[k] = np.append(labels[k], [0])
                elif sample[0] > 88 and sample[1] < 40:
                    if np.sqrt((88 - sample[0]) ** 2 + (40 - sample[1]) ** 2) <= 24:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [0])
                        else:
                            labels[k] = np.append(labels[k], [1])
                    else:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [1])
                        else:
                            labels[k] = np.append(labels[k], [0])
                elif sample[0] > 88 and sample[1] > 88:
                    if np.sqrt((88 - sample[0]) ** 2 + (88 - sample[1]) ** 2) <= 24:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [0])
                        else:
                            labels[k] = np.append(labels[k], [1])
                    else:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [1])
                        else:
                            labels[k] = np.append(labels[k], [0])
                elif sample[0] < 40 and sample[1] > 88:
                    if np.sqrt((40 - sample[0]) ** 2 + (88 - sample[1]) ** 2) <= 24:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [0])
                        else:
                            labels[k] = np.append(labels[k], [1])
                    else:
                        if x > 3 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [1])
                        else:
                            labels[k] = np.append(labels[k], [0])
                else:
                    if (sample[0] - 8 * x < 0 or 128 < sample[0] + 8 * x \
                        or sample[1] - 8 * x < 0 or 128 < sample[1] + 8 * x) \
                            and k == 0 and noise:
                        labels[k] = np.append(labels[k], [0])
                    else:
                        labels[k] = np.append(labels[k], [1])
            else:
                if (32 < sample[0] + 8 * x and sample[0] - 8 * x < 96 \
                    and 32 < sample[1] + 8 * x and sample[1] - 8 * x < 96) \
                        and k == 0 and noise:
                    labels[k] = np.append(labels[k], [1])
                else:
                    labels[k] = np.append(labels[k], [0])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def generate_three(num_train_samples=200, num_test_samples=32 * 32, noise=False):
    """Generates a non-linear, 2D three-class problem dataset and returns it."""
    # Generate the dataset
    # Initialize two 2D fields with num_train_samples and num_test_samples resp.
    train_samples = np.random.uniform(0.0, 128.0, (num_train_samples, 2))
    num_test_samples = np.sqrt(num_test_samples)
    test_samples = list(itertools.product(np.linspace(0.5, 127.5, num_test_samples),
                                          np.linspace(0.5, 127.5, num_test_samples)))

    # Compute train and test labels
    labels = [[], []]

    for k, samples in enumerate((train_samples, test_samples)):
        for i in range(0, len(samples)):
            x = 8 * np.random.poisson()
            sample = samples[i]
            if sample[1] >= np.cos(sample[0] / 128 * np.pi) * 50 + 78:
                if sample[1] - x < np.cos(sample[0] / 128 * np.pi) * 50 + 78 - 4 and k == 0 and noise:
                    if sample[1] >= (sample[0] - 30) * 2:
                        if sample[1] - 1.2 * x < (sample[0] - 30) * 2 - 4 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [2])
                        else:
                            labels[k] = np.append(labels[k], [1])
                    else:
                        if sample[1] + 1.2 * x >= (sample[0] - 30) * 2 + 4 and k == 0 and noise:
                            labels[k] = np.append(labels[k], [1])
                        else:
                            labels[k] = np.append(labels[k], [2])
                else:
                    labels[k] = np.append(labels[k], [0])
            else:
                if sample[1] + x > np.cos(sample[0] / 128 * np.pi) * 50 + 78 + 4 and k == 0 and noise:
                    labels[k] = np.append(labels[k], [0])
                elif sample[1] >= (sample[0] - 30) * 2:
                    if sample[1] - 1.2 * x < (sample[0] - 30) * 2 - 4 and k == 0 and noise:
                        labels[k] = np.append(labels[k], [2])
                    else:
                        labels[k] = np.append(labels[k], [1])
                else:
                    if sample[1] + 1.2 * x >= (sample[0] - 30) * 2 + 4 and k == 0 and noise:
                        labels[k] = np.append(labels[k], [1])
                    else:
                        labels[k] = np.append(labels[k], [2])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def generate_gaussian(num_train_samples=200, num_test_samples=32 * 32, noise=None):
    """Generates a non-linear, 2D three-class problem dataset
    based on Gaussian distributions and returns it."""

    # Train and test samples
    samples = [[], []]
    # Train and test labels
    labels = [[], []]

    # Initialize two 2D fields with num_train_samples and num_test_samples resp.
    for i, num_samples in enumerate([num_train_samples, num_test_samples]):
        n_0 = int(0.6 * num_samples)
        n_0a = int(0.45 * n_0)
        n_0b = int(0.1 * n_0)
        n_0c = n_0 - n_0a - n_0b
        n_1 = int(0.25 * num_samples)
        n_2 = num_samples - n_0 - n_1

        class_0a = np.random.multivariate_normal((40, 100), ((130, 0), (0, 35)), n_0a)
        class_0b = np.random.multivariate_normal((100, 80), ((50, 0), (0, 50)), n_0b)
        class_0c = np.random.multivariate_normal((85, 35), ((80, -40), (-40, 80)), n_0c)
        class_1 = np.random.multivariate_normal((30, 60), ((100, 0), (0, 200)), n_1)
        class_2 = np.random.multivariate_normal((60, 25), ((80, -60), (-60, 60)), n_2)

        samples[i] = np.concatenate((class_0a, class_0b, class_0c, class_1, class_2), axis=0)
        labels[i] = np.concatenate((n_0 * [0], n_1 * [1], n_2 * [2]), axis=0)

    # Convert data type
    train_samples = np.asarray(samples[0], dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(samples[1], dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def generate_toy_color(num_train_samples=10000, num_test_samples=1000):
    # Generate toy color images
    train_samples = np.random.randint(low=0, high=4, size=num_train_samples * 5 * 5).reshape((num_train_samples, 5 * 5))
    test_samples = np.random.randint(low=0, high=4, size=num_test_samples * 5 * 5).reshape((num_test_samples, 5 * 5))

    # Label the images
    labels = [[], []]
    for k, samples in enumerate((train_samples, test_samples)):
        for i, sample in enumerate(samples):
            # Give label 1 iff top three pixels are different or the lower two corners are equal
            # if sample[1] == sample[2] == sample[3] or 0 == sample[20] == sample[24]:
            if sample[1] != sample[2] != sample[3] and sample[1] != sample[3] or sample[20] == sample[24]:
                labels[k] = np.append(labels[k], [1])
            else:
                labels[k] = np.append(labels[k], [0])

    # Convert dataset into one-hot representation
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def convert_toy_data_to_one_hot(samples):
    one_hot_samples = np.empty((0, 4 * 25))
    for sample in samples:
        one_hot_sample = np.zeros((25, 4))
        one_hot_sample[np.arange(25), sample] = 1
        one_hot_samples = np.vstack((one_hot_samples, one_hot_sample.flatten()))
    return one_hot_samples


def evaluate_spn_performance(spn, train_samples, train_labels, test_samples, test_labels, label_idx, stats_file=None):
    """Evaluates the performance of a given SPN by means of given train and test data.
    Returns a boolean vector containing an entry for the correctness of each single test prediction.

    :param spn: the Sum-Product-Network
    :param train_samples: list of training samples (without labels) of shape (X, Y)
    :param train_labels: list of train labels of shape (X, 1)
    :param test_samples: list of test samples (without labels) of shape (Z, Y)
    :param test_labels: list of test labels of shape (Z, 1)
    :param label_idx: position of the label when fed into the SPN
    :param stats_file: optional output file to save evaluation results
    :return: boolean vector of length Z where entry i is True iff test label i was correctly predicted
    :return: vector of predicted test labels
    """

    num_train_samples = len(train_samples)
    num_test_samples = len(test_samples)

    # Predict train labels
    train_performance_data = np.column_stack((train_samples, [np.nan] * num_train_samples))
    train_predictions = mpe(spn, train_performance_data)
    predicted_train_labels = train_predictions[:, label_idx]

    # Accuracy on train set
    correct_train_answers = np.reshape(train_labels, -1) == predicted_train_labels
    acc = np.count_nonzero(correct_train_answers) / num_train_samples

    train_text = "\n\nTrain set performance:" \
                 "\nTrain sample count: %d" \
                 "\nTrain set accuracy: %.2f %%" % \
                 (num_train_samples, acc * 100)
    print(train_text, end='')
    if stats_file is not None:
        stats_file.write(train_text)

    # Predict test labels
    test_performance_data = np.column_stack((test_samples, [np.nan] * num_test_samples))
    test_predictions = mpe(spn, test_performance_data)
    predicted_test_labels = test_predictions[:, label_idx]

    # Accuracy on test set
    correct_test_answers = np.reshape(test_labels, -1) == predicted_test_labels
    acc = np.count_nonzero(correct_test_answers) / num_test_samples

    test_text = "\n\nTest set performance:" \
                "\nTest sample count: %d" \
                "\nTest set accuracy: %.2f %%\n" % \
                (num_test_samples, acc * 100)
    print(test_text)
    if stats_file is not None:
        stats_file.write(test_text)

    return (predicted_train_labels, correct_train_answers), (predicted_test_labels, correct_test_answers)


def plot_samples(samples, labels, plot_title, plot_pdf=None, classes=None, size=10, colors=None,
                 plot_labels=None, test_sample=None):
    """Generates a coloured scatter plot."""
    if classes is None:
        classes = np.sort(np.unique(labels))

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, c in enumerate(classes):
        # Extract samples of class c
        samples_of_c = np.array([samples[k] for k in range(0, len(samples)) if labels[k] == c])
        if samples_of_c.size == 0:
            continue
        c_x = samples_of_c.transpose()[0].tolist()
        c_y = samples_of_c.transpose()[1].tolist()
        if colors is not None and plot_labels is not None:
            plt.scatter(c_x, c_y, label=plot_labels[i], s=size, c=colors[i])
        else:
            plt.scatter(c_x, c_y, label="Class %d" % i, s=size)

    if test_sample is not None:
        plt.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3',
                    label=r'$z_{\mathrm{test}}$')

    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])
    plt.legend()
    plt.title(plot_title)
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def plot_likelihoods(spn, classes, plot_pdf, res=100, test_sample=None):
    """Generates for each class a 2D heightmap of likelihoods."""
    if len(classes) > 10:
        raise Exception("Not more than 10 distinct classes allowed for likelihood "
                        "plot, but given %d." % len(classes))

    # Samples
    lin = np.linspace(0.5, 127.5, res)
    xgrid, ygrid = np.meshgrid(lin, lin)
    samples = np.asarray(list(itertools.product(lin, lin)), dtype=np.float32)

    # Compute likelihood values for each sample regarding each class
    n = res ** 2
    likelihoods = np.empty((0, n))
    for c in classes:
        data = np.column_stack((samples, c * np.ones(n)))
        likelihoods_of_c = likelihood(spn, data).reshape((1, n)) * 100000
        likelihoods = np.append(likelihoods, likelihoods_of_c, axis=0)

    # Determine colormap levels
    l_max = np.max(likelihoods)
    l_min = np.min(likelihoods)
    levels = np.linspace(l_min, l_max, 15, endpoint=True)

    # Get default colormap for default colors
    def_cmap = plt.get_cmap("tab10")

    # Plot the likelihoods
    for i, c in enumerate(classes):
        fig, ax = plt.subplots(figsize=(6, 5))
        zgrid = griddata(samples, likelihoods[i], (xgrid, ygrid))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), def_cmap(i)])
        cont = plt.contourf(xgrid, ygrid, zgrid, levels, cmap=cmap)
        if test_sample is not None:
            ax.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3',
                       label=r'$z_{\mathrm{test}}$')
            plt.legend()
        plt.title('Learned Probability Distribution\nfor Class %d' % c)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_xlim([0, 128])
        ax.set_ylim([0, 128])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        fig.colorbar(cont, cax=cax)
        plot_pdf.savefig(fig)
        plt.show()


def plot_decision_boundaries(spn, marg_spn, classes, plot_pdf, res=100, test_sample=None):
    """Generates a 2D heightmap of marginal likelihoods and draws decision boundaries."""
    if len(classes) > 10:
        raise Exception("Not more than 10 distinct classes allowed for decision "
                        "boundary plot, but given %d." % len(classes))

    # Samples
    lin = np.linspace(0.5, 127.5, res)
    y, x = np.meshgrid(lin, lin)
    samples = np.asarray(list(itertools.product(lin, lin)), dtype=np.float32)

    # Predictions and likelihoods
    data = np.column_stack((samples, res ** 2 * [np.nan]))
    mpe_data = mpe(spn, data)
    predictions = mpe_data[:, -1]
    likelihoods = likelihood(marg_spn, samples).reshape((res, res)) * 100000

    # Determine colormap levels
    l_max = np.max(likelihoods)
    l_min = np.min(likelihoods)
    levels = np.linspace(l_min, l_max, 15, endpoint=True)

    # Get default colormap for default colors
    def_cmap = plt.get_cmap("tab10")

    conts = []

    # Plot the decision boundary and likelihoods
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(classes):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), def_cmap(i)])
        preds_c = np.array([1 if pred == c else 0 for pred in predictions]).reshape((res, res))
        conts.append(plt.contourf(x, y, likelihoods * preds_c, levels, cmap=cmap))
        plt.contour(x, y, preds_c, [0.5], colors="k")

    if test_sample is not None:
        plt.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3',
                    label=r'$z_{\mathrm{test}}$')
        plt.legend()

    plt.axis('equal')
    plt.title('Decision Boundary\nand Marginal Log-Likelihood')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])
    divider = make_axes_locatable(ax)

    for i, c in enumerate(classes):
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        if i == len(classes) - 1:
            fig.colorbar(conts[i], cax=cax)
        else:
            fig.colorbar(conts[i], cax=cax, ticks=[])

    plot_pdf.savefig(fig)
    plt.show()


def plot_influences(influences, samples, plot_title, plot_pdf=None, test_sample=None):
    """Generates a coloured scatter plot where colour indicates influence value."""
    range = np.max(np.abs(influences))
    x = samples.transpose()[0].tolist()
    y = samples.transpose()[1].tolist()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=influences, cmap="RdBu", vmin=-range, vmax=range)
    if test_sample is not None:
        ax.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3',
                   label=r'$z_{\mathrm{test}}$')
        plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title(plot_title)
    fig.colorbar(sc, ax=ax)
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def plot_influences_with_multiple_colors(influences, samples, labels, plot_title, plot_pdf, test_sample=None):
    """Generates a coloured scatter plot where colour indicates influence value.
    It uses individual color scales for each class label instead of only a single
    one for all classes."""
    classes = np.sort(np.unique(labels))
    if len(classes) > 3:
        raise Exception("Not more than 3 distinct classes allowed for influence "
                        "value plot with multiple colors, but given %d." % len(classes))

    samples_x = samples[:, 0]
    samples_y = samples[:, 1]

    # preconfigured colormaps
    cmaps = ["winter", "autumn", "summer"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(classes):
        class_c = labels == c
        influences_of_c = np.extract(class_c, influences)
        samples_x_c = np.extract(class_c, samples_x)
        samples_y_c = np.extract(class_c, samples_y)
        sc = ax.scatter(samples_x_c, samples_y_c, c=influences_of_c, cmap=cmaps[i])
        fig.colorbar(sc, ax=ax)

    if test_sample is not None:
        ax.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3',
                   label=r'$z_{\mathrm{test}}$')
        plt.legend()

    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title(plot_title)
    plot_pdf.savefig(fig)
    plt.show()


def plot_gradients(gradients, samples, plot_title, plot_pdf=None, test_sample=None, test_label=None, labels=None):
    """Generates a vector field of influence gradient vectors. Plots optionally the golden sample. Vectors are
    optionally coloured, if labels are given"""
    inf_grad_x = gradients[:, 0]
    inf_grad_y = gradients[:, 1]
    train_samples_x = samples[:, 0]
    train_samples_y = samples[:, 1]

    test_sample_color = 'black'

    plt.subplots(figsize=(5, 5))
    Q = plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    Q._init()
    scale = Q.scale
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))

    if labels is None:
        plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    else:
        classes = np.sort(np.unique(labels))
        if len(classes) > 10:
            raise Exception("Not more than 10 distinct classes allowed for gradient plot.")
        # Get default colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # scale = np.max(list(map(np.linalg.norm, gradients)))*15

        for i, c in enumerate(classes):
            class_c = labels == c
            inf_grad_x_c = np.extract(class_c, inf_grad_x)
            inf_grad_y_c = np.extract(class_c, inf_grad_y)
            train_samples_x_c = np.extract(class_c, train_samples_x)
            train_samples_y_c = np.extract(class_c, train_samples_y)
            plt.quiver(train_samples_x_c, train_samples_y_c, inf_grad_x_c, inf_grad_y_c, color=colors[i], scale=scale,
                       label='Class %d' % c)
            if test_label == c:
                test_sample_color = colors[i]

    if test_sample is not None:
        ax.scatter(test_sample[0], test_sample[1], c=test_sample_color, s=200, edgecolors='w', linewidth='3',
                   label=r'$z_{\mathrm{test}}$')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.title(plot_title)
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def plot_gradients_with_likelihoods(spn, marg_spn, gradients, classes, plot_title, plot_pdf, train_samples, res=100,
                                    test_sample=None, test_sample_label=None, true_train_labels=None):
    """Generates a 2D heightmap of marginal likelihoods combined with influence gradients."""
    if len(classes) > 10:
        raise Exception("Not more than 10 distinct classes allowed for gradient with likelihood "
                        "plot, but given %d." % len(classes))

    # Samples
    lin = np.linspace(0.5, 127.5, res)
    y, x = np.meshgrid(lin, lin)
    samples = np.asarray(list(itertools.product(lin, lin)), dtype=np.float32)

    # Predictions and likelihoods
    data = np.column_stack((samples, res ** 2 * [np.nan]))
    mpe_data = mpe(spn, data)
    predictions = mpe_data[:, -1]
    likelihoods = likelihood(marg_spn, samples).reshape((res, res)) * 100000

    # Determine colormap levels
    l_max = np.max(likelihoods)
    l_min = np.min(likelihoods)
    levels = np.linspace(l_min, l_max, 15, endpoint=True)

    # Get default colormap for default colors
    def_cmap = plt.get_cmap("tab10")

    inf_grad_x = gradients[:, 0]
    inf_grad_y = gradients[:, 1]
    train_samples_x = train_samples[:, 0]
    train_samples_y = train_samples[:, 1]

    test_sample_color = 'black'

    # Initialize plot
    plt.subplots(figsize=(5, 5))
    Q = plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    Q._init()
    scale = Q.scale
    plt.close()
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the likelihoods
    for i, c in enumerate(classes):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), def_cmap(i)])
        preds_c = np.array([1 if pred == c else 0 for pred in predictions]).reshape((res, res))
        plt.contourf(x, y, likelihoods * preds_c, levels, cmap=cmap)

    # Plot the vector field
    if true_train_labels is None:
        plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    else:
        classes = np.sort(np.unique(true_train_labels))
        if len(classes) > 10:
            raise Exception("Not more than 10 distinct classes allowed for gradient plot.")
        # Get default colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # scale = np.max(list(map(np.linalg.norm, gradients)))*15

        for i, c in enumerate(classes):
            class_c = true_train_labels == c
            inf_grad_x_c = np.extract(class_c, inf_grad_x)
            inf_grad_y_c = np.extract(class_c, inf_grad_y)
            train_samples_x_c = np.extract(class_c, train_samples_x)
            train_samples_y_c = np.extract(class_c, train_samples_y)
            plt.quiver(train_samples_x_c, train_samples_y_c, inf_grad_x_c, inf_grad_y_c, color=colors[i], scale=scale,
                       label='Class %d' % c)
            if test_sample_label == c:
                test_sample_color = colors[i]

    if test_sample is not None:
        ax.scatter(test_sample[0], test_sample[1], c=test_sample_color, s=200, edgecolors='w', linewidth='3',
                   label=r'$z_{\mathrm{test}}$')

    plt.title(plot_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    ax.set_xlim([0, 128])
    ax.set_ylim([0, 128])

    plot_pdf.savefig(fig)
    plt.show()


def plot_digit(image, res, plot_title, plot_pdf, figsize=1.5, plot_xlabel=None):
    """Plots a single MNIST digit image with given resolution and optional detailed text."""
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(image.reshape(res, res), cmap=plt.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if plot_xlabel is not None:
        plt.xlabel(plot_xlabel)
    plt.title(plot_title)
    plot_pdf.savefig(fig)
    plt.show()


def plot_digits(images, res, plot_title, labels, plot_pdf, lls=None, marg_lls=None, if_vals=None, figsize=4):
    """Generates multiple tiny plots of MNIST digit images with given resolution."""
    n = np.sqrt(images.shape[0])
    assert n % 1 == 0, "Number of digit images must be a square number, but is %d." % images.shape[0]
    fig, axs = plt.subplots(nrows=int(n), ncols=int(n), figsize=(figsize, figsize))
    fig.suptitle(plot_title)
    for i, ax in enumerate(np.array([axs]).flat):
        ax.imshow(images[i].reshape(res, res), cmap=plt.cm.binary)
        if lls is not None and marg_lls is not None:
            ax.set_xlabel("\"%d\", %.2e\n%.2e" % (labels[i], np.exp(lls[i]), np.exp(marg_lls[i])))
        elif if_vals is not None:
            ax.set_xlabel("\"%d\", %.2e" % (labels[i], if_vals[i]))
        else:
            ax.set_xlabel("\"%d\"" % labels[i])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_pdf.savefig(fig)
    plt.show()


def plot_completion(true_image, predicted_image, true_label, predicted_label, res, plot_pdf):
    """Plots two MNIST digit images with given resolution."""
    fig, axs = plt.subplots(figsize=(4, 2), nrows=1, ncols=2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    bounds = np.max(np.abs((true_image, predicted_image)))
    axs[0].imshow(true_image.reshape(res, res), cmap=plt.cm.binary, vmin=0, vmax=bounds)
    axs[1].imshow(predicted_image.reshape(res, res), cmap=plt.cm.binary, vmin=0, vmax=bounds)
    axs[0].set_title('Ground Truth')
    axs[1].set_title('Completed Image')
    axs[0].set_xlabel('True Label: "%d"' % true_label)
    axs[1].set_xlabel('Predicted Label: "%d"' % predicted_label)
    plot_pdf.savefig(fig)
    plt.show()


def plot_heatmaps(intensities, xdim, ydim, plot_title, labels, plot_pdf=None, rescale=False, absolute=False):
    """Generates multiple screened influence heatmaps where the influence value determines the colour intensity."""
    n = np.sqrt(intensities.shape[0])
    assert n % 1 == 0, "Number of intensity maps must be a square number, but is %d." % intensities.shape[0]

    if absolute:
        intensities = np.abs(intensities)
        cmap = plt.cm.binary
    else:
        cmap = "RdBu"

    fig, axs = plt.subplots(nrows=int(n), ncols=int(n), figsize=(5, 5))
    for i, ax in enumerate(np.array([axs]).flat):
        intensity_map = intensities[i].reshape(xdim, ydim)
        bounds = np.max(np.abs(intensity_map), initial=1e-15)
        if rescale:
            intensity_map = intensity_map / bounds  # normalize the data
            # rescale the data with sigmoid for better visualization:
            intensity_map = 2 / (1 + np.exp(-10 * intensity_map)) - 1
            ax.imshow(intensity_map, cmap=cmap, vmin=-1, vmax=1)
        else:
            ax.imshow(intensity_map, cmap=cmap, vmin=-bounds, vmax=bounds)
        ax.set_xlabel("\"%d\"" % labels[i])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(plot_title)
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def plot_influence_components(influences, labels, plot_pdf):
    """Generates three scatter plots, comparing influence values of modified versions
    of IF I_{up,loss}.

    :param influences: Contains four vectors of influence values:
     - influence values with train loss & Hessian
     - influence values without Hessian
     - # influence values without train loss
     - # influence values without train loss & Hessian
    """
    classes = np.sort(np.unique(labels))
    if len(classes) > 10:
        raise Exception("Not more than 10 distinct classes allowed for influence components "
                        "plot, but given %d." % len(classes))

    # Get default colormap for default colors
    def_cmap = plt.get_cmap("tab10")

    # Bounds for plot framing
    # bounds = np.max(np.abs(influences)) * 1.08

    fig, ax = plt.subplots(figsize=(5, 5))

    x = influences[0]
    y = influences[1]

    # Plot diagonal line
    ax.plot([np.min(influences) * 1.08, np.max(influences) * 1.08],
            [np.min(influences) * 1.08, np.max(influences) * 1.08], ls="--", c=".5")
    # ax.set_xlim([-bounds, bounds])
    # ax.set_ylim([-bounds, bounds])

    # Plot scatter per class
    for j, c in enumerate(classes):
        class_c = labels == c
        x_c = np.extract(class_c, x)
        y_c = np.extract(class_c, y)
        ax.scatter(x_c, y_c, c=def_cmap(j))

    ax.set_xlabel(r'$I_{\mathrm{up},\mathrm{loss}}$ w/o Hessian')
    ax.set_ylabel(r'$I_{\mathrm{up},\mathrm{loss}}$')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax[1].set_xlabel(r'$I_{\mathrm{up},\mathrm{loss}}$ w/o Hessian')
    # ax[2].set_xlabel(r'$I_{\mathrm{up},\mathrm{loss}}$ w/o train loss & Hess.')
    plt.title('Effect of Hessian')
    plot_pdf.savefig(fig)
    plt.show()


def plot_toy_color(image, plot_title, figsize=1.5, plot_pdf=None, label=None):
    """Plots a single toy color image."""
    fig = plt.figure(figsize=(figsize, figsize))
    plt.imshow(image.reshape(5, 5), cmap=toy_cmap)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if label is not None:
        plt.xlabel("True label: \"%d\"" % label)
    plt.title(plot_title)
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def plot_toy_colors(images, plot_title, plot_pdf=None, labels=None):
    """Generates multiple tiny plots of MNIST digit images with given resolution."""
    n = np.sqrt(images.shape[0])
    assert n % 1 == 0, "Number of toy color images must be a square number, but is %d." % images.shape[0]
    fig, axs = plt.subplots(nrows=int(n), ncols=int(n), figsize=(5, 5))
    for i, ax in enumerate(np.array([axs]).flat):
        ax.imshow(images[i].reshape(5, 5), cmap=toy_cmap)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if labels is not None:
            ax.set_xlabel("\"%d\"" % labels[i])
    fig.suptitle(plot_title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if plot_pdf is not None:
        plot_pdf.savefig(fig)
    plt.show()


def save_object_to(obj, destination_path):
    """Saves a python object to a specified (relative) destination path."""
    abs_dest_path = os.path.abspath(destination_path)
    f = open(abs_dest_path, 'wb')
    pickle.dump(obj, f)
    f.close()


def load_object_from(source_path):
    """Loads a python object from a specified (relative) source path."""
    abs_source_path = os.path.abspath(source_path)
    f = open(abs_source_path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def convert_spn_to_tf_graph(spn, test_data, batch_size, dtype=None):
    """Converts an SPN into a tf.Graph."""
    print('\033[1mStart SPN conversion into tf.Tensor...\033[0m')
    start_time = time.time()

    # Do the conversion from SPN to tf.Graph
    spn_root, data_placeholder, variable_dict = spn_to_tf_graph(
        spn,
        test_data,
        batch_size=batch_size,
        dtype=dtype
    )

    duration = time.time() - start_time
    print('\033[1mFinished conversion after %.3f sec.\033[0m' % duration)

    return spn_root, data_placeholder, variable_dict


def export_model(root_dir, export_dir="/output/spns/exported_model", force_overwrite=False):
    """Saves the current TF default graph to a specified export directory relative
    to a given root directory."""
    print('\033[1mStart model export...\033[0m')
    start_time = time.time()

    abs_export_dir = os.path.abspath(root_dir + export_dir)

    # Validate that directory does not already exist
    if not force_overwrite and os.path.isdir(abs_export_dir):
        # In case the target directory exists, enumerate directory name
        i = 0
        while os.path.isdir("%s_%s" % (abs_export_dir, i)):
            i += 1
        abs_export_dir = "%s_%s" % (abs_export_dir, i)

    # Create directory
    try:
        os.mkdir(abs_export_dir)
    except OSError:
        print("Creation of the directory %s failed." % abs_export_dir)

    # Take folder name as model name (used for file naming)
    model_name = os.path.basename(os.path.normpath(abs_export_dir))

    # Save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, "%s/%s" % (abs_export_dir, model_name))

    duration = time.time() - start_time
    print('\033[1mFinished model export after %.3f sec.\033[0m' % duration)

    # Return the actually used export directory
    return abs_export_dir


def import_model(export_dir, input_map=None):
    """Loads a tf.Graph from a specified export directory."""
    print('\033[1mStart model import of %s...\033[0m' % os.path.basename(export_dir))
    start_time = time.time()

    # Take folder name as model name (used for file naming)
    model_name = os.path.basename(os.path.normpath(export_dir))
    meta_graph_path = os.path.abspath("%s/%s.meta" % (export_dir, model_name))
    model_path = os.path.abspath("%s/%s" % (export_dir, model_name))

    with tf.Session() as sess:
        if input_map:
            saver = tf.train.import_meta_graph(meta_graph_path, input_map=input_map)
        else:
            saver = tf.train.import_meta_graph(meta_graph_path)
        saver.restore(sess, model_path)
    graph = tf.get_default_graph()

    duration = time.time() - start_time
    print('\033[1mFinished model import after %.3f sec.\033[0m' % duration)

    return graph


def plot_tf_graph(graph_elem, feed_dict, log_dir="../output", sess=None):
    """Exports a log of the current TF default graph to a specified destination.
    Can optionally be used with initialized session. Returns the computed result of run()."""
    if sess is not None:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        result = sess.run(graph_elem, feed_dict=feed_dict)
        print(result)
        writer.close()
    else:
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            result = sess.run(graph_elem, feed_dict=feed_dict)
            print(result)
            writer.close()

    absolute_log_path = os.path.abspath(log_dir)

    print("Open graph visualization with TensorBoard. To do this, execute in terminal:\n"
          "tensorboard --logdir=\"%s\"" % absolute_log_path)

    return result
