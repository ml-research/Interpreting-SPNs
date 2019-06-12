import numpy as np
import tensorflow as tf
import pickle  # for saving python objects
import time  # for timing SPN learning duration
import os  # for reading and writing files and directories
import itertools  # for combining
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation


def load_mnist(num_train_samples=60000, num_test_samples=10000, normalization=True):
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

    # Reshape data
    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    return (train_images, train_labels), (test_images, test_labels)


def load_linear_two_class(num_train_samples=1000, num_test_samples=128 * 128):
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
            if sample[0] < 64:
                labels[k] = np.append(labels[k], [0])
            else:
                labels[k] = np.append(labels[k], [1])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def load_linear_two_class_noise(num_train_samples=1000, num_test_samples=128 * 128):
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
                if sample[0] + x > 70 and k == 0:
                    labels[k] = np.append(labels[k], [1])
                else:
                    labels[k] = np.append(labels[k], [0])
            else:
                if sample[0] - x < 58 and k == 0:
                    labels[k] = np.append(labels[k], [0])
                else:
                    labels[k] = np.append(labels[k], [1])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def load_two_class(num_train_samples=1000, num_test_samples=128 * 128):
    """Generates a 2D two-class problem dataset and returns it."""
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
            if 16 <= sample[0] <= 112 and 16 <= sample[1] <= 112:
                if sample[0] < 40 and sample[1] < 40:
                    if np.sqrt((40 - sample[0]) ** 2 + (40 - sample[1]) ** 2) <= 24:
                        labels[k] = np.append(labels[k], [1])
                    else:
                        labels[k] = np.append(labels[k], [0])
                elif sample[0] > 88 and sample[1] < 40:
                    if np.sqrt((88 - sample[0]) ** 2 + (40 - sample[1]) ** 2) <= 24:
                        labels[k] = np.append(labels[k], [1])
                    else:
                        labels[k] = np.append(labels[k], [0])
                elif sample[0] > 88 and sample[1] > 88:
                    if np.sqrt((88 - sample[0]) ** 2 + (88 - sample[1]) ** 2) <= 24:
                        labels[k] = np.append(labels[k], [1])
                    else:
                        labels[k] = np.append(labels[k], [0])
                elif sample[0] < 40 and sample[1] > 88:
                    if np.sqrt((40 - sample[0]) ** 2 + (88 - sample[1]) ** 2) <= 24:
                        labels[k] = np.append(labels[k], [1])
                    else:
                        labels[k] = np.append(labels[k], [0])
                else:
                    labels[k] = np.append(labels[k], [1])
            else:
                labels[k] = np.append(labels[k], [0])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


def load_three_class(num_train_samples=1000, num_test_samples=128 * 128):
    """Generates a 2D three-class problem dataset and returns it."""
    # Generate the dataset
    # Initialize two 2D fields with num_train_samples and num_test_samples resp.
    train_samples = np.random.uniform(0.0, 128.0, (num_train_samples, 2))
    num_test_samples = np.sqrt(num_test_samples)
    test_samples = list(itertools.product(np.linspace(0.5, 127.5, num_test_samples),
                                          np.linspace(0.5, 127.5, num_test_samples)))

    # compute train and test labels
    labels = [[], []]

    for k, samples in enumerate((train_samples, test_samples)):
        for i in range(0, len(samples)):
            sample = samples[i]
            if sample[1] >= np.cos(sample[0] / 128 * np.pi) * 50 + 78:
                labels[k] = np.append(labels[k], [0])
            else:
                if sample[1] >= (sample[0] - 30) * 2:
                    labels[k] = np.append(labels[k], [1])
                else:
                    labels[k] = np.append(labels[k], [2])

    # Convert data type
    train_samples = np.asarray(train_samples, dtype=np.float32)
    train_labels = np.asarray(labels[0], dtype=np.float32)
    test_samples = np.asarray(test_samples, dtype=np.float32)
    test_labels = np.asarray(labels[1], dtype=np.float32)

    return (train_samples, train_labels), (test_samples, test_labels)


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
    correct_answers = np.reshape(train_labels, -1) == predicted_train_labels
    acc = np.count_nonzero(correct_answers) / num_train_samples

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
    correct_answers = np.reshape(test_labels, -1) == predicted_test_labels
    acc = np.count_nonzero(correct_answers) / num_test_samples

    test_text = "\n\nTest set performance:" \
                "\nTest sample count: %d" \
                "\nTest set accuracy: %.2f %%" % \
                (num_test_samples, acc * 100)
    print(test_text)
    if stats_file is not None:
        stats_file.write(test_text)

    return correct_answers, predicted_test_labels


def plot_decision_boundaries(likelihoods, pred_test_labels, num_test_samples_sqrt, plot_path):
    classes = list(set(pred_test_labels))
    if len(classes) > 10:
        raise Exception("Not more than 10 distinct classes allowed for decision boundary plot.")

    # Test sample coordinates
    lin = np.linspace(0.5, 127.5, num_test_samples_sqrt)
    y, x = np.meshgrid(lin, lin)

    # Determine colormap levels
    l_max = np.max(likelihoods)
    l_min = np.min(likelihoods)
    levels = np.linspace(l_min, l_max, 15, endpoint=True)

    # Get default colormap for default colors
    def_cmap = plt.get_cmap("tab10")

    conts = []

    # Plot the decision boundaries
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(classes):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 1, 1, 0), def_cmap(i)])
        preds = np.array([1 if pred_test_labels[k] == c else 0 for k in range(0, len(pred_test_labels))]).reshape(
            (num_test_samples_sqrt, num_test_samples_sqrt))
        conts.append(plt.contourf(x, y, likelihoods * preds, levels, cmap=cmap))
        plt.contour(x, y, preds, [0.5], colors="k")

    plt.axis('equal')
    plt.title('Decision Boundaries \n With Summed Likelihood')
    divider = make_axes_locatable(ax)

    for i, c in enumerate(classes):
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        if i == len(classes) - 1:
            fig.colorbar(conts[i], cax=cax)
        else:
            fig.colorbar(conts[i], cax=cax, ticks=[])

    plt.savefig(plot_path + "/dec-bound.pdf")
    plt.show()


def plot_influences(influences, samples, test_sample, plot_title, plot_path, plot_file_name):
    range = np.max(np.abs(influences))
    x = samples.transpose()[0].tolist()
    y = samples.transpose()[1].tolist()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=influences, cmap="RdBu", vmin=-range, vmax=range)
    if test_sample is not None:
        ax.scatter(test_sample[0], test_sample[1], c='gold', s=200, edgecolors='w', linewidth='3')
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title(plot_title)
    fig.colorbar(sc, ax=ax)
    plt.savefig(plot_path + "/" + plot_file_name)
    plt.show()


def plot_gradients(gradients, samples, test_sample, plot_title, plot_path, plot_file_name):
    inf_grad_x = gradients[:, 0]
    inf_grad_y = gradients[:, 1]
    train_samples_x = samples[:, 0]
    train_samples_y = samples[:, 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    ax.scatter(test_sample[0], test_sample[1], c='black', s=60)
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title(plot_title)
    plt.savefig(plot_path + "/" + plot_file_name)
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
    print('\033[1mStart model import...\033[0m')
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


def plot_tf_graph(graph, feed_dict, log_dir="../output"):
    """Exports a log of the current TF default graph to a specified destination."""
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        print(sess.run(graph, feed_dict=feed_dict))
        writer.close()

    absolute_log_path = os.path.abspath(log_dir)

    print("Open graph visualization with TensorBoard. To do this, execute in terminal:\n"
          "tensorboard --logdir=\"%s\"" % absolute_log_path)
