import numpy as np
import tensorflow as tf
import pickle  # for saving python objects
import time  # for timing SPN learning duration
import os  # for reading and writing files and directories

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


def save_object_to(obj, destination_path):
    """Saves a python object to a specified (relative) destination path."""
    f = open(destination_path, 'wb')
    pickle.dump(obj, f)
    f.close()


def load_object_from(source_path):
    """Loads a python object from a specified (relative) source path."""
    f = open(source_path, 'rb')
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


def export_model(export_dir="output/exported_model"):
    """Saves the current TF default graph to a specified export directory."""
    print('\033[1mStart model export...\033[0m')
    start_time = time.time()

    # Validate that directory does not already exist
    if os.path.isdir(export_dir):
        # In case the target directory exists, enumerate directory name
        i = 0
        while os.path.isdir("%s_%s" % (export_dir, i)):
            i += 1
        export_dir = "%s_%s" % (export_dir, i)

    # Create directory
    try:
        os.mkdir(export_dir)
    except OSError:
        print("Creation of the directory %s failed." % export_dir)

    # Take folder name as model name (used for file naming)
    model_name = os.path.basename(os.path.normpath(export_dir))

    # Save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, "%s/%s" % (export_dir, model_name))

    duration = time.time() - start_time
    print('\033[1mFinished model export after %.3f sec.\033[0m' % duration)

    # Return the actually used export directory
    return export_dir


def import_model(export_dir, input_map=None):
    """Loads a tf.Graph from a specified export directory."""
    print('\033[1mStart model import...\033[0m')
    start_time = time.time()

    # Take folder name as model name (used for file naming)
    model_name = os.path.basename(os.path.normpath(export_dir))
    meta_graph_path = "%s/%s.meta" % (export_dir, model_name)
    model_path = "%s/%s" % (export_dir, model_name)

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
