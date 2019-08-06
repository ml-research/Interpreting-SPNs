if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    import scipy.special as sp

    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from spn.algorithms.Statistics import get_structure_stats  # SPN statistics output
    from spn.algorithms.Inference import log_likelihood  # log-likelihood computation
    from src.help_functions import *

    # Get train and test set
    num_train_samples = 10000
    num_test_samples = 10000
    (train_images, train_labels), (test_images, test_labels) = load_mnist(num_train_samples, num_test_samples, normalization=False)
    train_data = np.column_stack((train_images, train_labels))
    test_data = np.column_stack((test_images, test_labels))
    label_idx = 784
    num_classes = 10

    # The SPN to test
    output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output/spns"
    file_name = "mnist_spn_9.pckl"
    spn = load_object_from(output_path + "/" + file_name)

    # Print SPN node statistics
    print(get_structure_stats(spn))

    # ---- Model Performance Evaluation ----

    # Predict train labels
    train_performance_data = np.column_stack((train_images, [np.nan] * num_train_samples))
    train_predictions = mpe(spn, train_performance_data)
    predicted_train_labels = train_predictions[:, 784]

    # Accuracy on train set
    correct_answers = train_labels == predicted_train_labels
    acc = np.count_nonzero(correct_answers) / num_train_samples

    print('\033[1mTrain set performance:\033[0m')
    print("Train sample count:", num_train_samples)
    print("Train set accuracy:", acc * 100, "%")

    print("Prediction distribution:")
    for i in range(10):
        print("    # of occurrence of", i, "in train predictions:", np.count_nonzero(predicted_train_labels == i))

    # Predict test labels
    test_performance_data = np.column_stack((test_images, [np.nan] * num_test_samples))
    test_predictions = mpe(spn, test_performance_data)
    predicted_test_labels = test_predictions[:, 784]

    # Accuracy on test set
    correct_answers = test_labels == predicted_test_labels
    acc = np.count_nonzero(correct_answers) / num_test_samples

    print('\033[1mTest set performance:\033[0m')
    print("Test sample count:", num_test_samples)
    print("Test set accuracy:", acc * 100, "%")

    print("Prediction distribution:")
    for i in range(10):
        print("    # of occurrence of", i, "in test predictions:", np.count_nonzero(predicted_test_labels == i))

    # Some individual predictions:
    logits = []
    for i in range(10):
        logits.append(log_likelihood(spn, np.reshape(np.append(train_images[0], i), [-1, 785])))

    logits = np.array(logits).flatten()
    print(logits)
    print(sp.softmax(logits))
