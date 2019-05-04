if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    from spn.structure.Base import Context  # for SPN learning
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier  # for SPN learning
    from spn.structure.leaves.parametric.Parametric import Categorical  # leaf node type
    from spn.structure.leaves.parametric.Parametric import Gaussian  # leaf node type
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *

    # ---- Data Preprocessing ----

    # Get train and test set
    num_train_samples = 2000
    num_test_samples = 1000
    (train_images, train_labels), (test_images, test_labels) = load_mnist(num_train_samples, num_test_samples, normalization=False)
    train_data = np.column_stack((train_images, train_labels))
    test_data = np.column_stack((test_images, test_labels))
    label_idx = 784

    # Plot a random train image
    plt.figure()
    i = np.random.randint(0, num_train_samples)
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # ---- Model Learning ----

    # Training parameters
    parametric_types = [Gaussian] * 784 + [Categorical]
    min_instances_slice = 200  # smaller value leads to deeper SPN
    threshold = 0.4  # alpha: the smaller alpha the more product nodes are added

    context = Context(parametric_types=parametric_types).add_domains(train_data)

    # Model training
    print('\033[1mStart SPN training...\033[0m')
    start_time = time.time()

    spn = learn_classifier(data=train_data,
                           ds_context=context,
                           spn_learn_wrapper=learn_parametric,
                           label_idx=label_idx,
                           min_instances_slice=min_instances_slice,
                           threshold=threshold,
                           cpus=12)

    duration = time.time() - start_time
    print('\033[1mFinished training after %.3f sec.\033[0m' % duration)
    save_object_to(spn, "/tmp/Projects/Interpreting-SPNs/output/spns/mnist_spn.pckl")
