if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import scipy.special as sp
    from tensorflow.contrib.learn.python.learn.datasets import base

    from spn.structure.Base import Context  # for SPN learning
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier  # for SPN learning
    from spn.structure.leaves.parametric.Parametric import Categorical  # leaf node type
    from spn.structure.leaves.parametric.Parametric import Gaussian  # leaf node type
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *
    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
    from spn.algorithms.Statistics import get_structure_stats  # SPN statistics output
    from spn.algorithms.Inference import log_likelihood  # log-likelihood computation
    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from prettytable import PrettyTable

    # ---- Data Preprocessing ----

    # Get train and test set
    num_train_samples = 1000
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
    min_instances_slice = 250  # smaller value leads to deeper SPN
    threshold = 0.5  # alpha: the smaller alpha the more product nodes are added

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

    # Save model
    output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output/spns"
    spn_name = "mnist_spn_test"
    save_object_to(spn, output_path + "/" + spn_name + ".pckl")

    # The SPN to test
    spn = load_object_from(output_path + "/" + spn_name + ".pckl")

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

    # Convert the SPN
    # Load a saved, trained SPN
    spn = load_object_from(output_path + "/" + spn_name + ".pckl")
    batch_size = 1

    # Convert the trained SPN into a tf.Tensor (test_images needed for shape)
    spn_tensor, data_placeholder, variable_dict = convert_spn_to_tf_graph(
        spn,
        test_data,
        batch_size=batch_size,
        dtype=np.float32
    )

    ''''Fails with exit code -1073740791 (0xC0000409)
    # Optimize the converted SPN regarding the likelihood
    print('\033[1mStart SPN optimization...\033[0m')
    start_time = time.time()

    epochs = 200
    optimize_tf_graph(spn_tensor, variable_dict, data_placeholder, test_data, epochs=epochs)

    duration = time.time() - start_time
    print('\033[1mFinished optimization after %.3f sec.\033[0m' % duration)'''

    # Export the converted and optimized model
    root = tf.identity(spn_tensor, name="Root")
    export_dir = export_model(root_dir=output_path, export_dir="/tf_" + spn_name)
    print("Successfully exported SPN tensor to \"%s\"." % export_dir)

    tf.reset_default_graph()

    # Import a trained, saved and converted model with new placeholders
    sample_placeholder = tf.placeholder(dtype=np.float32,
                                        shape=(batch_size, test_images.shape[1]),
                                        name="Sample_Placeholder")
    label_placeholder = tf.placeholder(dtype=np.float32,
                                       shape=(batch_size, 1),
                                       name="Label_Placeholder")
    input_placeholder = tf.concat([sample_placeholder, label_placeholder], 1)
    input_map = {"Placeholder:0": input_placeholder}
    restored_spn_graph = import_model(output_path + "/tf_" + spn_name, input_map)
    new_root = restored_spn_graph.get_tensor_by_name("Root:0")

    # Test it
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('\033[1mStart bottom-up evaluation...\033[0m')
        start_time = time.time()

        print(sess.run(new_root, feed_dict={"Sample_Placeholder:0": [train_images[0]],
                                            "Label_Placeholder:0": [[train_labels[0]]]}))

        duration = time.time() - start_time
        print('\033[1mFinished bottom-up evaluation after %.3f sec.\033[0m' % duration)

    # ---- Influence Inspection ----

    # Convert datasets into Influence DataSet objects
    train_set = DataSet(train_images, np.expand_dims(train_labels, 1))
    test_set = DataSet(test_images, np.expand_dims(test_labels, 1))

    validation_set = None

    # Collect SPN attributes
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)
    model_name = "SPN"
    num_classes = 10

    # Initialize interpretable MNIST SPN
    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    spn = InterpretableSpn(root_node=new_root,
                           input_placeholder=sample_placeholder,
                           label_placeholder=label_placeholder,
                           data_sets=data_sets,
                           num_classes=num_classes,
                           label_idx=label_idx,
                           batch_size=batch_size,
                           num_epochs=15,
                           model_name=model_name,
                           train_dir=output_path + '/training')

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    influence = spn.get_influence_on_test_loss(test_indices=[0],
                                               train_idx=[0],
                                               ignore_hessian=True)
    print("Influence of train sample 0 on test loss of test sample 0 (without Hessian):", influence)

    influence = spn.get_influence_on_test_loss(test_indices=[0],
                                               train_idx=[0],
                                               ignore_hessian=False)
    print("Influence of train sample 0 on test loss of test sample 0 (with Hessian):", influence)

    test_idx = 0
    influence_grad = spn.get_grad_of_influence_wrt_input(test_indices=[test_idx], train_indices=[0])
