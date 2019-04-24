if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets import base
    import matplotlib.pyplot as plt

    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from src.help_functions import *
    from prettytable import PrettyTable

    # Get train and test set
    num_train_samples = 10000
    num_test_samples = 10000
    (train_images, train_labels), (test_images, test_labels) = load_mnist(num_train_samples, num_test_samples,
                                                                          normalization=False)
    train_data = np.column_stack((train_images, train_labels))
    test_data = np.column_stack((test_images, test_labels))
    label_idx = 784
    num_classes = 10



    # Load a saved, trained SPN
    spn = load_object_from("./output/mnist_spn_5.pckl")

    # Convert the trained SPN into a tf.Tensor (test_images needed for shape)
    spn_tensor, data_placeholder, variable_dict = convert_spn_to_tf_graph(
        spn,
        test_data,
        batch_size=1,
        dtype=np.float32
    )

    root = tf.identity(spn_tensor, name="Root")



    # Load a saved, trained and converted SPN
    # export_dir = "output/tf_mnist_spn_1"
    # spn_graph = import_model(export_dir)
    # root = spn_graph.get_tensor_by_name("Root:0")
    # data_placeholder = spn_graph.get_tensor_by_name("Placeholder:0")

    # with tf.Session() as sess:
    #     print(sess.run(root, feed_dict={"Placeholder": np.append(train_data[0], 5)}))

    # ---- Influence Inspection ----

    # Convert datasets into Influence DataSet objects
    train_set = DataSet(train_images, train_labels)
    test_set = DataSet(test_images, test_labels)

    validation_set = None

    # Collect SPN attributes
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)
    model_name = "SPN"
    input_dim = 3
    batch_size = 1

    # Initialize interpretable MNIST SPN
    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    spn = InterpretableSpn(root_node=root,
                           input_placeholder=data_placeholder,
                           data_sets=data_sets,
                           input_dim=input_dim,
                           num_classes=num_classes,
                           label_idx=label_idx,
                           batch_size=batch_size,
                           num_epochs=15,
                           model_name=model_name,
                           train_dir='output')

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    influence = spn.get_influence_on_test_loss(test_indices=[0],
                                               train_idx=[0],
                                               ignore_hessian=True)
    print("Influence on test loss:", influence)

    train_idx = 0
    print("Influence of train sample %s (with label %s) on test images (without Hessian):" % (
        train_idx, train_labels[train_idx]))
    influences = PrettyTable(['Index', 'Label value', 'Influence'])
    for i in range(20):
        influence = spn.get_influence_on_test_loss(test_indices=[i],
                                                   train_idx=[train_idx],
                                                   ignore_hessian=True)
        influences.add_row([i, test_labels[i], influence])
    print(influences)

    train_idx = 22
    print("Influence of train sample %s (with label %s) on test images (without Hessian):" % (
        train_idx, train_labels[train_idx]))
    influences = PrettyTable(['Index', 'Label value', 'Influence'])
    for i in range(20):
        influence = spn.get_influence_on_test_loss(test_indices=[i],
                                                   train_idx=[train_idx],
                                                   ignore_hessian=True)
        influences.add_row([i, test_labels[i], influence])
    print(influences)

    test_idx = 0
    influence_grad = spn.get_grad_of_influence_wrt_input(test_indices=[test_idx], train_indices=[0])

    plt.figure()
    i = np.random.randint(0, num_train_samples)
    plt.imshow(influence_grad[-1:], cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()
