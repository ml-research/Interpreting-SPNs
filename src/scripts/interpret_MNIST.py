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
    num_train_samples = 5000
    num_test_samples = 10000
    (train_images, train_labels), (test_images, test_labels) = load_mnist(num_train_samples, num_test_samples,
                                                                          normalization=False)

    train_set = DataSet(train_images, np.expand_dims(train_labels, 1))
    test_set = DataSet(test_images, np.expand_dims(test_labels, 1))
    validation_set = None
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

    label_idx = 784
    num_classes = 10
    batch_size = 1

    output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output/spns"
    file_name = "tf_mnist_spn_9"

    # Import a trained, saved and converted model with new placeholders
    sample_placeholder = tf.placeholder(dtype=np.float32,
                                        shape=(batch_size, test_images.shape[1]),
                                        name="Sample_Placeholder")
    label_placeholder = tf.placeholder(dtype=np.float32,
                                       shape=(batch_size, 1),
                                       name="Label_Placeholder")
    input_placeholder = tf.concat([sample_placeholder, label_placeholder], 1)
    input_map = {"Placeholder:0": input_placeholder}
    restored_spn_graph = import_model(output_path + "/" + file_name, input_map)
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

    # Initialize interpretable MNIST SPN
    model_name = "SPN"

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

    train_idx = 23
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
