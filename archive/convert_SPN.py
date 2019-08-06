if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf

    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *
    from spn.io.Graphics import plot_spn  # plot SPN

    # Get train and test set
    num_train_samples = 10000
    num_test_samples = 10000
    (train_images, train_labels), (test_images, test_labels) = load_mnist(num_train_samples, num_test_samples,
                                                                          normalization=False)
    test_data = np.column_stack((test_images, test_labels))
    batch_size = 1

    output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output/spns"
    file_name = "mnist_spn_9"

    # Load a saved, trained SPN
    spn = load_object_from(output_path + "/" + file_name + ".pckl")

    # plot_spn(spn, "plot")

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
    export_dir = export_model(root_dir=output_path, export_dir="/tf_" + file_name)
    print("Successfully exported SPN tensor to \"%s\"." % export_dir)
