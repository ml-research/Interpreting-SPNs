if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf

    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *
    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)

    # ==== Performance Evaluation of a sum-product network (SPN) on MNIST ====

    # --- Parameters ---
    spn_name = "mnist_spn_2"
    output_path = "output"
    res = 28
    force_overwrite = True

    # --- Performance Evaluation ----
    # Load model and data
    spn = load_object_from(output_path + "/spns/" + spn_name + ".pckl")
    (train_images, train_labels), (test_images, test_labels) = load_mnist(0, 10000, res=res)

    evaluate_spn_performance(spn, train_images, train_labels, test_images, test_labels, res**2)
