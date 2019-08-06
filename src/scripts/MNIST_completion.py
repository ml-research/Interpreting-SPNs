if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    import matplotlib.backends.backend_pdf

    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *
    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)

    # ==== Demonstration of the use of a sum-product network (SPN) for image completion ====

    # --- Parameters ---
    spn_name = "mnist_spn_2"
    plot_name = "%s_completion" % spn_name
    # output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
    output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output"
    plot_path = output_path + "/plots/mnist/" + plot_name
    res = 28
    force_overwrite = True

    # --- Completion ----
    # Directory and file initialization
    create_dir(plot_path, force_overwrite=force_overwrite)
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path + "/" + plot_name + ".pdf")

    # Load model and data
    print("Loading SPN...")
    spn = load_object_from(output_path + "/spns/" + spn_name + ".pckl")
    print("Finished loading SPN.")
    (_, _), (test_images, test_labels) = load_mnist(0, 40, res=res)

    # Make MPE and plots for first 10 test images
    for i in range(0, 40):
        print("Doing MPE for test image %d..." % i)
        test_image = test_images[i]
        test_label = test_labels[i]

        label = np.nan
        upper_half = int(res ** 2 * 0.5)
        lower_half = res * res - upper_half
        evidence = [np.concatenate((test_image[0:upper_half], lower_half * [np.nan], [label]))]

        mpe_values = mpe(spn, evidence)
        predicted_label = mpe_values[0][-1]
        predicted_image = mpe_values[0][:-1]

        plot_completion(test_image, predicted_image, test_label, predicted_label, res, pdf)

    pdf.close()
