if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    import matplotlib.backends.backend_pdf

    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)
    from spn.gpu.TensorFlow import spn_to_tf_graph  # conversion into TensorFlow representation
    from src.help_functions import *
    from spn.algorithms.MPE import mpe  # most probable explanation (MPE)

    # ==== Script for MPE generation of a sum-product network (SPN) on MNIST ====

    spn_name = "mnist_spn_2"
    plot_name = "%s_mpe" % spn_name
    output_path = "output"
    plot_path = output_path + "/plots/mnist/" + plot_name

    create_dir(plot_path, force_overwrite=True)
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path + "/" + plot_name + ".pdf")

    res = 28
    spn = load_object_from(output_path + "/spns/" + spn_name + ".pckl")

    for label in range(0, 10):
        evidence = [np.append(res ** 2 * [np.nan], [label])]
        mpe_values = mpe(spn, evidence)
        plot_digit(mpe_values[0][:-1], res, "Label: \"%d\"" % label, pdf)

    pdf.close()
