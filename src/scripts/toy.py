if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import os
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets import base
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from spn.structure.Base import Context  # for SPN learning
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier  # for SPN learning
    from spn.structure.leaves.parametric.Parametric import Categorical  # leaf node type
    from spn.structure.leaves.parametric.Parametric import Gaussian  # leaf node type
    from spn.algorithms.Statistics import get_structure_stats  # SPN statistics output
    from spn.algorithms.Marginalization import marginalize  # SPN marginalization
    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from src.help_functions import *
    from spn.algorithms.Inference import log_likelihood  # log-likelihood computation

    # ==== Influence inspection of a sum-product network (SPN) on the toy color data set ====

    # ---- Parameters ----
    # Dataset parameters
    num_train_samples = 1000  # Train sample count
    num_test_samples_sqrt = 100  # Root of test sample count
    seed = 23081996  # Random seed

    # Parameters for influence investigation
    t = 4  # Index of a test sample (the golden sample) which is used for inference computation and validity checks
    t_label = None  # If a value is given, the test sample is set to this label, set to "None" for original label
    ignore_weights = False  # When true, weight parameters from sum nodes are ignored for influence computation
    ignore_means = False  # When true, mean parameters from Gaussian nodes are ignored for influence computation
    ignore_variances = False  # When true, stdev parameters from Gaussian nodes are ignored for influence computation
    type_of_loss = "conditional_ll"  # The used likelihood for the loss. Select from: "joint_ll", "conditional_ll"

    # Paths and names
    spn_name = "toy_spn_6"
    cached_tf_spn_name = None  # if a string is given, the string is used as the name of converted SPN to be loaded
    cached_spn_name = None  # if a string is given, the string is used as the name of an SPN to be loaded
    plot_name = "%s_%d_%s" % (spn_name, t, type_of_loss)
    output_path = "output"
    plot_path = output_path + "/plots/toy/" + plot_name
    force_overwrite = True  # Force the overwrite of old plots at plot location
    save_spn = True  # if True, SPN is saved with Pickle after training under its SPN name

    # SPN learning parameters
    min_instances_slice = 200  # smaller value leads to deeper SPN
    threshold = 0.3  # alpha: the smaller alpha the more product nodes are added

    # HVP (LiSSA) approximation parameters
    scale = 20
    damping = 0.6  # select in interval [0, 1)
    recursion_depth = 10

    # Miscellaneous
    log_tf_graph = False
    test_bottom_up_eval = True
    compute_lls = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPUs to be used (-1 for no GPUs)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    num_threads = 16

    # ---- Initialization & Model Setup ----
    create_dir(plot_path, force_overwrite=force_overwrite)
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path + "/" + plot_name + ".pdf")

    # Get train and test set
    np.random.seed(seed=seed)
    num_test_samples = num_test_samples_sqrt ** 2
    (train_samples, train_labels), (test_samples, test_labels) = generate_toy_color(num_train_samples,
                                                                                    num_test_samples)

    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    train_data = np.column_stack((train_samples, train_labels))
    test_data = np.column_stack((test_samples, test_labels))

    train_set = DataSet(train_samples, train_labels)
    test_set = DataSet(test_samples, test_labels)
    validation_set = None
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

    if t_label is not None:
        # Redefine the label of the test sample
        test_labels[t] = t_label
        test_data[t][-1] = t_label

    batch_size = 1
    res = 5
    label_idx = res ** 2

    if cached_tf_spn_name is None:
        if cached_spn_name is None:
            # Training parameters
            parametric_types = [Gaussian] * res ** 2 + [Categorical]
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
                                   cpus=num_threads)

            duration = time.time() - start_time
            print('\033[1mFinished training after %.3f sec.\033[0m' % duration)

            if save_spn:
                save_object_to(spn, output_path + "/spns/" + spn_name + ".pckl")
        else:
            # Model loading
            print('\033[1mStart SPN loading...\033[0m')
            start_time = time.time()

            spn = load_object_from(output_path + "/spns/" + cached_spn_name + ".pckl")

            duration = time.time() - start_time
            print('\033[1mFinished loading after %.3f sec.\033[0m' % duration)

        # Model performance evaluation
        spn_stats = get_structure_stats(spn)
        print(spn_stats)
        stats_file = open(plot_path + "/spn_stats.txt", "w+")
        stats_file.write(spn_stats)
        correct_test_preds, pred_test_labels = evaluate_spn_performance(spn, train_samples, train_labels, test_samples,
                                                                        test_labels, label_idx, stats_file)

        # Save metadata into stats file
        metadata = "\nSeed: %d\n" % seed + \
                   "Test sample ID: %d\n" % t + \
                   "Minimum instances per slice: %d\n" % min_instances_slice + \
                   "Alpha (threshold): %f\n" % threshold + \
                   "Type of loss: %s\n" % type_of_loss + \
                   "Weights ignored: %r\n" % ignore_weights + \
                   "Means ignored: %r\n" % ignore_means + \
                   "Variances ignored: %r\n" % ignore_variances + \
                   "Lissa parameters:\n" + \
                   "   - Scale: %f\n" % scale + \
                   "   - Damping: %.1e\n" % damping + \
                   "   - Recursion depth: %d\n" % recursion_depth
        stats_file.write(metadata)

        # Marginalize SPN
        spn_marg = marginalize(spn, list(range(res ** 2)))

        # SPN sanity check
        ll = log_likelihood(spn, np.array([test_data[t]]))
        ll_marg = log_likelihood(spn_marg, np.array([test_data[t]]))
        print("Let t be train sample no. %d, which is:" % t)
        print(test_data[t])
        print("Log-likelihood of t:", ll)
        print("Likelihood of t:", np.exp(ll))
        print("Marginal log-likelihood of t:", ll_marg)
        print("Marginal likelihood of t:", np.exp(ll_marg))

        # Convert the model
        spn_tensor, _, _ = convert_spn_to_tf_graph(
            spn,
            test_data,
            batch_size=batch_size,
            dtype=np.float32
        )

        # Export the model
        root = tf.identity(spn_tensor, name="Root")
        export_dir = export_model(root_dir=output_path, export_dir="/spns/tf_" + spn_name, force_overwrite=True)
        print("Successfully exported SPN tensor to \"%s\"." % export_dir)

        tf.reset_default_graph()

        # Convert the marginalized model
        spn_marg_tensor, _, _ = convert_spn_to_tf_graph(
            spn_marg,
            test_data,
            batch_size=batch_size,
            dtype=np.float32
        )

        # Export the marginalized model
        root_marg = tf.identity(spn_marg_tensor, name="Root")
        export_dir = export_model(root_dir=output_path, export_dir="/spns/tf_" + spn_name + "_marg",
                                  force_overwrite=True)
        print("Successfully exported marginal SPN tensor to \"%s\"." % export_dir)

        tf.reset_default_graph()

        spn_name_to_be_imported = spn_name
    else:
        stats_file = open(plot_path + "/spn_stats.txt", "a")
        spn_name_to_be_imported = cached_tf_spn_name

    # Import the models with new placeholders
    sample_placeholder = tf.placeholder(dtype=np.float32,
                                        shape=(batch_size, test_samples.shape[1]),
                                        name="Sample_Placeholder")
    label_placeholder = tf.placeholder(dtype=np.float32,
                                       shape=(batch_size, test_labels.shape[1]),
                                       name="Label_Placeholder")
    input_placeholder = tf.concat([sample_placeholder, label_placeholder], 1)
    input_marg = tf.concat([sample_placeholder, [[np.nan]]], 1)

    with tf.name_scope("SPN"):
        input_map = {"Placeholder:0": input_placeholder}
        restored_spn_graph = import_model(output_path + "/spns/tf_" + spn_name_to_be_imported, input_map)
        new_root = restored_spn_graph.get_tensor_by_name("SPN/Root:0")

    with tf.name_scope("SPN_Marg"):
        input_map_marg = {"Placeholder:0": input_marg}
        restored_marg_spn_graph = import_model(output_path + "/spns/tf_" + spn_name_to_be_imported + "_marg",
                                               input_map_marg)
        new_root_marg = restored_marg_spn_graph.get_tensor_by_name("SPN_Marg/Root:0")

    results = [np.nan, np.nan]

    if test_bottom_up_eval:
        # Try if models runs consistently
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            print('\033[1mStart bottom-up evaluation...\033[0m')
            start_time = time.time()

            results = sess.run([new_root, new_root_marg], feed_dict={"Sample_Placeholder:0": [test_samples[t]],
                                                                     "Label_Placeholder:0": [test_labels[t]]})
            print("Log-likelihood:", results[0])
            print("Likelihood:", np.exp(results[0]))
            print("Marginal log-likelihood:", results[1])
            print("Marginal likelihood:", np.exp(results[1]))

            duration = time.time() - start_time
            print('\033[1mFinished bottom-up evaluation after %.3f sec.\033[0m' % duration)

    if log_tf_graph:
        # Create a graph log to visualize the TF graph with TensorBoard
        plot_tf_graph([new_root, new_root_marg],
                      {sample_placeholder: [test_samples[t]],
                       label_placeholder: [test_labels[t]]},
                      log_dir=output_path + "/logs")

    # Initialize interpretable SPN
    model_name = "SPN"

    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    spn = InterpretableSpn(root_node=new_root,
                           root_node_marg=new_root_marg,
                           sample_placeholder=sample_placeholder,
                           label_placeholder=label_placeholder,
                           data_sets=data_sets,
                           model_name=model_name,
                           train_dir=output_path + '/training',
                           ignore_weights=ignore_weights,
                           ignore_means=ignore_means,
                           ignore_variances=ignore_variances,
                           loss=type_of_loss)

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    # ---- Influence Inspection ----
    # Plot test sample
    plot_toy_color(image=test_samples[t],
                   plot_title="Regarded Test Sample",
                   plot_pdf=pdf,
                   label=test_labels[t])

    # Plot train samples
    plot_toy_colors(images=train_samples[0:16],
                    plot_title="Train Samples",
                    plot_pdf=pdf,
                    labels=train_labels[0:16])

    # 1.a Influence gradients of image pixels without Hessian

    # Get IF gradients
    influence_grads = spn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                          train_indices=range(16),
                                                          ignore_hessian=True,
                                                          output_file=stats_file)

    # Plot influence gradient heatmaps
    plot_heatmaps(intensities=influence_grads,
                  xdim=res, ydim=res,
                  plot_title="Feature Influences w/o Hessian",
                  labels=train_labels[0:16],
                  plot_pdf=pdf,
                  absolute=True)

    # 1.b Influence gradients of image pixels with Hessian

    print('\033[1mStart influence gradient computation with Hessian...\033[0m')
    start_time = time.time()

    # Get IF gradients
    influence_grads = spn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                          train_indices=range(16),
                                                          ignore_hessian=False,
                                                          approx_type='lissa',
                                                          approx_params={"batch_size": batch_size,
                                                                         "scale": scale,
                                                                         "damping": damping,
                                                                         "num_samples": 1,
                                                                         "recursion_depth": recursion_depth},
                                                          output_file=stats_file)

    duration = time.time() - start_time
    print('\033[1mFinished influence gradient computation after %.3f sec.\033[0m' % duration)

    # Plot influence gradient heatmaps
    plot_heatmaps(intensities=influence_grads,
                  xdim=res, ydim=res,
                  plot_title="Feature Influences w/ Hessian",
                  labels=train_labels[0:16],
                  plot_pdf=pdf,
                  absolute=True)

    '''
    # 2. Influence values of all train images
    print('\033[1mStart influence value computation with Hessian...\033[0m')
    start_time = time.time()

    influences = spn.get_influence_on_test_loss(test_indices=[t],
                                                train_idx=range(0, n),
                                                ignore_hessian=False,
                                                force_refresh=False,
                                                approx_type='lissa')

    duration = time.time() - start_time
    print('\033[1mFinished influence value computation after %.3f sec.\033[0m' % duration)

    # Sort in ascending order
    sorted_indices_asc = np.argsort(influences)
    plot_digits(images=train_samples[sorted_indices_asc[0:16]],
                res=res,
                plot_title="Samples with Lowest Influence Values",
                labels=train_labels[sorted_indices_asc[0:16]],
                plot_pdf=pdf,
                if_vals=influences[sorted_indices_asc[0:16]])

    # Sort in descending order
    sorted_indices_desc = sorted_indices_asc[::-1]
    plot_digits(images=train_samples[sorted_indices_desc[0:16]],
                res=res,
                plot_title="Samples with Highest Influence Values",
                labels=train_labels[sorted_indices_desc[0:16]],
                plot_pdf=pdf,
                if_vals=influences[sorted_indices_desc[0:16]])

    # Sort closest to zero
    closest_to_zero = np.argsort(np.abs(influences))
    plot_digits(images=train_samples[closest_to_zero[0:16]],
                res=res,
                plot_title="Samples with Smallest Absolute Influence Values",
                labels=train_labels[closest_to_zero[0:16]],
                plot_pdf=pdf,
                if_vals=influences[closest_to_zero[0:16]])'''

    stats_file.close()
    pdf.close()
