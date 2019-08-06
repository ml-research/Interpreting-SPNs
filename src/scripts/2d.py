if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import os
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets import base
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from spn.io.Graphics import plot_spn  # plot SPN
    from spn.structure.Base import Context  # for SPN learning
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier  # for SPN learning
    from spn.structure.leaves.parametric.Parametric import Categorical  # leaf node type
    from spn.structure.leaves.parametric.Parametric import Gaussian  # leaf node type
    from spn.algorithms.Statistics import get_structure_stats  # SPN statistics output
    from spn.algorithms.Marginalization import marginalize  # SPN marginalization
    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from src.help_functions import *
    from spn.algorithms.Inference import likelihood  # likelihood inference
    from spn.algorithms.Inference import log_likelihood  # log-likelihood computation

    # ==== Influence inspection of a sum-product network (SPN) on 2D data sets ====

    # ---- Parameters ----
    # Dataset parameters
    dataset_loader = generate_linear  # The synthetic classification problem to be generated
    dataset_name = "linear"
    num_train_samples = 200  # Train sample count (max. 60,000)
    num_test_samples = 1000  # Test sample count (max. 10,000)
    noisy_dataset = True  # When True, noisy data sets are generated
    seed = 23081996  # Random seed for reproduction of results

    # Parameters for influence investigation
    t = 596  # Index of the test sample which is used for inference computation and for validity checks
    t_features = None  # If a value is given, the features are set to this value, set to "None" for original features
    t_label = None  # If a value is given, the label is set to this value, set to "None" for original label
    ignore_weights = False  # When true, weight parameters from sum nodes are ignored for influence computation
    ignore_means = False  # When true, mean parameters from Gaussian nodes are ignored for influence computation
    ignore_variances = False  # When true, stdev parameters from Gaussian nodes are ignored for influence computation
    type_of_loss = "conditional_ll"  # The used likelihood for the loss. Select "joint_ll" or "conditional_ll"
    n = num_train_samples  # Number of train samples to be investigated (max. num_train_samples)

    # Paths and names
    spn_name = "2d_spn"
    # output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
    output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output"
    plot_name = "%s_%d_%s_test" % (spn_name, t, type_of_loss)
    plot_path = output_path + "/plots/%s/%s" % (dataset_name, plot_name)
    force_overwrite = True  # Force the overwrite of old plots at plot location
    save_spn = True  # If True, SPN is saved with Pickle after training under its SPN name

    # SPN learning parameters
    min_instances_slice = 200  # Smaller value leads to deeper SPN (default 200)
    threshold = 0.3  # Smaller value leads to more product nodes (default 0.3)

    # HVP (LiSSA) approximation parameters
    scale = 20
    damping = 0.85  # Select in interval [0, 1)
    recursion_depth = 5

    # Miscellaneous
    plot_res = 300  # Resolution of the decision boundary and likelihood plots
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPUs to be used (-1 for no GPUs)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    num_threads = 32  # Number threads used for SPN learning

    # ---- Initialization & Model Setup ----
    create_dir(plot_path, force_overwrite=force_overwrite)
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path + "/" + plot_name + ".pdf")

    # Get train and test set
    np.random.seed(seed=seed)
    (train_samples, train_labels), (test_samples, test_labels) = dataset_loader(num_train_samples,
                                                                                num_test_samples,
                                                                                noise=noisy_dataset)
    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    train_data = np.column_stack((train_samples, train_labels))
    test_data = np.column_stack((test_samples, test_labels))

    if t_label is not None:
        # Redefine the label of the test sample
        test_labels[t] = t_label
        test_data[t][-1] = t_label
    if t_features is not None:
        # Redefine the feature values of the test sample
        test_samples[t] = t_features
        test_data[t] = np.concatenate((t_features, test_labels[t]))

    train_set = DataSet(train_samples, train_labels)
    test_set = DataSet(test_samples, test_labels)
    validation_set = None
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

    # Plot train samples
    plot_samples(train_samples, train_labels, plot_pdf=pdf,
                 plot_title='Train Dataset')

    # Plot train samples with test sample
    plot_samples(train_samples, train_labels, plot_pdf=pdf,
                 plot_title='Train Dataset', test_sample=test_samples[t])

    # Plot test samples
    plot_samples(test_samples, test_labels, plot_pdf=pdf,
                 plot_title='Test Dataset',
                 size=10)

    parametric_types = [Gaussian, Gaussian, Categorical]
    context = Context(parametric_types=parametric_types).add_domains(train_data)
    label_idx = 2
    batch_size = 1

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

    # Model performance evaluation
    spn_stats = get_structure_stats(spn)
    print(spn_stats, end="")
    stats_file = open(plot_path + "/spn_stats.txt", "w+")
    stats_file.write(spn_stats)
    plot_spn(spn, plot_path + "/spn_struct.pdf")
    (predicted_train_labels, correct_train_preds), (predicted_test_labels, correct_test_preds) = \
        evaluate_spn_performance(spn, train_samples, train_labels, test_samples,
                                 test_labels, label_idx, stats_file)

    # Save metadata into stats file
    metadata = "\nSeed: %d\n" % seed + \
               "Test sample ID: %d\n" % t + \
               "Noisy dataset: %r\n" % noisy_dataset + \
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

    # Plot train predictions
    plot_samples(train_samples, correct_train_preds, plot_pdf=pdf,
                 plot_title='Train Data Prediction',
                 classes=[False, True],
                 size=10,
                 colors=["darkred", "darkgray"],
                 plot_labels=["Wrongly predicted", "Correctly predicted"])

    # Plot test predictions
    plot_samples(test_samples, correct_test_preds, plot_pdf=pdf,
                 plot_title='Test Data Prediction',
                 classes=[False, True],
                 size=10,
                 colors=["darkred", "darkgray"],
                 plot_labels=["Wrongly predicted", "Correctly predicted"])

    # Plot likelihoods
    plot_likelihoods(spn=spn,
                     classes=np.sort(np.unique(test_labels)),
                     res=plot_res,
                     plot_pdf=pdf)

    # Plot likelihoods with test sample
    plot_likelihoods(spn=spn,
                     classes=np.sort(np.unique(test_labels)),
                     res=plot_res,
                     plot_pdf=pdf,
                     test_sample=test_samples[t])

    # Marginalize SPN
    spn_marg = marginalize(spn, [0, 1])

    # Plot decision boundaries
    plot_decision_boundaries(spn=spn,
                             marg_spn=spn_marg,
                             classes=np.sort(np.unique(test_labels)),
                             res=plot_res,
                             plot_pdf=pdf)

    # Plot decision boundaries with test sample
    plot_decision_boundaries(spn=spn,
                             marg_spn=spn_marg,
                             classes=np.sort(np.unique(test_labels)),
                             res=plot_res,
                             plot_pdf=pdf,
                             test_sample=test_samples[t])

    # SPN sanity check
    ll = log_likelihood(spn, np.array([test_data[t]]))
    ll_marg = log_likelihood(spn_marg, np.array([test_data[t]]))
    print("Let t be the Golden Sample with ID %d, which is:" % t)
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
    export_dir = export_model(root_dir=output_path, export_dir="/spns/tf_" + spn_name + "_marg", force_overwrite=True)
    print("Successfully exported marginal SPN tensor to \"%s\"." % export_dir)

    tf.reset_default_graph()

    # Import the models with new placeholders
    sample_placeholder = tf.placeholder(dtype=np.float32,
                                        shape=(batch_size, test_samples.shape[1]),
                                        name="Sample_Placeholder")
    label_placeholder = tf.placeholder(dtype=np.float32,
                                       shape=(batch_size, test_labels.shape[1]),
                                       name="Label_Placeholder")
    input_placeholder = tf.concat([sample_placeholder, label_placeholder], 1)
    input_marg = tf.concat([sample_placeholder, [[np.nan]]], 1)

    # Import the models with new placeholders
    with tf.name_scope("SPN"):
        input_map = {"Placeholder:0": input_placeholder}
        restored_spn_graph = import_model(output_path + "/spns/tf_" + spn_name, input_map)
        new_root = restored_spn_graph.get_tensor_by_name("SPN/Root:0")

    with tf.name_scope("SPN_Marg"):
        input_map_marg = {"Placeholder:0": input_marg}
        restored_marg_spn_graph = import_model(output_path + "/spns/tf_" + spn_name + "_marg", input_map_marg)
        new_root_marg = restored_marg_spn_graph.get_tensor_by_name("SPN_Marg/Root:0")

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

    # Initialize interpretable SPN
    model_name = "SPN"

    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    interpretable_spn = InterpretableSpn(root_node=new_root,
                                         root_node_marg=new_root_marg,
                                         sample_placeholder=sample_placeholder,
                                         label_placeholder=label_placeholder,
                                         data_sets=data_sets,
                                         model_name=model_name,
                                         train_dir=output_path + '/training',
                                         ignore_weights=ignore_weights,
                                         ignore_means=ignore_means,
                                         ignore_variances=ignore_variances,
                                         type_of_loss=type_of_loss)

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    # Loss sanity check
    print('\033[1mStart bottom-up evaluation...\033[0m')
    start_time = time.time()

    # Create a graph log to visualize the TF graph with TensorBoard
    result = plot_tf_graph(graph_elem=interpretable_spn.total_loss,
                           feed_dict={"Sample_Placeholder:0": [test_samples[t]],
                                      "Label_Placeholder:0": [test_labels[t]]},
                           log_dir=output_path + "/logs",
                           sess=interpretable_spn.sess)

    print("Loss of golden sample:", result)

    duration = time.time() - start_time
    print('\033[1mFinished bottom-up evaluation after %.3f sec.\033[0m' % duration)

    # Loss gradients
    grads = interpretable_spn.get_grad_loss_wrt_input(range(0, len(test_samples)))

    plot_gradients(gradients=grads,
                   samples=test_samples,
                   plot_title='Loss Gradients of all Test Samples\nRegarding their Coordinates',
                   plot_pdf=pdf,
                   labels=test_labels)

    # ---- Influence Inspection ----
    # 1. Influences on Golden Sample w/o Hessian
    influences_no_hess = interpretable_spn.get_influence_on_test_loss(test_indices=[t],
                                                                      train_idx=range(0, n),
                                                                      ignore_hessian=True)

    plot_influences(influences=influences_no_hess,
                    samples=train_samples,
                    plot_title='Influence Values w/o Hessian',
                    plot_pdf=pdf,
                    test_sample=test_samples[t])

    plot_influences_with_multiple_colors(influences=influences_no_hess,
                                         samples=train_samples,
                                         labels=train_labels,
                                         plot_title='Influence Values w/o Hessian',
                                         plot_pdf=pdf,
                                         test_sample=test_samples[t])

    # 2. Influences on Golden Sample w/ Hessian
    influences_hess = interpretable_spn.get_influence_on_test_loss(test_indices=[t],
                                                                   train_idx=range(0, n),
                                                                   ignore_hessian=False,
                                                                   force_refresh=True,
                                                                   approx_type='lissa',
                                                                   approx_params={"batch_size": batch_size,
                                                                                  "scale": scale,
                                                                                  "damping": damping,
                                                                                  "num_samples": 1,
                                                                                  "recursion_depth": recursion_depth},
                                                                   output_file=stats_file)

    plot_influences(influences=influences_hess,
                    samples=train_samples,
                    plot_title='Influence Values w/ Hessian',
                    plot_pdf=pdf,
                    test_sample=test_samples[t])

    plot_influences_with_multiple_colors(influences=influences_hess,
                                         samples=train_samples,
                                         labels=train_labels,
                                         plot_title='Influence Values w/ Hessian',
                                         plot_pdf=pdf,
                                         test_sample=test_samples[t])

    # 3. Influence Gradients regarding Golden Sample w/o Hessian
    influence_grads = interpretable_spn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                        train_indices=range(0, n),
                                                                        ignore_hessian=True)

    # Vector field of gradients
    plot_gradients(gradients=influence_grads,
                   samples=train_samples,
                   test_sample=test_samples[t],
                   test_label=test_labels[t],
                   labels=train_labels,
                   plot_title='Feature Influences w/o Hessian',
                   plot_pdf=pdf)

    plot_gradients_with_likelihoods(spn=spn,
                                    marg_spn=spn_marg,
                                    gradients=influence_grads,
                                    classes=np.sort(np.unique(test_labels)),
                                    train_samples=train_samples,
                                    test_sample=test_samples[t],
                                    test_sample_label=test_labels[t],
                                    true_train_labels=train_labels,
                                    res=plot_res,
                                    plot_title='Feature Influences and Likelihoods,\n'
                                               'w/o Hessian',
                                    plot_pdf=pdf)

    # 4. Influence Gradients regarding Golden Sample w/ Hessian
    influence_grads = interpretable_spn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                        train_indices=range(0, n),
                                                                        ignore_hessian=False,
                                                                        force_refresh=False,
                                                                        approx_type='lissa')
    '''influence_norms = [np.linalg.norm(s) for s in influence_grads]

    plot_influences(influences=influence_norms,
                    samples=train_samples,
                    plot_title='Influence Gradients Norms of Each Training Sample\non a Single Test Sample w/ Hessian',
                    plot_pdf=pdf,
                    test_sample=golden_sample)'''

    # Vector field of gradients
    plot_gradients(gradients=influence_grads,
                   samples=train_samples,
                   test_sample=test_samples[t],
                   test_label=test_labels[t],
                   labels=train_labels,
                   plot_title='Feature Influences w/ Hessian',
                   plot_pdf=pdf)

    plot_gradients_with_likelihoods(spn=spn,
                                    marg_spn=spn_marg,
                                    gradients=influence_grads,
                                    classes=np.sort(np.unique(test_labels)),
                                    train_samples=train_samples,
                                    test_sample=test_samples[t],
                                    test_sample_label=test_labels[t],
                                    true_train_labels=train_labels,
                                    res=plot_res,
                                    plot_title='Feature Influences and Likelihoods,\n'
                                               'w/ Hessian',
                                    plot_pdf=pdf)

    # 6. Components of influence
    influences = [influences_no_hess, influences_hess]
    plot_influence_components(influences, labels=train_labels, plot_pdf=pdf)

    # 7. Influence gradients of samples far away from decision boundary
    condition = (train_samples[:, 0] < 25) | (105 < train_samples[:, 0])
    selected_influence_grads = np.extract(np.transpose([condition, condition]), influence_grads).reshape((-1, 2))
    selected_samples = np.extract(np.transpose([condition, condition]), train_samples).reshape((-1, 2))
    selected_labels = np.extract(condition, train_labels)

    plot_gradients(gradients=selected_influence_grads,
                   samples=selected_samples,
                   test_sample=test_samples[t],
                   test_label=test_labels[t],
                   labels=selected_labels,
                   plot_title='Feature Influences w/ Hessian\nfor Selected Samples',
                   plot_pdf=pdf)

    '''# 8. Summed influences on all test samples w/ Hessian
    influences = np.zeros(num_train_samples)
    for i in range(0, num_test_samples):
        influences += spn.get_influence_on_test_loss(test_indices=[i],
                                                     train_idx=range(0, n),
                                                     ignore_hessian=False,
                                                     force_refresh=False)

    plot_influences(influences=influences,
                    samples=train_samples,
                    plot_title='Summed Influence of All Train Samples\nRegarding All Test Samples w/ Hessian',
                    plot_pdf=pdf)

    # 6. Summed influence gradients on all test samples w/ Hessian
    grads = np.zeros((num_train_samples, 2))
    for i in range(0, num_test_samples):
        grads += spn.get_grad_of_influence_wrt_input(test_indices=[i],
                                                     train_indices=range(0, n),
                                                          force_refresh=False)

    plot_gradients(gradients=grads,
                   samples=train_samples,
                   plot_title='Summed Influence Gradients of All Train Samples\n'
                              'Regarding All Test Samples w/ Hessian',
                   plot_pdf=pdf)'''

    stats_file.close()
    pdf.close()
