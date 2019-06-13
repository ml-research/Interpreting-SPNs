if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
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
    from spn.algorithms.Inference import likelihood  # log-likelihood inference
    from spn.gpu.TensorFlow import optimize_tf  # optimize SPN parameters
    from spn.algorithms.Inference import log_likelihood  # log-likelihood computation

    # ---- Model Setup ----
    spn_name = "two_class_spn"
    output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
    plot_path = output_path + "/plots/two-class-linear-noise/11"
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path + "/plots.pdf")

    # Get train and test set
    num_train_samples = 200
    num_test_samples_sqrt = 64
    num_test_samples = num_test_samples_sqrt ** 2
    (train_samples, train_labels), (test_samples, test_labels) = load_linear_two_class_noise(num_train_samples,
                                                                                             num_test_samples)
    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    train_data = np.column_stack((train_samples, train_labels))
    test_data = np.column_stack((test_samples, test_labels))

    train_set = DataSet(train_samples, train_labels)
    test_set = DataSet(test_samples, test_labels)
    validation_set = None
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

    # Plot train samples
    plot_samples(train_samples, train_labels, pdf,
                 plot_title='Train Data Distribution')

    # Plot test samples
    plot_samples(test_samples, test_labels, pdf,
                 plot_title='Test Data Distribution',
                 size=10)

    # Training parameters
    parametric_types = [Gaussian, Gaussian, Categorical]
    min_instances_slice = 250  # smaller value leads to deeper SPN
    threshold = 0.5  # alpha: the smaller alpha the more product nodes are added

    context = Context(parametric_types=parametric_types).add_domains(train_data)
    label_idx = 2
    batch_size = 1
    num_classes = 2

    # Model training
    print('\033[1mStart SPN training...\033[0m')
    start_time = time.time()

    spn = learn_classifier(data=train_data,
                           ds_context=context,
                           spn_learn_wrapper=learn_parametric,
                           label_idx=label_idx,
                           min_instances_slice=min_instances_slice,
                           threshold=threshold)

    spn = optimize_tf(spn, train_data)

    duration = time.time() - start_time
    print('\033[1mFinished training after %.3f sec.\033[0m' % duration)

    # Model performance evaluation
    spn_stats = get_structure_stats(spn)
    print(spn_stats)
    stats_file = open(plot_path + "/spn_stats.txt", "w+")
    stats_file.write(spn_stats)
    plot_spn(spn, plot_path + "/spn_struct.pdf")
    correct_test_preds, pred_test_labels = evaluate_spn_performance(spn, train_samples, train_labels, test_samples,
                                                                    test_labels, label_idx, stats_file)
    stats_file.close()

    correct_preds = np.array([test_samples[k] for k in range(0, len(test_samples)) if correct_test_preds[k] != 0])
    wrong_preds = np.array([test_samples[k] for k in range(0, len(test_samples)) if correct_test_preds[k] == 0])

    # Plot predictions
    plot_samples(test_samples, correct_test_preds, pdf,
                 plot_title='Test Data Prediction',
                 size=10,
                 colors=["darkred", "darkgray"],
                 plot_labels=["Wrongly predicted", "Correctly predicted"])

    # Plot decision boundaries
    spn_marg = marginalize(spn, [0, 1])
    likelihoods = likelihood(spn_marg, test_data).reshape((num_test_samples_sqrt, num_test_samples_sqrt)) * 100000
    plot_decision_boundaries(likelihoods,
                             pred_test_labels,
                             num_test_samples_sqrt,
                             pdf)

    # Log-likelihood validation test
    ll = log_likelihood(spn, np.array([train_data[20]]))
    ll_marg = log_likelihood(spn_marg, np.array([train_data[20]]))
    print("Let t be train sample no. 20:", train_data[20])
    print("Log-likelihood of t under SPN:", ll)
    print("Likelihood of t under SPN:", np.exp(ll))
    print("Log-likelihood of t under marginal SPN:", ll_marg)
    print("Likelihood of t under marginal SPN:", np.exp(ll_marg))

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
    print("Successfully exported SPN tensor to \"%s\"." % export_dir)

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

        results = sess.run([new_root, new_root_marg], feed_dict={"Sample_Placeholder:0": [train_samples[20]],
                                                                 "Label_Placeholder:0": [train_labels[20]]})
        print("Log-likelihood of SPN:", results[0])
        print("Likelihood of SPN:", np.exp(results[0]))
        print("Log-likelihood of marginalized SPN:", results[1])
        print("Likelihood of marginalized SPN:", np.exp(results[1]))

        duration = time.time() - start_time
        print('\033[1mFinished bottom-up evaluation after %.3f sec.\033[0m' % duration)

    # Create a graph log to visualize the TF graph with TensorBoard
    plot_tf_graph([new_root, new_root_marg],
                  {sample_placeholder: [test_samples[1]],
                   label_placeholder: [test_labels[1]]},
                  log_dir=output_path + "/logs")

    # Initialize interpretable SPN
    model_name = "SPN"

    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    spn = InterpretableSpn(root_node=new_root,
                           root_node_marg=new_root_marg,
                           input_placeholder=sample_placeholder,
                           label_placeholder=label_placeholder,
                           data_sets=data_sets,
                           num_classes=num_classes,
                           model_name=model_name,
                           train_dir=output_path + '/training',
                           mini_batch=False)

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    # ---- Influence Inspection ----
    t = 2000  # Index of test sample which is used for inference computation
    # t = np.random.randint(0, len(test_samples))  # Index of test sample which is used for inference computation
    single_test_sample = test_samples[t]  # The test sample under inference investigation
    n = len(train_samples)  # Number of train samples to be investigated (not more than 1900)

    # Split classes
    samples_c0 = np.array([train_samples[k] for k in range(0, n) if train_labels[k] == 0])
    samples_c1 = np.array([train_samples[k] for k in range(0, n) if train_labels[k] == 1])

    # Plot train samples
    c0_x = samples_c0.transpose()[0].tolist()
    c0_y = samples_c0.transpose()[1].tolist()
    c1_x = samples_c1.transpose()[0].tolist()
    c1_y = samples_c1.transpose()[1].tolist()

    # 1. Influences on test sample no. t w/o Hessian
    influences = spn.get_influence_on_test_loss(test_indices=[t],
                                                train_idx=range(0, n),
                                                ignore_hessian=True)

    plot_influences(influences=influences,
                    samples=train_samples,
                    plot_title='Influence of Each Training Sample \n on a Single Test Sample w/o Hessian',
                    plot_pdf=pdf,
                    test_sample=single_test_sample)

    # 2. Influences on test sample no. t w/ Hessian
    influences = spn.get_influence_on_test_loss(test_indices=[t],
                                                train_idx=range(0, n),
                                                ignore_hessian=False,
                                                approx_type='lissa',
                                                approx_params={"batch_size": batch_size,
                                                               "scale": 10,
                                                               "damping": 0.01,
                                                               "num_samples": 1,
                                                               "recursion_depth": 10000})

    plot_influences(influences=influences,
                    samples=train_samples,
                    plot_title='Influence of Each Training Sample \n on a Single Test Sample w/ Hessian',
                    plot_pdf=pdf,
                    test_sample=single_test_sample)

    # 3. Influence Gradients regarding test sample no. t w/ Hessian
    influence_grad = spn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                         train_indices=range(0, n),
                                                         approx_type='lissa',
                                                         approx_params={"batch_size": batch_size,
                                                                        "scale": 10,
                                                                        "damping": 0.01,
                                                                        "num_samples": 1,
                                                                        "recursion_depth": 10000})
    influence_norms = [np.linalg.norm(s) for s in influence_grad]

    plot_influences(influences=influence_norms,
                    samples=train_samples,
                    plot_title='Influence Gradient Norms of Each Training Sample \n on a Single Test Sample w/ Hessian',
                    plot_pdf=pdf,
                    test_sample=single_test_sample)

    # Vector field of gradients
    plot_gradients(gradients=influence_grad,
                   samples=train_samples,
                   test_sample=single_test_sample,
                   plot_title='Influence Gradient of Training Samples \n Regarding a Single Test Sample w/ Hessian',
                   plot_pdf=pdf)

    # 4. Loss gradients (no influence)
    grads = spn.get_grad_loss_wrt_input(range(0, len(test_samples)))

    plot_gradients(gradients=grads,
                   samples=test_samples,
                   plot_title='Loss Gradient of Test Samples \n Regarding their Coordinates',
                   plot_pdf=pdf)

    # 5. Summed Influences on all test samples w/ Hessian
    influences = spn.get_influence_on_test_loss(test_indices=[0],
                                                train_idx=range(0, n),
                                                ignore_hessian=False,
                                                approx_type='lissa',
                                                approx_params={"batch_size": batch_size,
                                                               "scale": 10,
                                                               "damping": 0.01,
                                                               "num_samples": 1,
                                                               "recursion_depth": 10000})
    for i in range(1, num_test_samples):
        influences += spn.get_influence_on_test_loss(test_indices=[i],
                                                     train_idx=range(0, n),
                                                     ignore_hessian=False,
                                                     approx_type='lissa',
                                                     approx_params={"batch_size": batch_size,
                                                                    "scale": 10,
                                                                    "damping": 0.01,
                                                                    "num_samples": 1,
                                                                    "recursion_depth": 10000})

    plot_influences(influences=influences,
                    samples=test_samples,
                    plot_title='Summed Influence of Each Training Sample \n on All Test Samples w/ Hessian',
                    plot_pdf=pdf,
                    test_sample=single_test_sample)

    pdf.close()
