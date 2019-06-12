if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn.datasets import base
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from spn.io.Graphics import plot_spn  # plot SPN
    from spn.structure.Base import Context  # for SPN learning
    from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier  # for SPN learning
    from spn.structure.leaves.parametric.Parametric import Categorical  # leaf node type
    from spn.structure.leaves.parametric.Parametric import Gaussian  # leaf node type
    from spn.algorithms.Statistics import get_structure_stats  # SPN statistics output
    from spn.algorithms.Inference import likelihood  # log-likelihood inference
    from spn.gpu.TensorFlow import optimize_tf  # optimize SPN parameters
    from spn.algorithms.Marginalization import marginalize  # SPN marginalization
    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from src.help_functions import *

    # ---- Model Setup ----
    spn_name = "two_class_spn"
    output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
    plot_path = output_path + "/plots/three-class/15"

    # Get train and test set
    num_train_samples = 800
    num_test_samples_sqrt = 64
    num_test_samples = num_test_samples_sqrt ** 2
    (train_samples, train_labels), (test_samples, test_labels) = load_three_class(num_train_samples, num_test_samples)
    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    train_data = np.column_stack((train_samples, train_labels))
    test_data = np.column_stack((test_samples, test_labels))

    train_set = DataSet(train_samples, train_labels)
    test_set = DataSet(test_samples, test_labels)
    validation_set = None
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

    # Split classes
    samples_c0 = np.array([train_samples[k] for k in range(0, len(train_samples)) if train_labels[k] == 0])
    samples_c1 = np.array([train_samples[k] for k in range(0, len(train_samples)) if train_labels[k] == 1])
    samples_c2 = np.array([train_samples[k] for k in range(0, len(train_samples)) if train_labels[k] == 2])

    # Plot train samples
    c0_x = samples_c0.transpose()[0].tolist()
    c0_y = samples_c0.transpose()[1].tolist()
    c1_x = samples_c1.transpose()[0].tolist()
    c1_y = samples_c1.transpose()[1].tolist()
    c2_x = samples_c2.transpose()[0].tolist()
    c2_y = samples_c2.transpose()[1].tolist()

    plt.subplots(figsize=(5, 5))
    plt.scatter(c0_x, c0_y, label="Class 0")
    plt.scatter(c1_x, c1_y, label="Class 1")
    plt.scatter(c2_x, c2_y, label="Class 2")
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.legend()
    plt.axis('equal')
    plt.title('Train Data Distribution')
    plt.savefig(plot_path + "/distribution-train.pdf")
    plt.show()

    # Split classes
    samples_c0 = np.array([test_samples[k] for k in range(0, len(test_samples)) if test_labels[k] == 0])
    samples_c1 = np.array([test_samples[k] for k in range(0, len(test_samples)) if test_labels[k] == 1])
    samples_c2 = np.array([test_samples[k] for k in range(0, len(test_samples)) if test_labels[k] == 2])

    # Plot test samples
    c0_x = samples_c0.transpose()[0].tolist()
    c0_y = samples_c0.transpose()[1].tolist()
    c1_x = samples_c1.transpose()[0].tolist()
    c1_y = samples_c1.transpose()[1].tolist()
    c2_x = samples_c2.transpose()[0].tolist()
    c2_y = samples_c2.transpose()[1].tolist()

    plt.subplots(figsize=(5, 5))
    plt.scatter(c0_x, c0_y, label="Class 0", s=10)
    plt.scatter(c1_x, c1_y, label="Class 1", s=10)
    plt.scatter(c2_x, c2_y, label="Class 2", s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.legend()
    plt.axis('equal')
    plt.title('Test Data Distribution')
    plt.savefig(plot_path + "/distribution-test.pdf")
    plt.show()

    # Training parameters
    parametric_types = [Gaussian, Gaussian, Categorical]
    min_instances_slice = 250  # smaller value leads to deeper SPN
    threshold = 0.5  # alpha: the smaller alpha the more product nodes are added

    context = Context(parametric_types=parametric_types).add_domains(train_data)
    label_idx = 2
    batch_size = 1
    num_classes = 3

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
    plt.subplots(figsize=(5, 5))
    plt.scatter(correct_preds[:, 0], correct_preds[:, 1], label="Correctly predicted", c="darkgray", s=10)
    plt.scatter(wrong_preds[:, 0], wrong_preds[:, 1], label="Wrongly predicted", c="darkred", s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.legend()
    plt.axis('equal')
    plt.title('Test Data Prediction')
    plt.savefig(plot_path + "/test-pred.pdf")
    plt.show()

    # Plot decision boundaries
    spn = marginalize(spn, [0, 1])
    likelihoods = likelihood(spn, test_data).reshape((num_test_samples_sqrt, num_test_samples_sqrt)) * 100000
    plot_decision_boundaries(likelihoods, pred_test_labels, num_test_samples_sqrt, plot_path)

    # Convert the model
    spn_tensor, data_placeholder, variable_dict = convert_spn_to_tf_graph(
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

    # Import the model with new placeholders
    sample_placeholder = tf.placeholder(dtype=np.float32,
                                        shape=(batch_size, test_samples.shape[1]),
                                        name="Sample_Placeholder")
    label_placeholder = tf.placeholder(dtype=np.float32,
                                       shape=(batch_size, test_labels.shape[1]),
                                       name="Label_Placeholder")
    input_placeholder = tf.concat([sample_placeholder, label_placeholder], 1)
    input_map = {"Placeholder:0": input_placeholder}
    restored_spn_graph = import_model(output_path + "/spns/tf_" + spn_name, input_map)
    new_root = restored_spn_graph.get_tensor_by_name("Root:0")

    # Try if model runs consistently
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('\033[1mStart bottom-up evaluation...\033[0m')
        start_time = time.time()

        print(sess.run(new_root, feed_dict={"Sample_Placeholder:0": [train_samples[0]],
                                            "Label_Placeholder:0": [train_labels[0]]}))

        duration = time.time() - start_time
        print('\033[1mFinished bottom-up evaluation after %.3f sec.\033[0m' % duration)

    # Create a graph log to visualize the TF graph with TensorBoard
    plot_tf_graph(new_root,
                  {sample_placeholder: [test_samples[1]],
                   label_placeholder: [test_labels[1]]},
                  log_dir=output_path + "/logs")

    # Initialize interpretable SPN
    model_name = "SPN"

    print('\033[1mStart InterpretableSpn class initialization...\033[0m')
    start_time = time.time()

    spn = InterpretableSpn(root_node=new_root,
                           input_placeholder=sample_placeholder,
                           label_placeholder=label_placeholder,
                           data_sets=data_sets,
                           num_classes=num_classes,
                           batch_size=batch_size,
                           num_epochs=15,
                           model_name=model_name,
                           train_dir=output_path + '/training',
                           mini_batch=False)

    duration = time.time() - start_time
    print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

    # ---- Influence Inspection ----
    t = np.random.randint(0, len(test_samples))  # Index of test sample which is used for inference computation
    single_test_sample = test_samples[t]  # The test sample under inference investigation
    n = len(train_samples)  # Number of train samples to be investigated (not more than 1900)

    # Split classes
    samples_c0 = np.array([train_samples[k] for k in range(0, n) if train_labels[k] == 0])
    samples_c1 = np.array([train_samples[k] for k in range(0, n) if train_labels[k] == 1])
    samples_c2 = np.array([train_samples[k] for k in range(0, n) if train_labels[k] == 2])

    # Plot train samples
    c0_x = samples_c0.transpose()[0].tolist()
    c0_y = samples_c0.transpose()[1].tolist()
    c1_x = samples_c1.transpose()[0].tolist()
    c1_y = samples_c1.transpose()[1].tolist()
    c2_x = samples_c2.transpose()[0].tolist()
    c2_y = samples_c2.transpose()[1].tolist()

    # 1. Influences on test sample no. t w/o Hessian
    influences = spn.get_influence_on_test_loss(test_indices=[t],
                                                train_idx=range(0, n),
                                                ignore_hessian=True)

    plot_influences(influences=influences,
                    samples=train_samples,
                    plot_title='Influence of Each Training Sample \n on a Single Test Sample w/o Hessian',
                    plot_path=plot_path,
                    plot_file_name="influence-no-hessian.pdf",
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
                    plot_path=plot_path,
                    plot_file_name="influence-hessian.pdf",
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
                    plot_path=plot_path,
                    plot_file_name="influence-grad-norms-hessian.pdf",
                    test_sample=single_test_sample)

    # Vector field of gradients
    inf_grad_x = influence_grad[:, 0]
    inf_grad_y = influence_grad[:, 1]
    train_samples_x = train_samples[:, 0]
    train_samples_y = train_samples[:, 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.quiver(train_samples_x, train_samples_y, inf_grad_x, inf_grad_y)
    ax.scatter(single_test_sample[0], single_test_sample[1], c='black', s=60)
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title('Influence Gradients of Each Training Sample \n Regarding a Single Test Sample w/ Hessian')
    plt.savefig(plot_path + "/influence-grads-hessian.pdf")
    plt.show()

    # 4. Loss gradients (no influence)
    grads = spn.get_grad_loss_wrt_input(range(0, len(test_samples)))
    test_samples_x = test_samples[:, 0]
    test_samples_y = test_samples[:, 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.quiver(test_samples_x, test_samples_y, grads[:, 0], grads[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title('Loss Gradients of Each Test Sample \n Regarding its Coordinates')
    plt.savefig(plot_path + "/loss-grads.pdf")
    plt.show()

    '''
    # 5. Summed Influences on all test samples w/ Hessian
    influences = spn.get_influence_on_test_loss(test_indices=[0],
                                                train_idx=range(0, n),
                                                ignore_hessian=False)
    for i in range(1, num_test_samples):
        influences += spn.get_influence_on_test_loss(test_indices=[i],
                                                     train_idx=range(0, n),
                                                     ignore_hessian=False)

    influences_c0 = np.array([influences[k] for k in range(0, n) if train_labels[k] == 0])
    influences_c1 = np.array([influences[k] for k in range(0, n) if train_labels[k] == 1])
    influences_c2 = np.array([influences[k] for k in range(0, n) if train_labels[k] == 2])

    fig, ax = plt.subplots(figsize=(8, 5))
    sc1 = ax.scatter(c0_x, c0_y, c=influences_c0, cmap="winter")
    sc2 = ax.scatter(c1_x, c1_y, c=influences_c1, cmap="autumn")
    sc3 = ax.scatter(c2_x, c2_y, c=influences_c2, cmap="cool")
    ax.scatter(single_test_sample[0], single_test_sample[1], c='darkgray', s=200)
    plt.xlabel('x')
    plt.ylabel('y')
    axes = plt.gca()
    axes.set_xlim([0, 128])
    axes.set_ylim([0, 128])
    plt.axis('equal')
    plt.title('Summed Influence of Each Training Sample \n on All Test Samples w/ Hessian')
    fig.colorbar(sc1, ax=ax)
    fig.colorbar(sc2, ax=ax)
    fig.colorbar(sc3, ax=ax)
    plt.savefig(plot_path + "/influence-summed-hessian.pdf")
    plt.show()
    '''
