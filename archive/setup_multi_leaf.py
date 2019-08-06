if __name__ == '__main__':  # needed to circumvent multiprocessing RuntimeError under Windows 10
    from src.InterpretableSpn import InterpretableSpn
    from src.influence.dataset import DataSet  # for train and test set creation
    from tensorflow.contrib.learn.python.learn.datasets import base
    from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical  # for SPN structure
    from src.help_functions import *

    # Generate dummy data sets (at least 5 instances needed for HVP, why soever...)
    train_samples = np.array([[1.0], [1.5], [0.9], [0.77], [0.66], [1.15]], dtype=np.float32)
    train_labels = np.array([[1], [1], [0], [0], [0], [1]], dtype=np.float32)
    train_set = DataSet(train_samples, train_labels)
    test_samples = np.array([[1.9], [0.4]], dtype=np.float32)
    test_labels = np.array([[1], [0]], dtype=np.float32)
    test_set = DataSet(test_samples, test_labels)
    validation_set = None

    # Collect SPN attributes
    data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)
    model_name = "Multi-leaf SPN"
    batch_size = 1

    # Construct multi-leaf SPN
    multi_leaf_spn = 0.1 * (Gaussian(mean=[0.67], stdev=[0.22], scope=0) *
                            Categorical(p=[0.2, 0.8], scope=1)) \
                     + 0.9 * (Gaussian(mean=[0.39], stdev=[0.99], scope=0) *
                              Categorical(p=[0.2, 0.8], scope=1))

    # Convert this SPN into a tf.Tensor (test_samples needed for shape)
    multi_leaf_spn_tensor, _, _ = convert_spn_to_tf_graph(multi_leaf_spn,
                                                          np.column_stack((test_samples, test_labels)),
                                                          batch_size,
                                                          dtype=np.float32)

    root = tf.identity(multi_leaf_spn_tensor, name="Root")

    # Export the model
    # output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output"
    output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
    spn_export_path = "/spns/converted_multi_leaf_spn"
    import_path = export_model(root_dir=output_path, export_dir=spn_export_path)

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
    restored_spn_graph = import_model(import_path, input_map)
    new_root = restored_spn_graph.get_tensor_by_name("Root:0")

    # Create a graph log to visualize the TF graph with TensorBoard
    plot_tf_graph(new_root,
                  {sample_placeholder: [test_samples[1]],
                   label_placeholder: [test_labels[1]]},
                  log_dir=output_path + "/logs")

    # Initialize single-leaf SPN
    interpretable_spn = InterpretableSpn(root_node=new_root,
                                         input_placeholder=sample_placeholder,
                                         label_placeholder=label_placeholder,
                                         data_sets=data_sets,
                                         num_classes=1,
                                         batch_size=batch_size,
                                         num_epochs=15,
                                         model_name=model_name,
                                         train_dir=output_path + '/training',
                                         mini_batch=False)

    # Experimental influence computations
    influence = interpretable_spn.get_influence_on_test_loss(test_indices=[1],
                                                             train_idx=[0],
                                                             ignore_hessian=False)
    print("Influence on test loss:", influence)

    influence_grad = interpretable_spn.get_grad_of_influence_wrt_input(test_indices=[0], train_indices=[0])
    print("Influence gradient:", influence_grad)
