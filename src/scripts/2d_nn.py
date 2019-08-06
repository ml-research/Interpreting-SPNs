from keras import backend as K
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow import set_random_seed
from src.help_functions import *
import matplotlib.pyplot as plt
from src.InterpretableNn import InterpretableNn
from src.influence.dataset import DataSet  # for train and test set creation
import numpy as np

# ==== Influence inspection of a classic neural net (NN) ====

n_train = 1000
n_test = 500
n_val = 100

seed = 112000  # Random seed
model_name = "nn"
output_path = "C:/Users/markr/Google Drive/[00] UNI/[00] Informatik/BA/Interpreting SPNs/output"
# output_path = "/home/ml-mrothermel/projects/Interpreting-SPNs/output"

t = 0
t_features = [40, 90]
t_label = 0
n = n_train - n_val

# HVP (LiSSA) approximation parameters
scale = 10
damping = 0.8  # select in interval [0, 1)
recursion_depth = 50

np.random.seed(seed)
set_random_seed(seed)
res = 5

(train_samples, train_labels), (test_samples, test_labels) = generate_linear(n_train, n_test)

# Split train data into train and validation set
val_samples = train_samples[:n_val]
train_samples = train_samples[n_val:]

val_labels = train_labels[:n_val]
train_labels = train_labels[n_val:]

if t_label is not None:
    # Redefine the label of the test sample
    test_labels[t] = t_label
if t_features is not None:
    # Redefine the feature values of the test sample
    test_samples[t] = t_features

# Adjust the shape
train_labels = np.expand_dims(train_labels, 1)
test_labels = np.expand_dims(test_labels, 1)
val_labels = np.expand_dims(val_labels, 1)

train_set = DataSet(train_samples, train_labels)
test_set = DataSet(test_samples, test_labels)
validation_set = DataSet(val_samples, val_labels)
data_sets = base.Datasets(train=train_set, test=test_set, validation=validation_set)

# Plot train samples
plot_samples(train_samples, train_labels,
             plot_title='Train Dataset')

# Plot train samples with test sample
plot_samples(train_samples, train_labels,
             plot_title='Train Dataset', test_sample=test_samples[t])

# Plot test samples
plot_samples(test_samples, test_labels,
             plot_title='Test Dataset',
             size=10)

# Setup the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(2,), name="Input_Layer"))
# model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="Output_Layer"))

model.summary()

# Model configuration
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Learn the model
history = model.fit(train_samples,
                    train_labels,
                    epochs=250,
                    batch_size=10,
                    validation_data=(val_samples, val_labels),
                    verbose=1)
results = model.evaluate(test_samples, test_labels)
print("Test loss:", results[0])
print("Test accuracy:", results[1])

# Get training history
history_dict = history.history

# plot history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_predictions = model.predict_classes(train_samples)
test_predictions = model.predict_classes(test_samples)
correct_train_preds = train_predictions == train_labels
correct_test_preds = test_predictions == test_labels

# Plot train predictions
plot_samples(train_samples, correct_train_preds,
             plot_title='Train Data Prediction',
             classes=[False, True],
             size=10,
             colors=["darkred", "darkgray"],
             plot_labels=["Wrongly predicted", "Correctly predicted"])

# Plot test predictions
plot_samples(test_samples, correct_test_preds,
             plot_title='Test Data Prediction',
             classes=[False, True],
             size=10,
             colors=["darkred", "darkgray"],
             plot_labels=["Wrongly predicted", "Correctly predicted"])

graph = K.get_session().graph
graph.as_default()

# Export the model
export_dir = export_model(root_dir=output_path, export_dir="/nns/tf_" + model_name, force_overwrite=True)
print("Successfully exported SPN tensor to \"%s\"." % export_dir)

tf.reset_default_graph()

# Import the models with new placeholders
sample_placeholder = tf.placeholder(dtype=np.float32,
                                    shape=(1, test_samples.shape[1]),
                                    name="Sample_Placeholder")
label_placeholder = tf.placeholder(dtype=np.float32,
                                   shape=(1, test_labels.shape[1]),
                                   name="Label_Placeholder")

with tf.name_scope("Neural_Net"):
    input_map = {"Input_Layer:0": sample_placeholder,
                 "Output_Layer_target:0": label_placeholder}
    restored_nn_graph = import_model(output_path + "/nns/tf_" + model_name, input_map)
    output_node = restored_nn_graph.get_tensor_by_name("Neural_Net/Output_Layer/Sigmoid:0")
    loss = restored_nn_graph.get_tensor_by_name("Neural_Net/loss/Output_Layer_loss/value:0")

# Create a graph log to visualize the TF graph with TensorBoard
plot_tf_graph(output_node,
              {sample_placeholder: [test_samples[t]],
               label_placeholder: [test_labels[t]]},
              log_dir=output_path + "/logs")

print('\033[1mStart InterpretableNn class initialization...\033[0m')
start_time = time.time()

interpretable_nn = InterpretableNn(output_node=output_node,
                                   sample_placeholder=sample_placeholder,
                                   label_placeholder=label_placeholder,
                                   loss_op=loss,
                                   data_sets=data_sets,
                                   model_name=model_name,
                                   train_dir=output_path + '/training')

duration = time.time() - start_time
print('\033[1mFinished initialization after %.3f sec.\033[0m' % duration)

# 1. Influences on Golden Sample w/o Hessian
influences_no_hess = interpretable_nn.get_influence_on_test_loss(test_indices=[t],
                                                                 train_idx=range(0, n),
                                                                 ignore_hessian=True)

plot_influences(influences=influences_no_hess,
                samples=train_samples,
                plot_title='Influence Values w/o Hessian',
                test_sample=test_samples[t])

# 2. Influences on Golden Sample w/ Hessian
influences_hess = interpretable_nn.get_influence_on_test_loss(test_indices=[t],
                                                              train_idx=range(0, n),
                                                              ignore_hessian=False,
                                                              force_refresh=True,
                                                              approx_type='lissa',
                                                              approx_params={"batch_size": 1,
                                                                             "scale": scale,
                                                                             "damping": damping,
                                                                             "num_samples": 1,
                                                                             "recursion_depth": recursion_depth})

plot_influences(influences=influences_hess,
                samples=train_samples,
                plot_title='Influence Values w/ Hessian',
                test_sample=test_samples[t])

# Get IF gradients
influence_grads = interpretable_nn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                   train_indices=range(n),
                                                                   ignore_hessian=True)

plot_gradients(gradients=influence_grads,
               samples=train_samples,
               test_sample=test_samples[t],
               test_label=test_labels[t],
               labels=train_labels,
               plot_title='Feature Influences w/ Hessian')

# Get IF gradients
influence_grads = interpretable_nn.get_grad_of_influence_wrt_input(test_indices=[t],
                                                                   train_indices=range(n),
                                                                   ignore_hessian=False,
                                                                   force_refresh=False,
                                                                   approx_type='lissa')

plot_gradients(gradients=influence_grads,
               samples=train_samples,
               test_sample=test_samples[t],
               test_label=test_labels[t],
               labels=train_labels,
               plot_title='Feature Influences w/o Hessian')
