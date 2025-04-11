# Advanced Deep Learning with TensorFlow and Keras

This repository contains a series of Colaboratory notebooks demonstrating a comprehensive range of advanced deep learning concepts using TensorFlow and Keras. 
- **Part 1** focuses on enhancing model generalization through various data augmentation techniques (often with A/B testing for comparison) and regularization methods implemented in TensorFlow, including L1/L2 regularization, dropout, early stopping, Monte Carlo dropout, diverse weight initialization strategies, batch normalization, custom dropout and regularization, the use of callbacks and TensorBoard for training management, hyperparameter tuning with Keras Tuner, leveraging KerasCV for image augmentation, exploring data augmentation for diverse data modalities (image, video, text, time series, tabular, speech, document images), and showcasing the data augmentation capabilities of the Fastai library.
- **Part 2** delves into advanced Keras deep learning constructs within TensorFlow, illustrating the creation and application of user-defined learning rate schedulers, custom dropout and normalization layers, effective TensorBoard integration, custom loss functions and evaluation metrics, custom activation functions, initializers, regularizers, and kernel weight constraints, the development of entirely custom layers and model architectures, the implementation of custom optimization algorithms, and the creation of fully customized training loops for maximum control over the learning process. This repository serves as a practical guide to mastering advanced techniques for building robust and high-performing deep learning models with TensorFlow and Keras.

## GitHub Directory Structure

**Part 1: Data Augmentation and Generalization Techniques**
- **1-a:** L1 and L2 Regularization
- **1-b:** Dropout
- **1-c:** Early Stopping
- **1-d:** Monte Carlo Dropout
- **1-e:** Various Initializations
- **1-f:** Batch Normalization
- **1-g:** Custom Dropout, Custom Regularization
- **1-h:** Using Callbacks and TensorBoard
- **1-i:** Using Keras Tuner
- **1-j:** Using KerasCV Data Augmentation
- **1-k:** Data Augmentation for Multiple Data Types
- **1-l:** Demonstrating Fastai Data Augmentation

**Part 2: Advanced Keras Deep Learning Constructs**
- **2-a:** User Custom Learning Rate Scheduler
- **2-b:** Use Custom Dropout
- **2-c:** Use Custom Normalization
- **2-d:** Use TensorBoard
- **2-e:** Use Custom Loss Function
- **2-f:** Use Custom Activation Function, Initializer, Regularizer, and Kernel Weight Constraint
- **2-g:** Use Custom Metric
- **2-h:** Use Custom Layers
- **2-i:** Use Custom Model
- **2-j:** Custom Optimizer
- **2-k:** Custom Training Loop

## Part 1: Data Augmentation and Generalization Techniques

This part explores various methods to improve the generalization of deep learning models by augmenting training data and applying regularization techniques.

### 1-a) L1 and L2 Regularization (`Part1a_TensorFlow_L1andL2Regularization.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MyixcepKrCK8DZ_qEttIa1rzOWaQLRWI?usp=sharing)

**Description:** Demonstrates how to apply L1 and L2 regularization to the kernel weights of dense layers in Keras.

**Execution in Colab:** Train these models and compare their performance (loss, accuracy) on a validation set to a baseline model without regularization. Observe how L1 tends to drive weights to zero (sparsity), while L2 penalizes large weights.

### 1-b) Dropout (`Part1b_TensorFlow_DropoutRegularization.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mreIxQqqIY2fPHY-JLWYWT7TFfpSSjZ0?usp=sharing)

**Description:** Illustrates the use of dropout layers to prevent overfitting.

**Execution in Colab:** Train this model and compare its validation performance to a model without dropout. Dropout should help reduce overfitting.

### 1-c) Early Stopping (`Part1c_TensorFlow_EarlyStopping.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OTcCJlxPGTV7HhsWQZDVjxUmhjo-4Bwd?usp=sharing)

**Description:** Shows how to use the `EarlyStopping` callback to halt training when validation loss stops improving.

**Execution in Colab:** Run the training and observe when the `EarlyStopping` callback triggers. The `restore_best_weights=True` argument will ensure that the model with the best validation performance is used.

### 1-d) Monte Carlo Dropout (`Part1d_TensorFlow_MonteCarloDropout.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mqs9GG5qJXqK1wX7QwAIgp5OLDswSFsN?usp=sharing)

**Description:** Demonstrates how to use dropout at inference time to get a distribution of predictions, which can be useful for uncertainty estimation.

**Execution in Colab:** Run the code and observe the mean and standard deviation of the predictions. Higher standard deviation indicates higher uncertainty.

### 1-e) Various Initializations (`Part1e_TensorFlow_Initializations.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PFuPBhzyYDZrPyiVyy5KF0T23pJle18F?usp=sharing)

**Description:** Shows how to use different weight initialization strategies in Keras layers and when to use them (e.g., `glorot_normal`, `he_normal`, `lecun_normal`).

**Execution in Colab:** Train these models and compare their training curves (loss and accuracy) to see how different initializations affect learning.

### 1-f) Batch Normalization (`Part1f_TensorFlow_BatchNormalization.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17JFTxnJ96AWcDv3VYy1tF6TatQR8LFUE?usp=sharing)

**Description:** Demonstrates the use of batch normalization layers to stabilize training and potentially improve generalization.

**Execution in Colab:** Train this model and compare its training speed and validation performance to a model without batch normalization. Batch normalization often allows for higher learning rates and faster convergence.

### 1-g) Custom Dropout, Custom Regularization (`Part1g_TensorFlow_CustomRegularizationDropout.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1goyP-TiMD_iK3wPgxapxdyb96lkJ71Zj?usp=sharing)

**Description:** Shows how to create custom dropout layers (e.g., based on activation) and custom regularization techniques by implementing custom layer classes.

**Execution in Colab:** Implement and train these models. Observe how the custom dropout behaves based on activation values and how the custom activity regularizer penalizes large activations.

### 1-h) Using Callbacks and TensorBoard (`Part1h_TensorFlow_Callbacks_TensorBoard.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RPB7i1cm402Euk1FUFUX6X-GaljrTBns?usp=sharing)

**Description:** Demonstrates how to use Keras callbacks like `ModelCheckpoint`, `EarlyStopping`, and `TensorBoard` for monitoring and controlling the training process. (Reference: [Hands-On ML3 - Chapter 10](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb))

**Execution in Colab:** Run the training with TensorBoard callbacks and then use the `%tensorboard` magic command to view the logs.

### 1-i) Using Keras Tuner (`Part1i_TensorFlow_KerasTuner.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O-1FrmMAj9BlAedeaESMuUVOiLbfwAjP?usp=sharing)

**Description:** Shows how to use Keras Tuner to perform hyperparameter optimization. (Hint: [Hands-On ML3 - Chapter 11](https://github.com/ageron/handson-ml3/blob/main/11_training_deep_neural_networks.ipynb), [TensorFlow.org](https://www.tensorflow.org/))

**Execution in Colab:** Run the Keras Tuner search. Observe the different hyperparameter combinations tried and the resulting validation accuracies. The best hyperparameters and the corresponding model will be printed.

### 1-j) Using KerasCV Data Augmentation (`Part1j_KerasCV_DataAugmentation.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wNTYji7sH9qfjDuRYUyj4BnrrJGZL6uH?usp=sharing)

**Description:** Demonstrates how to use the data augmentation layers provided by KerasCV. (Reference: [KerasCV Documentation](https://keras.io/keras_cv))

**Execution in Colab:** Explore and apply various KerasCV augmentation layers to image data and observe the results. Train a model with and without KerasCV augmentation to see the impact on performance.

### 1-k) Data Augmentation for Multiple Data Types (`Part1k_DataAugmentationAndClassification.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DibauYM0vabWkjsaMKohmGurQeF7ot_B?usp=sharing)

**Description:** This notebook explores data augmentation techniques for different data modalities, including image, video, text (using libraries like `nlpaug`), and potentially time series or tabular data (using libraries like `tsaug` or custom methods). (References: [Data Augmentation Review](https://github.com/AgaMiko/data-augmentation-review), [AugLy](https://ai.facebook.com/blog/augly-a-new-data-augmentation-library-to-help-build-more-robust-ai-models), [TensorFlow Image Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation))

**Execution in Colab:** Run the code to apply augmentation techniques to sample data from different modalities and observe the transformed data. Train separate models with and without augmentation for each data type to compare performance.

### 1-l) Demonstrating Fastai Data Augmentation (`Part1l_FastaiDataAugmentation.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ju_LYlK93Kt3KCVETd1ENX578k-Tdf8r?usp=sharing)

**Description:** This notebook demonstrates the data augmentation capabilities of the Fastai library. (Hint: [Fastbook - Chapter 7](https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb))

**Execution in Colab:** Load image data and use Fastai's `ImageDataLoaders` and `aug_transforms` to visualize augmented images. Train a model using Fastai's data augmentation pipeline and observe its performance.

## Part 2: Advanced Keras Deep Learning Constructs (`Part2_AdvancedKerasConstructs.ipynb`) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dVGh-t_8x-3sAh7hYumruJJj8pExT4Xz?usp=sharing)

I did all the parts of part 2 in a single colab. This notebook demonstrates various advanced Keras deep learning constructs. 

### 2-a) User Custom Learning Rate Scheduler

**Description:** Implements a custom learning rate scheduler, potentially including a OneCycleScheduler. (Reference: [Hands-On ML2 - Chapter 11](https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb))

**Execution in Colab:** Define and use a custom learning rate scheduler during model training. Plot the learning rate over epochs to observe its behavior and analyze the impact on training.

### 2-b) Use Custom Dropout

**Description:** Implements a custom dropout layer, such as MCAlphaDropout. (Reference: [Hands-On ML3 - Chapter 10](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb))

**Execution in Colab:** Implement the custom dropout layer and integrate it into a Keras model. Train the model and compare its performance to a model using standard dropout.

### 2-c) Use Custom Normalization

**Description:** Implements a custom normalization layer, such as MaxNormDense. (Reference: [Hands-On ML3 - Chapter 10](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb))

**Execution in Colab:** Implement the custom normalization layer and integrate it into a Keras model. Train the model and observe its behavior compared to using standard BatchNormalization or LayerNormalization.

### 2-d) Use TensorBoard

**Description:** Integrates TensorBoard for visualizing training metrics, model graphs, and other debugging information. (Reference: [Hands-On ML3 - Chapter 10](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb))

**Execution in Colab:** Run the training with TensorBoard callbacks and then use the `%tensorboard` magic command to view the logs.

### 2-e) Use Custom Loss Function

**Description:** Defines and uses a custom loss function, such as Huber Loss. (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement the custom loss function and compile it with a Keras model. Train the model and observe the training loss values. Compare the performance to using a standard loss function.

### 2-f) Use Custom Activation Function, Initializer, Regularizer, and Kernel Weight Constraint

**Description:** Defines and uses custom activation functions (e.g., Leaky ReLU), initializers (e.g., a custom Glorot initializer), regularizers (e.g., custom L1), and kernel weight constraints (e.g., enforcing positive weights). (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement these custom components and integrate them into a Keras model. Train the model and observe their effects on the learning process and model weights.

### 2-g) Use Custom Metric

**Description:** Defines and uses a custom metric, such as a custom Huber metric. (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement the custom metric and include it in the `metrics` list when compiling a Keras model. Observe the reported metric values during training and evaluation.

### 2-h) Use Custom Layers

**Description:** Implements custom layers (e.g., an exponential layer, a custom dense layer, a noise layer, or a layer normalization layer) by subclassing the `Layer` class. (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement the custom layers and integrate them into a Keras model. Train the model and verify that the custom layers behave as expected.

### 2-i) Use Custom Model

**Description:** Builds a custom model architecture (e.g., a Residual Regressor or a model with Residual Blocks) by subclassing the `Model` class. (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement the custom model architecture and train it on a suitable dataset. Verify that the model structure and forward pass are correctly defined.

### 2-j) Custom Optimizer

**Description:** Implements a custom optimizer, such as the `MyMomentumOptimizer` shown previously. (Reference: [Hands-On ML2 - Chapter 12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Implement the custom optimizer and use it to train a Keras model. Compare its performance to using a standard Keras optimizer like Adam or SGD.

### 2-k) Custom Training Loop

**Description:** Writes a custom training loop for more control over the training process. (Reference: [Hands-On ML3 - Section 13 for Fashion MNIST](https://github.com/ageron/handson-ml3/blob/main/13_custom_models_and_training_with_tensorflow.ipynb))

**Execution in Colab:** Run the custom training loop and observe the step-by-step training process, including loss calculation, gradient application, and metric tracking. Compare the results to training with the `fit()` method.
