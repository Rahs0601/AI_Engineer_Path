# Neural Networks Concepts

Neural Networks (NNs), also known as Artificial Neural Networks (ANNs), are the foundational building blocks of deep learning. Inspired by the human brain, they are designed to recognize patterns and relationships in data through a series of interconnected nodes (neurons) organized in layers.

## Key Concepts:

### 1. Neurons (Nodes)

*   The basic unit of a neural network.
*   Receives one or more inputs, applies a transformation (weighted sum + bias), and then an activation function to produce an output.

### 2. Layers

*   **Input Layer**: Receives the raw data. The number of neurons typically equals the number of features in the input data.
*   **Hidden Layers**: Layers between the input and output layers. These layers perform the bulk of the computation and pattern recognition. A network with one or more hidden layers is considered a "deep" neural network.
*   **Output Layer**: Produces the final prediction of the network. The number of neurons and the activation function depend on the type of problem (e.g., one neuron for binary classification, multiple for multi-class classification or regression).

### 3. Weights and Biases

*   **Weights (w)**: Parameters that determine the strength of the connection between neurons. They are learned during the training process.
*   **Bias (b)**: An additional parameter added to the weighted sum of inputs. It allows the activation function to be shifted, enabling the model to fit a wider range of data.

### 4. Activation Functions

*   A function applied to the output of each neuron (after the weighted sum + bias) to introduce non-linearity into the network.
*   Without non-linear activation functions, a neural network would simply be a linear model, regardless of the number of layers.
*   **Common Activation Functions**:
    *   **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))`. Outputs values between 0 and 1. Used in output layers for binary classification.
    *   **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`. Popular in hidden layers due to its computational efficiency and ability to mitigate the vanishing gradient problem.
    *   **Tanh (Hyperbolic Tangent)**: `f(x) = (e^x - e^-x) / (e^x + e^-x)`. Outputs values between -1 and 1.
    *   **Softmax**: Used in the output layer for multi-class classification. Converts a vector of numbers into a probability distribution, where the sum of probabilities is 1.

### 5. Forward Propagation

*   The process of passing input data through the network, from the input layer, through the hidden layers, to the output layer, to generate a prediction.
*   For each neuron: `output = activation_function(sum(input * weight) + bias)`

### 6. Loss Function (Cost Function)

*   Measures the discrepancy between the network's predicted output and the actual target output.
*   The goal of training is to minimize this loss function.
*   **Common Loss Functions**:
    *   **Mean Squared Error (MSE)**: For regression problems.
    *   **Binary Cross-Entropy**: For binary classification problems.
    *   **Categorical Cross-Entropy**: For multi-class classification problems.

### 7. Backpropagation

*   The core algorithm for training neural networks.
*   It's an algorithm for efficiently calculating the gradients of the loss function with respect to the weights and biases of the network.
*   It uses the chain rule of calculus to propagate the error backward from the output layer to the input layer, adjusting weights and biases along the way.

### 8. Optimizers

Optimizers are algorithms or methods used to change the attributes of your neural network, such as weights and learning rate, in order to reduce the losses. They are crucial for efficient training of deep learning models.

*   **Gradient Descent (GD)**: The most basic optimization algorithm. It updates the model parameters (weights and biases) in the direction opposite to the gradient of the loss function with respect to the parameters. It computes the gradient using the entire training dataset, which can be slow for large datasets.

*   **Stochastic Gradient Descent (SGD)**: Instead of computing the gradient using the entire dataset, SGD updates the parameters using the gradient of a single randomly chosen training example at each step. This makes it much faster than GD, but the updates are noisy, leading to oscillations.

*   **Mini-Batch Gradient Descent**: A compromise between GD and SGD. It updates parameters using the gradient of a small batch of training examples (typically 32, 64, 128, or 256). This reduces the noise of SGD while still being computationally more efficient than GD.

*   **Momentum**: An extension to SGD that helps accelerate convergence, especially in directions of consistent gradient and dampens oscillations in irrelevant directions. It adds a fraction of the update vector of the past time step to the current update vector.

*   **Adagrad (Adaptive Gradient Algorithm)**: Adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features and larger updates for parameters associated with infrequent features. It accumulates the square of past gradients.

*   **RMSprop (Root Mean Square Propagation)**: An unpublished, adaptive learning rate optimization algorithm proposed by Geoff Hinton. It tries to resolve Adagrad's radically diminishing learning rates by using a moving average of squared gradients.

*   **Adam (Adaptive Moment Estimation)**: One of the most popular and effective optimizers. It combines the advantages of both Adagrad and RMSprop. It computes adaptive learning rates for each parameter and also incorporates momentum.
    *   **Adamax**: A variant of Adam based on the infinity norm.
    *   **Nadam (Nesterov-accelerated Adaptive Moment Estimation)**: Adam with Nesterov momentum.

*   **Adadelta**: An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It does not require setting a default learning rate and is robust to different model architectures.

*   **Learning Rate Schedules**: Techniques to adjust the learning rate during training. Common schedules include:
    *   **Step Decay**: Reduce learning rate by a factor every few epochs.
    *   **Exponential Decay**: Learning rate decays exponentially over time.
    *   **Cosine Annealing**: Learning rate follows a cosine curve, starting high and decreasing to a minimum.

*   **Early Stopping**: A form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent. It stops the training process when the performance on a validation dataset starts to degrade.

### 9. Hyperparameters

*   Parameters that are set *before* the training process begins, not learned by the model.
*   Examples: Learning rate, number of hidden layers, number of neurons per layer, choice of activation function, batch size, number of epochs.

### 10. Overfitting and Regularization

*   Neural networks are prone to overfitting.
*   **Regularization Techniques**:
    *   **Dropout**: Randomly sets a fraction of neurons to zero during training, preventing complex co-adaptations.
    *   **L1/L2 Regularization**: Adds a penalty to the loss function based on the magnitude of weights.
    *   **Early Stopping**: Stops training when the performance on a validation set starts to degrade.

## Resources:

*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**
*   **Andrew Ng's Deep Learning Specialization (Coursera)**
*   **TensorFlow/Keras Documentation**
*   **3Blue1Brown: Neural Networks (YouTube series)**
