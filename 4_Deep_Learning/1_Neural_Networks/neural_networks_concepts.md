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

*   Algorithms used to adjust the weights and biases of the neural network during training to minimize the loss function.
*   **Gradient Descent**: The basic optimizer, where weights are updated in the direction opposite to the gradient of the loss function.
*   **Stochastic Gradient Descent (SGD)**: Updates weights using the gradient of a single training example or a small batch of examples.
*   **Adam (Adaptive Moment Estimation)**: A popular and efficient optimizer that combines ideas from RMSprop and Adagrad.

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
