# Recurrent Neural Networks (RNNs) Concepts

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data, where the order of information matters. Unlike traditional feedforward neural networks, RNNs have internal memory that allows them to use information from previous inputs to influence the processing of current inputs. This makes them suitable for tasks involving time series, natural language processing, and speech recognition.

## Key Concepts:

### 1. Recurrence

*   The defining characteristic of an RNN is its ability to pass information from one step of the sequence to the next.
*   A hidden state (or context) `h_t` at time step `t` is a function of the current input `x_t` and the previous hidden state `h_{t-1}`.
    *   `h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    *   `y_t = W_hy * h_t + b_y` (output at time `t`)
*   The same weights (`W_hh`, `W_xh`, `W_hy`) and biases (`b_h`, `b_y`) are reused across all time steps, which is known as **parameter sharing**.

### 2. Types of RNN Architectures

RNNs can be configured in various ways depending on the input and output sequence lengths:

*   **One-to-One**: Standard neural network (e.g., image classification).
*   **One-to-Many**: Single input, sequence output (e.g., image captioning).
*   **Many-to-One**: Sequence input, single output (e.g., sentiment analysis).
*   **Many-to-Many (Same Length)**: Sequence input, sequence output (e.g., machine translation, video classification frame by frame).
*   **Many-to-Many (Different Lengths)**: Sequence input, sequence output (e.g., sequence-to-sequence models for machine translation).

### 3. Vanishing and Exploding Gradients

*   **Vanishing Gradients**: During backpropagation through time (BPTT), gradients can become extremely small as they propagate backward through many time steps. This makes it difficult for the network to learn long-range dependencies.
*   **Exploding Gradients**: Conversely, gradients can become extremely large, leading to unstable training and large weight updates.
*   **Solutions**: Gradient clipping (to prevent exploding gradients), and more sophisticated RNN architectures like LSTMs and GRUs.

### 4. Long Short-Term Memory (LSTM) Networks

*   A special type of RNN designed to overcome the vanishing gradient problem and capture long-range dependencies.
*   Introduces a **cell state** (`C_t`) that runs through the entire chain, allowing information to be carried forward.
*   Uses **gates** to control the flow of information into and out of the cell state:
    *   **Forget Gate**: Decides what information to throw away from the cell state.
    *   **Input Gate**: Decides what new information to store in the cell state.
    *   **Output Gate**: Decides what part of the cell state to output.

### 5. Gated Recurrent Unit (GRU) Networks

*   A simplified version of LSTMs, also designed to handle vanishing gradients.
*   Combines the forget and input gates into a single **update gate**.
*   Combines the cell state and hidden state.
*   Generally has fewer parameters than LSTMs, making them computationally less expensive and sometimes faster to train, while often achieving comparable performance.

### 6. Applications of RNNs

*   **Natural Language Processing (NLP)**: Machine translation, sentiment analysis, text generation, speech recognition, named entity recognition.
*   **Time Series Analysis**: Stock price prediction, weather forecasting.
*   **Video Analysis**: Activity recognition.

## Resources:

*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**
*   **Andrew Ng's Deep Learning Specialization (Coursera)**
*   **TensorFlow/Keras Documentation**
*   **Colah's Blog: Understanding LSTMs (highly recommended visual explanation)**
