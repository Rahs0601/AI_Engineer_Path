# Calculus Concepts

Calculus is a branch of mathematics focused on rates of change and accumulation. In the context of AI, particularly machine learning and deep learning, calculus is essential for understanding optimization algorithms, such as gradient descent, which are used to train models.

## Key Concepts:

### 1. Functions

*   A rule that assigns each input exactly one output.
*   Notation: `f(x)`
*   Example: `f(x) = x^2`

### 2. Limits

*   The value that a function "approaches" as the input approaches some value.
*   Fundamental to the definition of derivatives and integrals.
*   Notation: `lim (x->a) f(x)`

### 3. Derivatives

*   **Definition**: The derivative of a function measures how sensitive the output of the function is to a change in its input. It represents the instantaneous rate of change of a function.
*   **Geometric Interpretation**: The slope of the tangent line to the graph of the function at a given point.
*   **Notation**: `f'(x)`, `dy/dx`, `d/dx f(x)`

#### Common Derivative Rules:

*   **Power Rule**: `d/dx (x^n) = n * x^(n-1)`
    *   Example: `d/dx (x^3) = 3x^2`
*   **Constant Rule**: `d/dx (c) = 0`
*   **Sum/Difference Rule**: `d/dx (f(x) +/- g(x)) = f'(x) +/- g'(x)`
*   **Product Rule**: `d/dx (f(x) * g(x)) = f'(x)g(x) + f(x)g'(x)`
*   **Quotient Rule**: `d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / (g(x))^2`
*   **Chain Rule**: `d/dx (f(g(x))) = f'(g(x)) * g'(x)` (used for composite functions)

### 4. Partial Derivatives

*   When a function has multiple input variables, a partial derivative measures the rate of change with respect to one variable, holding all other variables constant.
*   Notation: `∂f/∂x`
*   Example: For `f(x, y) = x^2 + y^3`, `∂f/∂x = 2x` and `∂f/∂y = 3y^2`

### 5. Gradient

*   **Definition**: For a multivariable function, the gradient is a vector that contains all the partial derivatives of the function.
*   It points in the direction of the steepest ascent of the function.
*   Notation: `∇f` (nabla f)
*   Example: For `f(x, y) = x^2 + y^3`, `∇f = [∂f/∂x, ∂f/∂y] = [2x, 3y^2]`

### 6. Optimization (Gradient Descent)

*   **Concept**: In machine learning, we often want to find the minimum of a cost or loss function. The gradient tells us the direction of the steepest increase, so to find the minimum, we move in the opposite direction of the gradient.
*   **Gradient Descent Algorithm**: Iteratively updates parameters by taking steps proportional to the negative of the gradient of the function at the current point.
    *   `parameter = parameter - learning_rate * gradient`

## Resources:

*   **Khan Academy**: Calculus (online course)
*   **3Blue1Brown**: Essence of Calculus (YouTube series)
*   **Textbook**: "Calculus" by James Stewart
