import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def fit(self, X, y):
        # Number of training examples and features
        m, n = X.shape

        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones((m, 1)), X]

        # Initialize theta (parameters) to zero
        self.theta = np.zeros(n + 1)

        # Gradient Descent
        for _ in range(self.iterations):
            gradients = self._compute_gradients(X, y, m)
            self.theta -= self.learning_rate * gradients

    def _compute_gradients(self, X, y, m):
        # Predicted values
        predictions = X.dot(self.theta)

        # Compute gradients
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)

        return gradients

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X.dot(self.theta)

    def cost_function(self, X, y):
        # Compute the cost (mean squared error)
        m = len(y)
        predictions = X.dot(self.theta)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost


# Create some sample data
X = np.array([[1], [2], [3], [4], [5]])  # Input feature (5 samples)
y = np.array([2, 4, 6, 8, 10])  # Target variable

# Create and train the model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7]]))

print("Predicted values:", predictions)
