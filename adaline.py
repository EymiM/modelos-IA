import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.random.random(X.shape[1] + 1)
        self.errors = []

        for _ in range(self.n_iterations):
            output = self.predict(X)
            errors = y - output
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.errors.append(cost)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

# Dados
data = np.array([
    [2.215, 2.063, -1],
    [0.224, 1.586, 1],
    [0.294, 0.651, 1],
    [2.327, 2.932, -1],
    [2.497, 2.322, -1],
    [0.169, 1.943, 1],
    [1.274, 2.428, -1],
    [1.526, 0.596, 1],
    [2.009, 2.161, -1],
    [1.759, 0.342, 1],
    [1.367, 0.938, 1],
    [2.173, 2.719, -1],
    [0.856, 1.904, 1],
    [2.21, 1.868, -1],
    [1.587, 1.642, -1],
    [0.35, 0.84, 1],
    [1.441, 0.09, 1],
    [0.185, 1.327, 1],
    [2.764, 1.149, -1],
    [1.947, 1.598, -1]
])

# Separar X e y
X = data[:, :-1]
y = data[:, -1]

# Normalização dos dados
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Treinamento
adaline = Adaline(learning_rate=0.01, n_iterations=1000)
adaline.fit(X_normalized, y)

# Plot do erro durante o treinamento
plt.plot(range(1, len(adaline.errors) + 1), adaline.errors, marker='o')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Total')
plt.title('Erro Quadrático Total durante o Treinamento')
plt.show()

# Teste
predictions = adaline.predict(X_normalized)
print("Saídas previstas:", predictions)
