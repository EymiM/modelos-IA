import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Definindo os dados de entrada
x_train = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
y_train = np.array([-0.9602, -0.5770, -0.729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000], dtype=np.float32)

# Definindo o modelo da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo com o cálculo do erro quadrático médio em cada época
history = model.fit(x_train, y_train, epochs=10000, verbose=0)

# Obtendo o erro quadrático médio em cada época
loss = history.history['loss']

# Plotando o gráfico do erro quadrático médio em função das épocas
plt.plot(range(1, len(loss) + 1), loss)
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio')
plt.title('Erro Quadrático Médio ao Longo das Épocas')
plt.grid(True)
plt.show()

# Testando o modelo
x_test = np.linspace(0, 1, 1000)
y_pred = model.predict(x_test)

# Plotando os resultados
plt.scatter(x_train, y_train, color='red', label='Dados de treinamento')
plt.plot(x_test, y_pred, color='blue', label='Predições')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproximação da função')
plt.show()
