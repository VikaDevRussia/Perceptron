import numpy as np

def perceptron_train(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        errors = 0
        for xi, target in zip(X, y):
            z = np.dot(xi, weights) + bias
            pred = 1 if z >= 0 else 0
            error = target - pred
            
            # Обновляем веса и смещение только при ошибке
            if error != 0:
                weights += lr * error * xi
                bias   += lr * error
                errors += 1

        # Если ошибок нет — обучение завершено раньше срока
        if errors == 0:  
            break

    return weights, bias


def perceptron_predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return (z >= 0).astype(int)


# Тестовый пример для XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])
w, b = perceptron_train(X, y)
print("Веса:", w, "смещение:", b)
print("Предсказания:", perceptron_predict(X, w, b))