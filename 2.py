import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, Xt, y, yt = train_test_split(
    iris.data, iris.target, test_size=0.2,
    random_state=42, stratify=iris.target
)

sc = StandardScaler()
X, Xt = sc.fit_transform(X), sc.transform(Xt)

for h in ([8], [16, 8], [32, 16, 8]):
    for a in ('relu', 'tanh', 'sigmoid'):
        m = tf.keras.Sequential()
        m.add(tf.keras.Input(shape=(4,)))
        for n in h:
            m.add(tf.keras.layers.Dense(n, activation=a))
        m.add(tf.keras.layers.Dense(3, activation='softmax'))

        m.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        m.fit(X, y, epochs=50, batch_size=5, verbose=0)
        loss, acc = m.evaluate(Xt, yt, verbose=0)
        print(h, a, "Accuracy:", round(acc, 4),
              "Loss:", round(loss, 4))

p = m.predict(Xt[:1], verbose=0).argmax()
print("Predicted Species:", iris.target_names[p])
