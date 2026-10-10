
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x, y), (xt, yt) = cifar10.load_data()
x = x.astype('float32') / 255
xt = xt.astype('float32') / 255

results = {}
for name, opt in [
    ('SGD', tf.keras.optimizers.SGD(0.01, momentum=0.9)),
    ('Adam', tf.keras.optimizers.Adam(0.001))
]:
    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(opt, 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(x, y, epochs=5, batch_size=128, verbose=0)
    results[name] = h.history
    loss, acc = model.evaluate(xt, yt, verbose=0)
    print(name, "Loss:", round(loss, 3),
          "Accuracy:", round(acc*100, 2))

for metric in ['loss', 'accuracy']:
    plt.figure()
    for name in results:
        plt.plot(results[name][metric], label=name)
    plt.title(metric)
    plt.legend()
    plt.show()
