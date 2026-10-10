
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense
)
import matplotlib.pyplot as plt

(x, y), (xt, yt) = cifar10.load_data()
x = x.astype('float32') / 255
xt = xt.astype('float32') / 255

model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile('adam', 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=5, batch_size=64)
loss, acc = model.evaluate(xt, yt, verbose=0)
print("Test Accuracy:", acc*100)

filters = model.layers[0].get_weights()[0]
for i in range(8):
    plt.subplot(2, 4, i+1)
    f = filters[:, :, :, i]
    plt.imshow((f-f.min())/(f.max()-f.min()+1e-8))
    plt.axis('off')
plt.show()
