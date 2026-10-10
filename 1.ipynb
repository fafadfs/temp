
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x, y), (xt, yt) = mnist.load_data()
x, xt = x.reshape(-1, 784)/255, xt.reshape(-1, 784)/255

np.random.seed(1)
w1 = np.random.randn(784, 64)*0.05
w2 = np.random.randn(64, 10)*0.05
b1, b2 = np.zeros(64), np.zeros(10)
losses = []

for epoch in range(5):
    total_loss = 0
    for i in range(0, len(x), 256):
        a = x[i:i+256]
        t = np.eye(10)[y[i:i+256]]
        h = 1/(1+np.exp(-(a@w1+b1)))
        z = h@w2+b2
        p = np.exp(z-z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)

        total_loss -= (t*np.log(p+1e-9)).sum()
        d2 = (p-t)/len(a)
        d1 = (d2@w2.T)*h*(1-h)

        w2 -= 0.5*h.T@d2
        b2 -= 0.5*d2.sum(axis=0)
        w1 -= 0.5*a.T@d1
        b1 -= 0.5*d1.sum(axis=0)

    losses.append(total_loss/len(x))
    print("Epoch:", epoch+1, "Loss:", round(losses[-1], 3))

h = 1/(1+np.exp(-(xt@w1+b1)))
pred = (h@w2+b2).argmax(axis=1)
print("Test Accuracy:", np.mean(pred == yt)*100)
print("Predicted Digit:", pred[0], "Actual:", yt[0])
plt.plot(losses)
plt.title("Training Loss")
plt.show()
