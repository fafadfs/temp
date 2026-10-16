from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

data = load_iris()
x, y = data.data, data.target

a, b, c, d = train_test_split(
    x, y, test_size=0.2, random_state=42
)
sc = StandardScaler()
a, b = sc.fit_transform(a), sc.transform(b)

for act in ['relu', 'tanh', 'sigmoid']:
    model = Sequential([
        Input(shape=(4,)),
        Dense(8, activation=act),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(a, c, epochs=30, verbose=0)
    loss, acc = model.evaluate(b, d, verbose=0)
    print(act, "Accuracy:", acc, "Loss:", loss)

# done by sah
