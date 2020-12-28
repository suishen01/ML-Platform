from keras.datasets import mnist
from MachineLearningModels.cnn import ConvolutionalNeuralNetwork
from pandas import DataFrame

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images /= 255
test_images /= 255

train_labels = DataFrame(train_labels).astype("str")
test_labels = DataFrame(test_labels).astype("str")
model = ConvolutionalNeuralNetwork(height=28, width=28, dimension=1, classes=10, label_headers=[0], epochs=100, batch_size=32)
model = model.fit(train_images, train_labels)
predictions = model.predict(test_images)
print(predictions)
score = model.score(test_images, test_labels)
print(score)
