import os

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import TensorBoard

config = dict(
    dropout=0.2,
    hidden_layer_size=128,
    layer_1_size=16,
    layer_2_size=32,
    learn_rate=0.01,
    decay=1e-6,
    momentum=0.9,
    epochs=50,
)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

img_width = 28
img_height = 28

X_train = X_train.astype("float32")
X_train /= 255.0
X_test = X_test.astype("float32")
X_test /= 255.0

# reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(
    lr=config["learn_rate"],
    decay=config["decay"],
    momentum=config["momentum"],
    nesterov=True,
)

# build model
model = Sequential()
model.add(
    Conv2D(
        config["layer_1_size"],
        (5, 5),
        activation="relu",
        input_shape=(img_width, img_height, 1),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(config["layer_2_size"], (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config["dropout"]))
model.add(Flatten())
model.add(Dense(config["hidden_layer_size"], activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

metrics = [
    "accuracy",
    "mse",
    metrics.Precision(),
    metrics.AUC(
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        label_weights=None,
    ),
]

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=metrics)
tensorboard = TensorBoard(log_dir="/opt/dkube/model/")
os.system("ls -l /opt/dkube/model")
model.fit(
    X_train,
    y_train,
    verbose=0,
    validation_data=(X_test, y_test),
    epochs=config["epochs"],
    callbacks=[tensorboard],
)

model.save("/opt/dkube/model/cnn.h5")
os.system("ls -l /opt/dkube/model")
