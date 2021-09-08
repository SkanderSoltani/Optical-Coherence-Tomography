from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import seaborn as sns
import numpy as np
import glob
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

# Train and val image paths
#train_images = glob.glob("../../data/new_data_split/train/*/*")
val_images = glob.glob("../../data/new_data_split/val/*/*") # 2% of the data
test_images = glob.glob("../../data/OCT2017/test/*/*")
print(len(val_images))


def get_train_images(train_percentage):
    """
    :param train_percentage: the percentage of samples wanted for the training dataset
    :return: balanced set of training data
    """
    train_images_cnv = glob.glob("../../data/new_data_split/train/CNV/*")
    train_images_drusen = glob.glob("../../data/new_data_split/train/DRUSEN/*")
    train_images_normal = glob.glob("../../data/new_data_split/train/NORMAL/*")
    train_images_dme = glob.glob("../../data/new_data_split/train/DME/*")

    train_images_cnv = np.random.choice(train_images_cnv, round(len(train_images_cnv) * (train_percentage/100)), replace=False)
    train_images_drusen = np.random.choice(train_images_drusen, round(len(train_images_drusen) * (train_percentage/100)), replace=False)
    train_images_normal = np.random.choice(train_images_normal, round(len(train_images_normal) * (train_percentage/100)), replace=False)
    train_images_dme = np.random.choice(train_images_dme, round(len(train_images_dme) * (train_percentage/100)), replace=False)

    list1 = np.append(train_images_dme, train_images_normal)
    list2 = np.append(train_images_cnv, train_images_drusen)
    return np.append(list1, list2)


train_images_10 = get_train_images(10)


def prepare_images(image_paths):
    """
    :param image_paths: paths to images
    :return: numpy array of images and labels
    """
    images = []
    labels = []

    for image in tqdm(image_paths):
        image_pixels = Image.open(image)
        image_pixels = image_pixels.convert("RGB")
        image_pixels = image_pixels.resize((224, 224))
        image_pixels = img_to_array(image_pixels)
        image_pixels = image_pixels / 255.

        label = image.split("/")[-1].split("-")[0]

        images.append(image_pixels)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    print(images.shape, labels.shape)

    return images, labels


X_train, y_train = prepare_images(train_images_10)
X_val, y_val = prepare_images(val_images)
X_test, y_test = prepare_images(test_images)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)


# Architecture utils
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
    base_model.trainable = True
    inputs = Input((224, 224, 3))
    h = base_model(inputs, training=False)
    h = GlobalAveragePooling2D()(h)

    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr


resnet_simclr = get_resnet_simclr(256, 128, 50)
resnet_simclr.load_weights(filepath="checkPoints/sim_weights")
resnet_simclr.summary()


def plot_training(H):
    with plt.xkcd():
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()


def get_linear_model(features):
    linear_model = Sequential([Dense(4, input_shape=(features,), activation="softmax")])
    """[
        Dense(1024, input_shape=(features,), activation="relu"),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0))
    ]"""
    return linear_model


resnet_simclr.layers[1].trainable = False
resnet_simclr.summary()
# Early Stopping to prevent overfitting
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=2, restore_best_weights=True)

"""
# Encoder model with non-linear projections
projection = Model(resnet_simclr.input, resnet_simclr.layers[-2].output)

# Extract train and val features
train_features = projection.predict(X_train)
val_features = projection.predict(X_val)
test_features = projection.predict(X_test)

linear_model = get_linear_model(128)
linear_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")
history = linear_model.fit(train_features, y_train_enc,
                           validation_data=(val_features, y_val_enc),
                           batch_size=64,
                           epochs=35,
                           callbacks=[es])
plot_training(history)

# Plot evaluation metrics on test data
y_pred = linear_model.predict(test_features)
y_true = y_test_enc
y_pred_max = y_pred.argmax(axis=1)

print(classification_report(y_true=y_true,y_pred=y_pred_max))
print(confusion_matrix(y_true=y_true,y_pred=y_pred_max))

# Encoder model with less non-linearity
projection = Model(resnet_simclr.input, resnet_simclr.layers[-4].output)

# Extract train and val features
train_features = projection.predict(X_train)
val_features = projection.predict(X_val)
test_features = projection.predict(X_test)

print(train_features.shape, val_features.shape)

linear_model = get_linear_model(256)
linear_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")
history = linear_model.fit(train_features, y_train_enc,
                           validation_data=(val_features, y_val_enc),
                           batch_size=64,
                           epochs=35,
                           callbacks=[es])
plot_training(history)

# Plot evaluation metrics on test data
y_pred = linear_model.predict(test_features)
y_true = y_test_enc
y_pred_max = y_pred.argmax(axis=1)

print(classification_report(y_true=y_true,y_pred=y_pred_max))
print(confusion_matrix(y_true=y_true,y_pred=y_pred_max))
"""
# Encoder model with no projection
projection = Model(resnet_simclr.input, resnet_simclr.layers[-6].output)

# Extract train and val features
train_features = projection.predict(X_train)
val_features = projection.predict(X_val)
test_features = projection.predict(X_test)

print(train_features.shape, val_features.shape)

linear_model = get_linear_model(2048)
linear_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")
history = linear_model.fit(train_features, y_train_enc,
                           validation_data=(val_features, y_val_enc),
                           batch_size=64,
                           epochs=35,
                           callbacks=[es])
plot_training(history)
# Plot evaluation metrics on test data
y_pred = linear_model.predict(test_features)
y_true = y_test_enc
y_pred_max = y_pred.argmax(axis=1)

print(classification_report(y_true=y_true,y_pred=y_pred_max))
print(confusion_matrix(y_true=y_true,y_pred=y_pred_max))
linear_model.save("SimCLR_model")
# Visualization of the representations
def plot_vecs_n_labels(v, labels):
    fig = plt.figure(figsize=(10, 10))
    sns.set_style("darkgrid")
    sns.scatterplot(v[:, 0], v[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", 4))
    plt.show()

    return fig

"""
# Representations with no nonlinear projections
tsne = TSNE()
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, y_train_enc)

# Representations with second last ReLU
tsne = TSNE()
projection = Model(resnet_simclr.input, resnet_simclr.layers[-4].output)
train_features = projection.predict(X_train)
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, y_train_enc)

# Representations with the last ReLU
tsne = TSNE()
projection = Model(resnet_simclr.input, resnet_simclr.layers[-2].output)
train_features = projection.predict(X_train)
low_vectors = tsne.fit_transform(train_features)
fig = plot_vecs_n_labels(low_vectors, y_train_enc)
"""