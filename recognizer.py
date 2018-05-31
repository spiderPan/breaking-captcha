import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from captcha import Captcha


class Recognizer:
    def __init__(self):
        self.data = []
        self.labels = []
        self.model_filename = "./model/captcha_model.hdf5"
        self.model_lables_filename = "./model/model_labels.dat"
        self.captcha = Captcha()

    def load_captcha_folder(self, folder):
        for image_file in paths.list_images(folder):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = self.captcha.resize_to_fit(image, 20, 20)
            image = np.expand_dims(image, axis=2)
            label = image_file.split(os.path.sep)[-2]
            self.data.append(image)
            self.labels.append(label)

        self.data = np.array(self.data, dtype="float") / 255.0
        self.labels = np.array(self.labels)

    def run_in_test_folder(self, folder):
        errors = []
        total = 0
        for image_file in paths.list_images(folder):
            total += 1
            predict_txt = self.predict_model(image_file)
            file_name = os.path.basename(image_file)
            label, file_type = os.path.splitext(file_name)
            if predict_txt != label:
                print ('FALSE: Image is {}, Recognizer reads {}'.format(label, predict_txt))
                errors.append(label)
            else:
                print ('TRUE: Image is {}, Recognizer reads {}'.format(label, predict_txt))

        print('Run testing in {}, and got {} errors, accuracy is {}'.format(total, len(errors), len(errors) / total))
        with open('errors.txt', 'a') as error_file:
            pickle.dump(errors, error_file)

    def train_model(self):
        # Split the training data into separate train and test sets
        (X_train, X_test, Y_train, Y_test) = train_test_split(self.data, self.labels, test_size=0.25, random_state=0)

        # Convert the labels (letters) into one-hot encodings that Keras can work with
        lb = LabelBinarizer().fit(Y_train)
        Y_train = lb.transform(Y_train)
        Y_test = lb.transform(Y_test)

        # Save the mapping from labels to one-hot encodings.
        # We'll need this later when we use the model to decode what it's predictions mean
        with open(self.model_lables_filename, "wb") as f:
            pickle.dump(lb, f)

        # Build the neural network!
        model = Sequential()

        # First convolutional layer with max pooling
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second convolutional layer with max pooling
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Hidden layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        # Output layer with 32 nodes (one for each possible letter/number we predict)
        model.add(Dense(32, activation="softmax"))

        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train the neural network
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

        # Save the trained model to disk
        model.save(self.model_filename)

    def predict_model(self, image_file):
        # Load up the model labels (so we can translate model predictions to actual letters)
        with open(self.model_lables_filename, "rb") as f:
            lb = pickle.load(f)

        # Load the trained neural network
        model = load_model(self.model_filename)
        image = cv2.imread(image_file)
        letter_image_regions, image = self.captcha.split_captcha_into_letters(image)
        predictions = []

        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = self.captcha.resize_to_fit(letter_image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

        # Print the captcha's text
        captcha_text = "".join(predictions)
        print("CAPTCHA text is: {}".format(captcha_text))

        return captcha_text
