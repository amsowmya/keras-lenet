from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K

class Lenet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        model = Sequential()
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Conv2D(50, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
