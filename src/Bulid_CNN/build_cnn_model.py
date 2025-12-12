import tensorflow as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_cnn_model(input_shape, num_classes):

    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),


        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),


        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),


        Flatten(),


        Dense(256, activation='relu'),


        Dropout(0.5),


        Dense(num_classes, activation='softmax')
    ])


    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    sample_shape = (128, 128, 3)

    sample_classes = 10

    model = build_cnn_model(sample_shape, sample_classes)
    model.summary()
    print("Model Built and Compiled Successfully.")