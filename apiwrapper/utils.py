# import numpy as np
import os
from os.path import join 
import tensorflow as tf
import numpy as np
# import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from flask import jsonify
from PIL import Image

dataset_path ='dataset'
classes = ['Bed', 'Sofa', 'Chair']

def prepare_image(image):
    # Create a copy of the input array
    image_copy = np.copy(image)

    # Resize the copy to (224, 224) using PIL
    image_resized = Image.fromarray(image_copy).resize((224, 224))

    # Convert the resized image to float32 and normalize it
    image_normalized = np.asarray(image_resized).astype('float32') / 255.0

    # Add an extra dimension to the image array to match the model input shape
    image_expanded = np.expand_dims(image_normalized, axis=0)

    return image_expanded



def create_classification_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_data_generator(dataset_path = 'Data for test', batch_size=32, target_size=(224, 224)):
    """
    Returns a generator that yields batches of images and labels.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    return generator

def print_stats(history):
    """
    Print the evaluation metrics for the trained model.
    """
    print('Train loss:', history.history['loss'][-1])
    print('Train accuracy:', history.history['accuracy'][-1])
    print('Validation loss:', history.history['val_loss'][-1])
    print('Validation accuracy:', history.history['val_accuracy'][-1])

def plot_stats(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # plot training and validation accuracy
    axs[0].plot(history.history['accuracy'], label='Train')
    axs[0].plot(history.history['val_accuracy'], label='Validation')
    axs[0].set_title('Accuracy')
    axs[0].legend()

    # plot training and validation loss
    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Validation')
    axs[1].set_title('Loss')
    axs[1].legend()

    plt.show()


def train(dataset_path) : 
    model = create_classification_model((224, 224, 3), 3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    train_dataset_path = join(dataset_path, 'train')
    test_dataset_path = join(dataset_path, 'test')
    train_generator = get_data_generator(train_dataset_path)
    test_generator = get_data_generator(test_dataset_path)

    history = model.fit(train_generator, epochs=10, validation_data=test_generator)
    # evaluate the model
    print_stats(history)
    plot_stats(history)

    model.save('models/model.h5')


def load_trained_model(model_path = 'models/model.h5', dataset_path = 'dataset', load_classes = True) :
    model = tf.keras.models.load_model(model_path)
    if load_classes: 
        classes = get_data_generator(join(dataset_path, 'train')).class_indices
        model.classes_ = classes

    return model

def infer(image, model_path = 'models/model.h5') : 
    if isinstance(image, str):
        image = Image.open(image)
    
    model = load_trained_model(model_path)
    input_image = prepare_image(image)
    res = model.predict(input_image)
    label = np.argmax(res)
    c = sorted(classes)[label]
    return jsonify({'status': 'success', 'label': str(label), 'class': c})


def split_dataset(dataset_path) : 
    train_dir = join(dataset_path, 'train')
    test_dir = join(dataset_path, 'test')
    test_ratio = 0.2

    class_dirs = [join(dataset_path, f) for f in os.listdir(dataset_path)]

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for f in class_dirs : 
        if os.path.isdir(f):
            print("size of ", f, " is ", len(os.listdir(f)))
            train_subdir = join(train_dir, os.path.basename(f))
            test_subdir = join(test_dir, os.path.basename(f))
            if not os.path.exists(train_subdir):
                os.mkdir(train_subdir)
            if not os.path.exists(test_subdir):
                os.mkdir(test_subdir)
            files = os.listdir(f)
            test_size = int(len(files) * test_ratio)
            for file in files[:test_size]:
                os.rename(join(f, file), join(test_subdir, file))
            for file in files[test_size:]:
                os.rename(join(f, file), join(train_subdir, file))

    print("size of train is ", len(os.listdir(train_dir)))
    print("size of test is ", len(os.listdir(test_dir)))


def show_image(image_path) : 
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    plt.imshow(image)
    plt.show()


if __name__ == "__main__" : 
    sample_image_path = join(dataset_path, 'test', 'Bed', 'Baxton Studio Adela Modern and Contemporary Grey Finished Wood Queen Size Platform Bed.jpg')
    # show_image(sample_image_path)
    res = infer(sample_image_path) 
    print(res)
