import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard
import pickle
from time import gmtime, strftime


class Trainer:
    def __init__(self, image_size=(32, 48), batch_size=16, number_of_epochs=100, learning_rate=0.0001):
        self.image_size = image_size
        img_height = self.image_size[0]
        img_width = self.image_size[1]
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_height, img_width)
        else:
            self.input_shape = (img_height, img_width, 1)
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate

    def train_model(self,
                    train_dir_path,
                    number_of_train_samples,
                    validation_dir_path,
                    number_of_validation_samples,
                    model_path,
                    class_to_id_mapping_path,
                    tensor_flow_log_dir_path=None):
        model = self.__create_model(number_of_classes=Trainer.__get_number_of_classes(dir_path=train_dir_path))

        train_data_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2])
        train_generator = self.__create_generator(
            data_generator=train_data_generator,
            dir_path=train_dir_path)

        validation_data_generator = ImageDataGenerator(rescale=1. / 255)
        validation_generator = self.__create_generator(
            data_generator=validation_data_generator,
            dir_path=validation_dir_path)

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=number_of_train_samples // self.batch_size,
            epochs=self.number_of_epochs,
            validation_data=validation_generator,
            validation_steps=number_of_validation_samples // self.batch_size,
            callbacks=Trainer.__create_callbacks(tensor_flow_log_dir_path))

        model.save(filepath=model_path)
        pickle.dump(obj=train_generator.class_indices,
                    file=open(class_to_id_mapping_path, 'wb'))

    @staticmethod
    def __get_number_of_classes(dir_path):
        return len(next(os.walk(dir_path))[1])

    def __create_model(self, number_of_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(number_of_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def __create_generator(self, data_generator, dir_path):
        return data_generator.flow_from_directory(
            directory=dir_path,
            target_size=self.image_size,
            color_mode='grayscale',
            batch_size=self.batch_size)

    @staticmethod
    def __create_callbacks(tensor_flow_log_dir_path):
        callbacks = []

        if tensor_flow_log_dir_path is None:
            return callbacks

        log_dir_path = tensor_flow_log_dir_path + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '/'
        callbacks.append(TensorBoard(log_dir=log_dir_path,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=False))
        return callbacks
