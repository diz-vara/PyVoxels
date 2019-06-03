from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import pickle
import numpy as np


class Classifier:
    def __init__(self, model_file_path, category_to_id_mapping_file_path):
        self.model = load_model(model_file_path)
        self.model._make_predict_function()
        self.category_to_id_mapping = pickle.load(file=open(category_to_id_mapping_file_path, 'rb'))
        self.id_to_category_mapping = {v: k for k, v in self.category_to_id_mapping.items()}
        
        input_shape = self.model.get_layer(index=0).input_shape
        self.target_size = (input_shape[1], input_shape[2])
        self.image_size = (self.target_size[1], self.target_size[0])

    def get_class_names(self):
        return self.category_to_id_mapping.keys()

    def get_number_of_classes(self):
        return len(self.category_to_id_mapping)

    def get_class_id_by_name(self, class_name):
        return self.category_to_id_mapping.get(class_name)

    def get_class_name_by_id(self, id):
        return self.id_to_category_mapping.get(id)

    def classify_image_from_array(self, image_as_array):
        return self.model.predict_classes(self.__prepare_image_array(image_as_array), batch_size=1)[0]

    def classify_image_from_file(self, image_file_path):
        image = load_img(image_file_path, color_mode='grayscale', target_size=self.target_size)
        return self.classify_image_from_array(img_to_array(image))

    def __prepare_image_array(self, image_as_array):
        return np.expand_dims(
            Classifier.__get_normalized_image_array(
                self.__get_resized_image_array(image_as_array)),
            axis=0)

    def __get_resized_image_array(self, image_as_array):
        shape = image_as_array.shape
        if (shape[0], shape[1]) == self.target_size:
            return image_as_array

        return img_to_array(array_to_img(image_as_array, scale=False).resize(self.image_size))

    @staticmethod
    def __get_normalized_image_array(image_as_array):
        if np.amax(image_as_array) > 1.0:
            image_as_array /= 255.0

        return image_as_array
