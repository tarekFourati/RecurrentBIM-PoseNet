import gc

import numpy as np
import tensorflow.keras as keras
import cv2


def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = img[int(height_offset):int(height_offset + output_side_length),
                  int(width_offset):int(width_offset + output_side_length)]
    return cropped_img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, directory, file_name, window, mean=None, image_size=(224, 224, 3), batch_size=8, shuffle=True,
                 random_state=42):
        'Initialization'
        self.epoch = 0
        self.directory = directory
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.labels = []
        self.images_path = []
        self.image_size = image_size
        with open(directory + file_name) as f:
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.labels.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(directory + fname)
        # images_out_train = preprocess_train(images_train)
        self.labels = np.array(self.labels)
        self.data_length = len(self.labels)
        self.sequence_length = (self.data_length - self.window + 1)
        self.indexes = list(range(self.sequence_length))
        if mean is None:
            self.mean = self.calculate_mean()
        else:
            self.mean = mean

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.sequence_length / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        images, labels = self.__data_generation(indexes)
        X = np.zeros((len(indexes), self.window, *self.image_size), dtype=np.float64)
        Y = np.zeros((len(indexes), self.window, 7), dtype=np.float64)
        j = 0
        for i in indexes:
            for k in range(self.window):
                X[j, k, :, :, :] = images[i + k]
                Y[j, k, :] = labels[i + k]
            j = j + 1

        return X, [Y[:, :, 0:3], Y[:, :, 3:], Y[:, :, 0:3], Y[:, :, 3:], Y[:, :, 0:3], Y[:, :, 3:]]

    def on_epoch_end(self):
        gc.collect()
        self.epoch += 1
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        images = {}
        labels = {}
        for index in indexes:
            for k in range(index, self.window + index):
                if k in images:
                    continue
                labels[k] = self.labels[k]
                image_path = self.images_path[k]
                X = cv2.imread(image_path)
                X = cv2.resize(X, (320, 240))
                X = centeredCrop(X, 224)
                # Subtract mean from all images
                X = np.transpose(X, (2, 0, 1))
                X = X - self.mean
                X = np.squeeze(X)
                X = np.transpose(X, (1, 2, 0))
                Y = np.expand_dims(X, axis=0)
                images[k] = Y

        return images, labels

    def calculate_mean(self):
        mean = np.zeros((1, 3, 224, 224))
        N = 0
        for image_path in self.images_path:
            X = cv2.imread(image_path)
            X = cv2.resize(X, (320, 240))
            X = centeredCrop(X, 224)
            # compute images mean
            mean[0][0] += X[:, :, 0]
            mean[0][1] += X[:, :, 1]
            mean[0][2] += X[:, :, 2]
            N += 1
        return mean[0] / N
