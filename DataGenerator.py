import os, csv
from PIL import Image
import numpy as np

BATCHSIZE = 32
root_path = os.path.dirname(__file__)
print(root_path)


class data_generator:

    def __init__(self, file_path, image_size, classes, mode):
        self.load_data(file_path=file_path)
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.classes = classes
        self.mode = mode
        self.load_images_labels()
        self.num_of_examples = self.max_example

    def load_data(self, file_path):
        # with open(file_path, 'r') as f:
        #     self.datasets = f.readlines()
        csv_reader = csv.reader(open(file_path, encoding='utf-8'))
        self.datasets = [row for row in csv_reader]
        self.max_example = len(self.datasets)

    def load_images_labels(self):
        images = []
        labels = []
        for i in range(0, len(self.datasets)):
            data_arr = self.datasets[i]
            image_path = os.path.join(root_path, self.mode, data_arr[0])
            img = Image.open(image_path)
            img = img.resize((self.image_size[0], self.image_size[1]), Image.ANTIALIAS)
            img = np.array(img)
            images.append(img)
            labels.append(np.array(data_arr[1:]))
        self.images = images
        self.labels = labels

    def normalize(self, image):

        mean = np.mean(image)
        var = np.mean(np.square(image - mean))

        image = (image - mean) / np.sqrt(var)

        return image

    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(self.images)):
                    self.index = 0
                # batch_images.append(self.normalize(self.images[self.index]))
                batch_images.append(self.images[self.index])
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels


if __name__ == "__main__":
    txt_path = 'datasets81_clean.txt'
    width, height = 224, 224
    IMAGE_SIZE = (width, height, 3)
    classes = 81
    train_gen = data_generator(txt_path, IMAGE_SIZE, classes)
    x, y = next(train_gen.get_mini_batch())
    print(x.shape)
    print(y.shape)
