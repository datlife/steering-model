import cv2
import os
import numpy as np
from utils.image_processor import random_transform, flip
from PIL import Image
from gps_handler import LatLontoUTM, RadToDeg, NearestWayPointCTEandDistance, total_lap_distance

CSV_HEADER = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']


def image_generator(data, batchSize, inputShape, outputShape, is_training=False):
    """
    The generator function for the training data for the fit_generator
        Input:
    :param data: an pandas data-frame containing the paths to the images, the steering angle,...
    :param batchSize: 
    :param inputShape: 
    :param outputShape: 
    :param is_training: 
    :return: in batch
    """

    while 1:
        img_arr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        speed_arr = np.zeros((batchSize, 1))
        pos_arr = np.zeros((batchSize, 1))
        labels = np.zeros((batchSize, outputShape[0]))

        done = 0
        while done < batchSize:
            indices = np.random.randint(0, len(data), batchSize)
            for i, index in zip(range(len(indices)), indices):
                row = data.iloc[index]
                img_file = row['center'].rsplit('/')[-1]
                file = os.path.join('./data/IMG/', img_file)
                image = cv2.imread(os.path.join('./data/IMG/', img_file))
                if image is None:
                    continue

                label = np.array([row['steer_angle'], row['throttle'], row['brake']])
                speed = row[['speed']].values
                # position = row[['longitude', 'latitude']].values

                # print("Input: {} Holder {}".format(np.shape(image), np.shape(img_arr[i])))
                labels[i] = label
                img_arr[i] = image

                r = np.random.rand()
                if r > .5:
                    image = flip(image)
                    label[0] *= -1

                if is_training:
                    image        = random_transform(image)
                    speed_arr[i] = [val + 0.02 * np.random.rand() - 0.01 for val in speed]
                else:
                    speed_arr[i] = speed

                done += 1
                if done == batchSize:
                    break  # inner loop

                # @TODO: PLUG GPS SENSOR INPUT HERE
                # utm_point = LatLontoUTM(RadToDeg(position[0]), RadToDeg(position[1]))
                #
                # _, _, lapDistance = NearestWayPointCTEandDistance(utm_point)
                # pos_arr[i] = 2 * (lapDistance / total_lap_distance) - 1

        yield ({'inputImg': img_arr, 'inputSpeed': np.array(speed_arr)},
               {'outputSteer': labels[:, 0], 'outputThr': labels[:, 1] - labels[:, 2], 'outputPos': np.array(pos_arr)})


def preprocess(image):
    """
    This function represents the default preprocessing for
    an image to prepare them for the network
    """
    image = cv2.resize(image, (320, 160))[::-1]
    image = image[image.shape[0]*1//5:image.shape[0]*7//8, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = (image - 128.) / 128.
    return image

