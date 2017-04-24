import numpy as np
import cv2
from PIL import Image
from loader import LatLontoUTM, RadToDeg, NearestWayPointCTEandDistance, total_lap_distance


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


def image_generator(data, batchSize, inputShape, outputShape, is_training=False):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    while 1:
        returnArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
        speedArr = np.zeros((batchSize, 1))
        vecArr = np.zeros((batchSize, 1))
        labels = np.zeros((batchSize, outputShape[0]))
        weights = np.zeros(batchSize)
        indices = np.random.randint(0, len(data), batchSize)
        for i, index in zip(range(len(indices)), indices):
            row = data.iloc[index]
            file = open(row['path'].strip(), 'rb')
            # Use the PIL raw decoder to read the data.
            #   - the 'F;16' informs the raw decoder that we are reading a little endian, unsigned integer 16 bit data.
            img = np.array(Image.frombytes('RGB', [960, 480], file.read(), 'raw'))
            file.close()

            image = preprocess(img)
            label = np.array([row['steering'], row['throttle'], row['brake']])
            xVector = row[['longitude', 'latitude']].values
            speedVector = row[['speed']].values

            flip = np.random.rand()
            if flip > .5:
                image = mirrorImage(image)
                label[0] *= -1
            if is_training:
                image, label = augmentImage(image, label)
            labels[i] = label

            returnArr[i] = image
            weights[i] = row['norm']
            if is_training:
                speedArr[i] = [val + 0.02 * np.random.rand() - 0.01 for val in speedVector]
            else:
                speedArr[i] = speedVector

            utmp = LatLontoUTM(RadToDeg(xVector[0]), RadToDeg(xVector[1]))
            _, _, lapDistance = NearestWayPointCTEandDistance(utmp)
            vecArr[i] = 2 * (lapDistance / total_lap_distance) - 1
        yield ({'inputImg': returnArr, 'inputSpeed': np.array(speedArr)},
               {'outputSteer': labels[:, 0], 'outputThr': labels[:, 1] - labels[:, 2], 'outputPos': np.array(vecArr)},
               [10 * weights, weights, 10 * weights])
