Driving Behavioral Cloning
==========================
**Status**: Completed

This project is to mimic human driving behavior into a car. 

### Dependencies

This project requires users to have additional libraries installed in order to use. Udacity provided a good solution by sharing  a ready-to-use environment `CarND-Term1-Starter-Kit` for students. Please refer [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) on how to install.


### How to use:
1. Download CarND Udacity Simulator ([here]())

2. Open car simulator in autonomous mode

3. Open a terminal, run `python drive.py model/cnn.h5` (cnn.h5 is my pre-trained model)

4. Enjoy your first autonomous drive!

### Result:

| Track 1       | Track 2       | 
| ------------- |---------------|
| ![alt text](https://github.com/dat-ai/behavioral-cloning/blob/master/docs/track1.gif)      | ![alt text](https://github.com/dat-ai/behavioral-cloning/blob/master/docs/track2.gif)|


    
## 1. Deep ResNet Pre-Activation works well
-------------------------------------------
### 1.1 Network Architecture Considerations

My appoarch is to try to minimize the amount of parameters low while retaining the accuracy as high as possible. Many suggests to use [NVIDIA End-to-End Learning For Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) since it is provenly well-suited for this problem. However, I want to explore something new. 

The input of this problem is temporal input. Recurrent Neural Network actually should be applied to this problem. In fact, the [winner](https://github.com/udacity/self-driving-car/tree/master/challenges/challenge-2) of the Udacity Challenge 2 used an LSTM + CNN to train his model. I tried to implemented it in `DatNet.py`, however, it was hard to train due to exploding vanishing gradient or I might not know how to train it properly yet. In theory, this should work better than a Convolutional Neural Networks.

As limited of time resources, I decided to switch back to CNN. I found an intersting paper by [`He et all`](https://arxiv.org/pdf/1603.05027.pdf). Basically, they did not use traditional Conv--->ReLU-->MaxPool. Instead, they eliminated MaxPool and performed BatchNorm before Activation along with `ResNet` architecture. So it would be something like following

![alt](https://github.com/dat-ai/behavioral-cloning/raw/master/docs/resnet_preact.PNG)

Before fully connected layer, they did avergage pooling. This significantly reduces the amount of parameters.  I tried to implement it. Woolah, it provided me much  better result.

Here is my Network architecture.

```shell
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cifar (InputLayer)               (None, 46, 160, 3)    0                                            
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 46, 160, 3)    0           cifar[0][0]                      
____________________________________________________________________________________________________
conv0 (Convolution2D)            (None, 23, 80, 32)    2400        lambda_1[0][0]                   
____________________________________________________________________________________________________
bn0 (BatchNormalization)         (None, 23, 80, 32)    92          conv0[0][0]                      
____________________________________________________________________________________________________
relu0 (Activation)               (None, 23, 80, 32)    0           bn0[0][0]                        
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 23, 80, 32)    0           relu0[0][0]                      
____________________________________________________________________________________________________
conv1a (Convolution2D)           (None, 23, 80, 32)    1024        dropout_1[0][0]                  
____________________________________________________________________________________________________
bn1b (BatchNormalization)        (None, 23, 80, 32)    92          conv1a[0][0]                     
____________________________________________________________________________________________________
relu1b (Activation)              (None, 23, 80, 32)    0           bn1b[0][0]                       
____________________________________________________________________________________________________
conv1b (Convolution2D)           (None, 23, 80, 32)    9216        relu1b[0][0]                     
____________________________________________________________________________________________________
bn1c (BatchNormalization)        (None, 23, 80, 32)    92          conv1b[0][0]                     
____________________________________________________________________________________________________
relu1c (Activation)              (None, 23, 80, 32)    0           bn1c[0][0]                       
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 23, 80, 32)    0           relu1c[0][0]                     
____________________________________________________________________________________________________
conv1c (Convolution2D)           (None, 23, 80, 32)    1056        dropout_2[0][0]                  
____________________________________________________________________________________________________
+1 (Merge)                       (None, 23, 80, 32)    0           conv1c[0][0]                     
                                                                   dropout_1[0][0]                  
____________________________________________________________________________________________________
bn2a (BatchNormalization)        (None, 23, 80, 32)    92          +1[0][0]                         
____________________________________________________________________________________________________
relu2a (Activation)              (None, 23, 80, 32)    0           bn2a[0][0]                       
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 23, 80, 32)    0           relu2a[0][0]                     
____________________________________________________________________________________________________
conv2a (Convolution2D)           (None, 23, 80, 32)    1024        dropout_3[0][0]                  
____________________________________________________________________________________________________
bn2b (BatchNormalization)        (None, 23, 80, 32)    92          conv2a[0][0]                     
____________________________________________________________________________________________________
relu2b (Activation)              (None, 23, 80, 32)    0           bn2b[0][0]                       
____________________________________________________________________________________________________
conv2b (Convolution2D)           (None, 23, 80, 32)    9216        relu2b[0][0]                     
____________________________________________________________________________________________________
bn2c (BatchNormalization)        (None, 23, 80, 32)    92          conv2b[0][0]                     
____________________________________________________________________________________________________
relu2c (Activation)              (None, 23, 80, 32)    0           bn2c[0][0]                       
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 23, 80, 32)    0           relu2c[0][0]                     
____________________________________________________________________________________________________
conv2c (Convolution2D)           (None, 23, 80, 32)    1056        dropout_4[0][0]                  
____________________________________________________________________________________________________
+2 (Merge)                       (None, 23, 80, 32)    0           conv2c[0][0]                     
                                                                   +1[0][0]                              
____________________________________________________________________________________________________
bnF (BatchNormalization)         (None, 23, 80, 32)    92          +2[0][0]                         
____________________________________________________________________________________________________
reluF (Activation)               (None, 23, 80, 32)    0           bnF[0][0]                        
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 23, 80, 32)    0           reluF[0][0]                      
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1.0, 3.0, 32)  0           dropout_9[0][0]                  
____________________________________________________________________________________________________
flat (Flatten)                   (None, 96.0)          0           avg_pool[0][0]                   
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 1024)          99328       flat[0][0]                       
____________________________________________________________________________________________________
dropout_10 (Dropout)             (None, 1024)          0           fc1[0][0]                        
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 512)           524800      dropout_10[0][0]                 
____________________________________________________________________________________________________
dropout_11 (Dropout)             (None, 512)           0           fc2[0][0]                        
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 256)           131328      dropout_11[0][0]                 
____________________________________________________________________________________________________
dropout_12 (Dropout)             (None, 256)           0           fc3[0][0]                        
____________________________________________________________________________________________________
output_1 (Dense)                 (None, 3)             771         dropout_12[0][0]                 
====================================================================================================
Total params: 805,007
Trainable params: 804,409
Non-trainable params: 598
```


### 1.2 Future goal, Recurrent Neural Network + CNN

In the future, I would like to use recurrent neural network for this project. The reason is every change in this world is respected to time domain. Personally, it is not intuitive to use static image for data changing over time. Nevertheless, I am satisfied with my final result. In the next section, I will dive into some tips and tricks that I used to tackle the project.

## 2. Data Augmentation
-----------------------

### 2.1 OpenCV is wonderful
The goal of data augmentation is to assist the model generalize better. In this project, I re-used to image tools from project 2, which take an image and perform multiple transformations(blurring, rotation and chaning brightness).
```shell
def random_transform(img):
    # There are total of 3 transformation
    # I will create an boolean array of 3 elements [ 0 or 1]
    a = np.random.randint(0, 2, [1, 3]).astype('bool')[0]
    if a[1] == 1:
        img = rotate(img)
    if a[2] == 1:
        img = blur(img)
    if a[3] == 1:
        img = gamma(img)
    return img
```
### 2.2 Flip that image!

You might found that during data collection you might be unconsciously biased toward one side of street. So flipping the image helps your model generalize better. As suggested by Udacity, driving in opposite direction also helps your model. The reason is the lap has too many left turns. By driving in reversed direction, you force your model to learn the right turn too. 
```shell
# #############################
# ## DATA AUGMENTATION ########
###############################

from utils.image_processor import random_transform
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)

    flipped_angle = measurement[0] * -1.0
    augmented_measurements.append((flipped_angle, measurement[1], measurement[2]))
    # #
    rand_image = random_transform(image)
    augmented_images.append(rand_image)
    augmented_measurements.append(measurement)
```

## 3. Training strategies
-------------------------

* In this particular project, training goal is to minimize the loss (mean root square errors) of the steering angle. In my labels, I had `[steering_angle, throttle, speed]` (for my future RNN's model), I had to write a custom loss function as following:
```shell
def mse_steer_angle(y_true, y_pred):
    ''' Custom loss function to minimize Loss for steering angle '''
    return mean_squared_error(y_true[0], y_pred[0])
```
* In order to use this custom loss function, I applied my loss during the compilation of my model:
```shell
 # Compile model
 model.compile(optimizer=Adam(lr=learn_rate), loss=[mse_steer_angle])
```
### 3.1 Becareful to high learning rate
![alttext](https://github.com/dat-ai/behavioral-cloning/raw/master/docs/alpha2.png)

Another issue is I initially set my training rate to `0.001` as many suggested. However, it did not work well for me. My loss kept fluctuating. So I decided to go lower learning rate `0.0001` and it worked. Therefore, if you see your loss flucuates during the training process. It is a strong indication that the learning rate might be too high. Try to lower the learning rate.

### 3.2 Know when to stop
One of the mistakes I made during the training process was too focused on minimizing loss for steering angle. What happened to me was if I trained my model too long, it would drive very weird in one map. Therefore, my training strategy is to lower my learning rate and to use early stopping technique. Once my model worked as expected in the simulator, I stopped the training process. If you are not very satisfied to your result. Try use lower learning rate `0.0001` to `0.00001`. Also, saving your model during every epochs could be helpful, too. Here is how I did it.

```shell
from keras.callbacks import ModelCheckpoint

....

checkpoint = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.h5', save_weights_only=True)

...
model.fit_generator(data_augment_generator(images, measurements, batch_size=batch_size), samples_per_epoch=data_size,
                     callbacks=[checkpoint], nb_val_samples=data_size * 0.2, nb_epoch=epochs)
```


## 4. From simulator to real RC racing car

Finally, I would like to advertise my current project [Autonomous 1/10th Self-Racing Car](https://github.com/dat-ai/jetson-car) . I applied  what I learned from Behavioral Cloning into a real RC car. This a wonderful chance to validate my model in real track. My car is used NVIDIA Jetson TK1 as a brain to control the steering servo and ESC (You can used Raspberry Pi 3 but it could be slow).
