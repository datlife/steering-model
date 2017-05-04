from keras.layers import Input
from keras.layers import Convolution2D, Dense, Lambda, Dropout
from keras.layers import BatchNormalization, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from PARAMS import *
from data_generator import image_generator
import time


def custom_loss(y_true, y_pred):
    """
    @TODO: Custom loss should show relationship between speed<-->steering, throttle
        This loss function adds some constraints on the angle to
        keep it small, if possible
    """
    return K.mean(K.abs(y_pred - y_true), axis=-1)  # +.01* K.mean(K.square(y_pred), axis = -1)


class PosNet(object):
    def __init__(self, img_shape):
        self.model = None
        self.img_shape = img_shape

        # Construct PosNet Model
        self.build(self.img_shape)

    def build(self, img_shape):
        if self.model is not None:
            print("PosNet has been constructed")
        else:
            img_input = Input(shape=(img_shape[0], img_shape[1], img_shape[2]), name='inputImg')
            x_conv = Convolution2D(24, 8, 8, border_mode='valid', subsample=(2, 2), name='conv1')(img_input)
            x_conv = BatchNormalization()(x_conv)
            x_conv = Activation('elu')(x_conv)
            print(x_conv.get_shape())
            x_conv = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), name='conv2')(x_conv)
            x_conv = BatchNormalization()(x_conv)
            x_conv = Activation('elu')(x_conv)
            print(x_conv.get_shape())
            x_conv = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), name='conv3')(x_conv)
            x_conv = BatchNormalization()(x_conv)
            x_conv = Activation('elu')(x_conv)
            print(x_conv.get_shape())
            x_conv = Convolution2D(64, 5, 5, border_mode='valid', name='conv4')(x_conv)
            x_conv = BatchNormalization()(x_conv)
            x_conv = Activation('elu')(x_conv)
            print(x_conv.get_shape())
            x_conv = Convolution2D(64, 5, 5, border_mode='valid', name='conv5', )(x_conv)
            x_conv = BatchNormalization()(x_conv)
            x_conv = Activation('elu')(x_conv)
            print(x_conv.get_shape())
            x_out = Flatten()(x_conv)
            print(x_out.get_shape())

            # Cut for transfer learning is here:
            speed_input = Input(shape=(1,), name='inputSpeed')

            x_out = Lambda(lambda x: K.concatenate(x, axis=1))([x_out, speed_input])
            x_out = Dense(200)(x_out)
            x_out = BatchNormalization()(x_out)
            x_out = Activation('elu')(x_out)
            x_out = Dense(200)(x_out)
            x_out = BatchNormalization()(x_out)
            x_end = Activation('elu')(x_out)

            # Branching from X_END to three branches (steer, throttle, position)
            steer = Dense(100)(x_end)
            steer = BatchNormalization()(steer)
            steer = Activation('elu')(steer)
            steer = Dropout(.2)(steer)
            steer = Dense(30)(steer)
            steer = BatchNormalization()(steer)
            steer = Activation('elu')(steer)
            steer = Dense(1, activation='sigmoid')(steer)
            steer = Lambda(lambda x: x * 10 - 5, name='outputSteer')(steer)

            throttle = Dense(100, name='thr1')(x_end)
            throttle = BatchNormalization(name='thr2')(throttle)
            throttle = Activation('elu')(throttle)
            throttle = Dropout(.2)(throttle)
            throttle = Dense(30, name='thr3')(throttle)
            throttle = BatchNormalization(name='thr4')(throttle)
            throttle = Activation('elu')(throttle)
            throttle = Dense(1, activation='sigmoid', name='thr5')(throttle)
            throttle = Lambda(lambda x: x * 2 - 1, name='outputThr')(throttle)

            position = Dropout(.3)(x_end)
            position = Dense(1, activation='sigmoid', name='pos5')(position)
            position = Lambda(lambda x: x * 2 - 1, name='outputPos')(position)
            self.model = Model((img_input, speed_input), (steer, throttle, position))

    def train(self, x_train, x_val, batch_size=128, lr=LEARN_RATE, epochs=1):

        self.model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse'])
        # @TODO: complete image_generator from path (img paths in csv file)

        # Image generators from csv file and img dir
        train_generator = image_generator(x_train, batch_size, self.img_shape, outputShape=[3], is_training=True)
        val_generator   = image_generator(x_val, batch_size, self.img_shape, outputShape=[3], is_training=False)

        # For backup model
        stop_callback = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.01)
        check_callback = ModelCheckpoint('psyncModel.ckpt', monitor='val_loss', save_best_only=True)
        vis_callback = TensorBoard(log_dir='./logs/%d' % int(time.time()), histogram_freq=0, write_graph=True,
                                   write_images=True)
        # Start training
        self.model.fit_generator(train_generator,
                                 callbacks=[stop_callback, check_callback, vis_callback],
                                 nb_epoch=40,
                                 samples_per_epoch=epochs,
                                 max_q_size=24,
                                 validation_data=val_generator,
                                 nb_val_samples=len(x_val),
                                 nb_worker=8,
                                 pickle_safe=True)

        self.model.load_weights('psyncModel.ckpt')
        self.model.save("Test")


