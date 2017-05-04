import pandas as pd
import numpy as np
from model import PosNet


CSV_HEADER = ['center', 'left', 'right', 'steer_angle', 'throttle', 'brake', 'speed']
df = pd.read_csv('./data/driving_log.csv', header=None, names=CSV_HEADER, index_col=False)
msk = np.random.rand(len(df)) < 0.8
x_train = df[msk]
x_val   = df[~msk]
print("Training data: {:d} samples\nTesting Data: {:d} samples\n\n".format(len(x_train), len(x_val)))
model = PosNet(img_shape=(160, 320, 3))

model.train(x_train, x_val, batch_size=128, lr=1e-4, epochs=5)
