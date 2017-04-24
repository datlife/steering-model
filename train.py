from model import PosNet

x_train = None
x_val = None
model = PosNet(img_shape=(100, 100, 3))

model.train(x_train, x_val, batch_size=128, lr=1e-4, epochs=5)
