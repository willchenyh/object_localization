import keras
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Dense
import numpy as np

inputs = Input(shape=(2,))
out = Dense(2, activation='tanh')(inputs)
model = Model(inputs=inputs, outputs=out)
model.summary()

sample_size = 10000
x = np.random.randint(0, 6, size=(sample_size,2)) / 10.0
# print x
y = np.zeros((sample_size,2))
y[:,0] = np.dot(x, np.ones((2,1))).reshape(sample_size,)
k2 = np.ones((2,1))
k2[1] = -1
y[:,1] = np.dot(x, k2).reshape(sample_size,)

# print y
'''
for i in range(100):
    print x[i,:], y[i,:]
'''
sgd = optimizers.SGD(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(x, y, epochs=1000, validation_split=0.2)
result = model.evaluate(x,y)
print
print result

new = np.random.randint(0, 6, size=(10,2)) / 10.0
print new
print model.predict(new)

