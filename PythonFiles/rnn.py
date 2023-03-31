import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a ds matrix
def create_ds(ds, look_back=1):
	dataX, dataY = [], []
	for i in range(len(ds)-look_back-1):
		a = ds[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(ds[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed 
numpy.random.seed(7)
# load the ds
df = read_csv('final_ar.csv', usecols=[1], engine='python')
ds = df.values
ds = ds.astype('float32')
# normalizing
scaler = MinMaxScaler(feature_range=(0, 1))
ds = scaler.fit_transform(ds)

trsize = int(len(ds) * 0.67)
tstsize = len(ds) - trsize
train, test = ds[0:trsize,:], ds[trsize:len(ds),:]

look_back = 1
trainX, trainY = create_ds(train, look_back)
testX, testY = create_ds(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# creating and fitting  LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
train_prdn = model.predict(trainX)
test_prdn = model.predict(testX)
# invert predictions
train_prdn = scaler.inverse_transform(train_prdn)
trainY = scaler.inverse_transform([trainY])
test_prdn = scaler.inverse_transform(test_prdn)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], train_prdn[:,0]))
print('Train%: %.2f RMS' % (15*trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], test_prdn[:,0]))
print('Test%: %.2f RMS' % (15*testScore))
# shift train predictions for plotting
train_prdnPlot = numpy.empty_like(ds)
train_prdnPlot[:, :] = numpy.nan
train_prdnPlot[look_back:len(train_prdn)+look_back, :] = train_prdn
# shift test predictions for plotting
test_prdnPlot = numpy.empty_like(ds)
test_prdnPlot[:, :] = numpy.nan
test_prdnPlot[len(train_prdn)+(look_back*2)+1:len(ds)-1, :] = test_prdn
# plot baseline and predictions
plt.plot(scaler.inverse_transform(ds))
plt.plot(train_prdnPlot)
plt.plot(test_prdnPlot)
plt.show()