import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset( dataset, look_back=1 ):
	input_dataset = []
	output_dataset = []
	for i in range(len(dataset) - look_back):
		input_dataset.append(dataset[i:i+look_back, 0])
		output_dataset.append(dataset[i+look_back, 0])
	return numpy.array(input_dataset), numpy.array(output_dataset)





numpy.random.seed(42)

dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]
print("Splitted dataset into %i training episodes and %i testing episodes." % (train_size, test_size))

look_back = 1
train_input, train_output = create_dataset(train, look_back)
test_input, test_output = create_dataset(test, look_back)

train_input = train_input.reshape(train_input.shape[0], 1, train_input.shape[1])
test_input = test_input.reshape(test_input.shape[0], 1, test_input.shape[1])

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_input, train_output, epochs=100, batch_size=1, verbose=2)

train_predict = model.predict(train_input)
test_predict = model.predict(test_input)

train_predict = scaler.inverse_transform(train_predict)
train_output = scaler.inverse_transform([train_output])
test_predict = scaler.inverse_transform(test_predict)
test_output = scaler.inverse_transform([test_output])

train_score = math.sqrt(mean_squared_error(train_output[0], train_predict[:,0]))
test_score = math.sqrt(mean_squared_error(test_output[0], test_predict[:,0]))

print("Train Score: %.2f" % train_score)
print("Test Score: %.2f" % test_score)

train_predict_plot = numpy.empty_like(dataset)
train_predict_plot[:,:] = numpy.nan
train_predict_plot[look_back:look_back+len(train_predict),:] = train_predict

test_predict_plot = numpy.empty_like(dataset)
test_predict_plot[:,:] = numpy.nan
test_predict_plot[len(train_predict)+look_back*2:len(dataset),:] = test_predict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()
