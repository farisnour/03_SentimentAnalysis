import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from keras.datasets import imdb

from sklearn.metrics import classification_report

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 180
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

for hidden_size in [20, 50, 100, 200, 500]:
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, hidden_size))
    model.add(SimpleRNN(hidden_size))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_test, y_test),
              verbose=0)
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score for %d dimension:' % hidden_size, score)
    print('Test accuracy for %d dimension:' % hidden_size, acc)

    # predicting the testing dataframe
    prediction = model.predict_classes(x_test)

    print(classification_report(prediction, y_test))




