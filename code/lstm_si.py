# lstm_sd.py: emotion recognition using long short-term memory with speaker independent

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rn
import pandas as pd

np.random.seed(123)
rn.seed(123)
tf.random.set_seed(123)

# load feature data
data_path = '../data/song/' # choose song or speech
x_train = np.load(data_path + 'x_train.npy')
x_test  = np.load(data_path + 'x_test.npy')
y_train = np.load(data_path + 'y_train.npy')
y_test  = np.load(data_path + 'y_test.npy')

# reshape x untuk lstm
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

# if labels are not in integer, convert it, otherwise comment it
y_train = y_train.astype(int)
y_test = y_test.astype(int)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True)

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

# function to define model
def model_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(axis=-1,
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))

    # compile model: set loss, optimizer, metric
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

# create the model
model = model_lstm()
print(model.summary())

# plot model
tf.keras.utils.plot_model(model,'lstm_model_si.pdf',show_shapes=True)

# train the model
hist = model.fit(x_train, 
                 y_train, 
                 epochs=100, 
                 shuffle=True, # mengacak data
                #  callbacks=earlystop, # akan berhenti saat konvergen meskipun belum sampai maksimal iterasi
                 validation_split=0.1, # 1% utk validasi
                 batch_size=16) # setiap 1 kali training ada 16 sampel

# evaluate the model on test partition
evaluate = model.evaluate(x_test, y_test, batch_size=16)
print("Loss={:.6f}, Accuracy={:.6f}".format(evaluate[0],evaluate[1]))

# plot accuracy
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid()
plt.legend(['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('LSTM Model')
plt.savefig('lstm_accuracy_si.svg')

# make prediction for confusion_matrix
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
predict = model.predict(x_test, batch_size=16)
emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']

# predicted emotions from the test set
y_pred = np.argmax(predict, 1)
predicted_emo = []
for i in range(0,y_test.shape[0]):
    emo = emotions[y_pred[i]]
    predicted_emo.append(emo)

# get actual emotion
actual_emo = []
y_true = y_test
for i in range(0,y_test.shape[0]):
    emo = emotions[y_true[i]]
    actual_emo.append(emo)

# generate the confusion matrix
cm = confusion_matrix(actual_emo, predicted_emo)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

index = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
columns = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(10, 6))
plt.title('Confusion Matrix of LSTM')
sns.heatmap(cm_df, annot=True)
plt.savefig('lstm_cm_si.svg')

# print unweighted average recall
print("UAR: ", cm.trace()/cm.shape[0])