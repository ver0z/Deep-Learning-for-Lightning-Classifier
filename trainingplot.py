import pandas as pd
from keras.layers import Dense
from collections import Counter
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE

# classification dataset
raw_data = open('noite - noite.csv', 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x)
# one hot encode output variable
X, Y = data[:, :-1].astype(float), data[:, -1]
print(X.shape, Y.shape)
da = Y
counter = Counter(Y)
print(counter)
for label, _ in counter.items():
	warnings.simplefilter(action='ignore', category=FutureWarning)
	row_ix = np.where(Y == label)[0]
	plt.figure(3)
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
print(X.shape, Y.shape)
counter = Counter(Y)
print(counter)
for label, _ in counter.items():
	warnings.simplefilter(action='ignore', category=FutureWarning)
	row_ix = np.where(Y == label)[0]
	plt.figure(4)
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
Y = to_categorical(Y, num_classes=6)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                                        random_state=1)


print(X_train.shape, X_test.shape, y_test.shape, y_train.shape)
y_verdade = y_test[:, :len(y_test)]
a = [np.where(r == 1)[0][0] for r in y_verdade]


# define model
model = Sequential()
model.add(Dense(128, input_dim=1100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
#opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
esCallback = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=512, epochs=150, verbose=0, callbacks=[esCallback])
# evaluate the model
y_pred = model.predict_classes(X_test)
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
classes=[0, 1, 2, 3, 4, 5]

np.seterr(divide='ignore', invalid='ignore')
con_mat = tf.math.confusion_matrix(labels=a, predictions=y_pred).numpy()
print(a, y_pred)
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)


model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
# model = load_model('my_model.h5')
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plt.style.use("ggplot")
plt.figure(1)
plt.title("Training Loss and Accuracy")
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
# plot accuracy during training
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")


plt.figure(2, figsize=(6, 6))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# pd.read_csv('noite - noite.csv', header=None).T.to_csv('output.csv', header=False, index=False)
r_data = open('output.csv', 'rt')
Xzip = csv.reader(r_data, delimiter=',', quoting=csv.QUOTE_NONE)
xz = list(Xzip)
datra = np.array(xz)
# XT = datra[:-1, :].astype(float)

df = pd.read_csv(('output.csv'))
print(df)
ax = plt.gca()
data_frame = df.sort_index(axis=1 ,ascending=True)
data_frame = data_frame.iloc[::-1]

for i in range(5, 10):
	plt.figure(5)
	plt.title("Campo Elétrico Classe six")
	ax = plt.gca()
	df.plot(label='_Hidden label', kind='line',y=f'six{i}', legend=False, ax=ax)	
plt.legend()
plt.show()
