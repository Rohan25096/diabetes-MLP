import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

model = Sequential()

#Declaring the first layer(Input layer)
model.add(Dense(units=128,activation='relu',input_dim=8,kernel_regularizer=tensorflow.keras.regularizers.l2(0.03)))
model.add(Dropout(0.3))

#Declaring the second layer(Hidden layer)
model.add(Dense(units=128,activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l2(0.03)))
model.add(Dropout(0.3))

#Declaring the third layer(Output layer)
model.add(Dense(units=1,activation='tanh'))

model.summary()

adam = Adam(learning_rate = 0.0002)
model.compile(loss='binary_crossentropy' ,optimizer='Adam',metrics=['accuracy'])

callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    verbose = 1,
    restore_best_weights = True
)

#Training the neural network(MLP)
history = model.fit(X_train_scaled,y_train,epochs=500,validation_split=0.2,callbacks=callback)

#Outputs on the test data(range 0 to 1)
y_log = model.predict(X_test_scaled)