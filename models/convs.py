from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Flatten, Activation, MaxPooling2D, BatchNormalization, Dropout

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


wmodel = Sequential()
wmodel.add(Conv2D(32, (3,3), padding='same', input_shape=(127,16,3)))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(Conv2D(32, (3,3), padding='same'))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(MaxPooling2D(pool_size=(2,2)))
wmodel.add(Dropout(0.2))
 
wmodel.add(Conv2D(64, (3,3), padding='same'))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(Conv2D(64, (3,3), padding='same'))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(MaxPooling2D(pool_size=(2,2)))
wmodel.add(Dropout(0.3))
 
wmodel.add(Conv2D(128, (3,3), padding='same'))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(Conv2D(128, (3,3), padding='same'))
wmodel.add(Activation('elu'))
wmodel.add(BatchNormalization())
wmodel.add(MaxPooling2D(pool_size=(2,2)))
wmodel.add(Dropout(0.4))
 
wmodel.add(Flatten())
wmodel.add(Dense(10, activation='softmax'))

wmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
print(wmodel.summary())
