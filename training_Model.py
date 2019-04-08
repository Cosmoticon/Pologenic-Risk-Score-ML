import subprocess

import numpy as np

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout

from keras.utils.training_utils import multi_gpu_model





def get_gpu_count():

    "get the number of GPUs on this computer"

    res = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)

    res = res.rstrip("\n")

    lines = res.split("\n")

    return len(lines)





x_train = np.loadtxt("x_train.csv", dtype=float, skiprows=1, delimiter=",")

y_train = x_train[:, 0]

x_train = x_train[:, 3:10000]

x_test = np.loadtxt("test_uk.csv", dtype=float, skiprows=1, delimiter=",")

y_test = x_test[:, 0]

x_test = x_test[:, 3:10000]



#model = Sequential()

#model.add(Dense(64, input_dim=16999, activation="relu"))

#model.add(Dropout(0.5))

#model.add(Dense(64, activation="relu"))

#model.add(Dropout(0.5))

#model.add(Dense(1, activation="sigmoid"))



# Convert model to multi-gpu model using the number

# of GPUs available on the computer the code is running on.

encoding_dim = 1000

input_img = Input(shape=(9997,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(9997, activation='sigmoid')(encoded)

decoded = Dense(9997, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,

                epochs=50,

                batch_size=len(x_train),

                shuffle=True,

                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)

#np.vstack((encoded_imgs,decoded_imgs))

#model = multi_gpu_model(model, gpus=get_gpu_count())

#model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])



#model.fit(x_train, y_train, epochs=20, batch_size=128)

#score = model.evaluate(x_test, y_test, batch_size=128)

np.savetxt("score.csv", decoded_imgs)
