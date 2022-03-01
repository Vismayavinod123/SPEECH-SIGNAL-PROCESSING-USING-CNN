#Program Code For Spectrogram Conversion
from __future__ import division, print_function
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    sample_rate, samples = wav.read(audio_path)
    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
def dir_to_spectrogram(audio_dir, spectrogram_dir, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
    for file_name in file_names:
    print(file_name)
    audio_path = audio_dir + file_name
    spectogram_path = spectrogram_dir + file_name.replace('.wav', '.png')
    wav_to_spectrogram(audio_path, spectogram_path, spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)

    audio_dir = "E:/dataset-master/recordings/"
    spectrogram_dir = "E:/dataset-master/spectrograms/"
    dir_to_spectrogram(audio_dir, spectrogram_dir)

#Program Code For Train_Test_Split For Speech Recognition
import os
from shutil import copyfile
def separate(source):
    for filename in os.listdir(source):
        first_split = filename.rsplit("_", 1)[1]
        second_split = first_split.rsplit(".", 1)[0]
        if int(second_split) <= 4:
        copyfile(source + "/" + filename, "E:/dataset-master/testing-spectrograms" + "/" + filename)
        else:
        copyfile(source + "/" + filename, "E:/dataset-master/training-spectrograms" + "/" + filename)
if __name__ == '__main__':
    separate("E:/dataset-master/spectrograms")

#Program Code For Train_Test_Split Of Speaker Recognition
import os
from shutil import copyfile
def separate(source):
    for filename in os.listdir(source):
        num = int(filename.rsplit("_")[2].rsplit(".")[0])
        if num <= 4:
        copyfile(source + "/" + filename, "E:/digit-dataset-master/speaker-test" + "/" + filename)
        else:
        copyfile(source + "/" + filename, "E:/dataset-master/speaker-train" + "/" + filename)
separate("E:/dataset-master/speaker_spectrograms")

#Convolutional Neural Network Model For Speech Recognition
import sys
from matplotlib import pyplot
import keras
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense,Dropout
from keras.layers import Flatten,BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64 ,3)))
	model.add(BatchNormalization())

	model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
	model.add(BatchNormalization())
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	return model
model=define_model()

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0, zoom_range=0, horizontal_flip=False, width_shift_range=0.1,height_shift_range=0.1)  # randomly shift images vertically (fraction of total height))

test_datagen = ImageDataGenerator(rescale=1./255)  

train_generator = train_datagen.flow_from_directory('/home/ai33/Downloads/train-spectrograms',target_size=(64, 64), batch_size=50, shuffle=True)

validation_generator = test_datagen.flow_from_directory('/home/ai33/Downloads/test-spectrograms',target_size=(64, 64),batch_size=50, shuffle=True)
#print validation_generator

history = model.fit_generator(train_generator,steps_per_epoch=36,verbose=1, validation_data=validation_generator,validation_steps=4,epochs=100)
#print(model.summary())
test_generator = test_datagen.flow_from_directory('/home/ai33/Downloads/test',target_size=(64,64),batch_size=1,shuffle=False)
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Speech Recognition On Own Recorded Data

test_generator = test_datagen.flow_from_directory('/home/ai33/Downloads/test2',target_size=(64, 64),shuffle = False,batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

#predict = model.predict_generator(test_generator,steps = nb_samples)
#print predict

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1, steps=nb_samples)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
"Predictions":predictions})
print (results)




#Convolutional Neural Network Model For Speaker Recognition
import sys
from matplotlib import pyplot
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense,Dropout
from keras.layers import Flatten,BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64 ,3)))
	model.add(BatchNormalization())

	model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
    model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(4, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	return model

model=define_model()

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0, zoom_range=0, horizontal_flip=False, width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width), 
height_shift_range=0.1)  # randomly shift images vertically (fraction of total height))

test_datagen = ImageDataGenerator(rescale=1./255)  

train_generator = train_datagen.flow_from_directory('/home/ai33/Downloads/speaker-train-class',target_size=(64, 64), batch_size=50, shuffle=True)
validation_generator = test_datagen.flow_from_directory('/home/ai33/Downloads/speaker-test-class',target_size=(64, 64),batch_size=50, shuffle=True)
#print validation_generator
history = model.fit_generator(train_generator,steps_per_epoch=36,verbose=1, validation_data=validation_generator,validation_steps=4,epochs=100)
print(model.summary())
# list all data in history
print(history.history.keys())

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


