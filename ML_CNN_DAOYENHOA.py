
import pandas as pd
import numpy as np
import tensorflow as tf
import random

#Processing Data (Zip and Define)
import os
import zipfile
from PIL import Image
from PIL import TiffImagePlugin
TiffImagePlugin.DEBUG = True
import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as img


#CNN
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,BatchNormalization
from tensorflow import keras
from keras.models import Sequential
from keras import callbacks


#Crossvalidation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#Importdata
#Importdata
zip = zipfile.ZipFile('/Users/salty/Desktop/ML/CatsDogs.zip', 'r') 
zip.extractall('cd')
zip.close()
print('Dogs file has' , len(os.listdir('./cd/CatsDogs/Dogs')), 'photos')
print('Cats file has' , len(os.listdir('./cd/CatsDogs/Cats')), 'photos')
os.remove('./cd/CatsDogs/Cats/666.jpg')
os.remove('./cd/CatsDogs/Dogs/11702.jpg')

#Convert - Rescale
for folder_name in ("Cats", "Dogs"):
    folder_path = os.path.join("./cd/CatsDogs", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fpath.endswith('.jpg'):
            image = Image.open(fpath)
            image = image.convert('RGB') 
            image.save(fpath)

Dogs = glob.glob('./cd/CatsDogs/Dogs/*.jpg')
Cats = glob.glob('./cd/CatsDogs/Cats/*.jpg')


#Dataframe
dfdogs = pd.DataFrame(Dogs, columns=["image"])
dfdogs["label"] = "dog"
dfcats = pd.DataFrame(Cats, columns=["image"])
dfcats["label"] = "cat"
df = pd.concat([dfdogs, dfcats],  ignore_index=True)
df = df.sample(frac = 1, random_state = 42).reset_index(drop=True)
df

#Splitting
trainval, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(trainval, test_size=0.2, random_state = 42)
print('Train has' , len(train), 'photos')
print('Val has' , len(val), 'photos')
print('Test has' , len(test), 'photos')

generator = ImageDataGenerator(rescale=1/255.)
traingenerator = generator.flow_from_dataframe(dataframe = train,
                                               directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 

valgenerator = generator.flow_from_dataframe(dataframe = val,
                                               directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 



testgenerator = generator.flow_from_dataframe(dataframe = test,
                                              directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True)

model = tf.keras.models.Sequential()


#input layer
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32, 32,3)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding='valid')) 
model.add(Conv2D(64,(3, 3), padding ='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='valid')) 


#Flatten
model.add(Flatten())

#Fully-Connected Layer/hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#Compile
model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer='rmsprop', metrics=['accuracy'])

model.summary()
tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96
)

hist = model.fit(traingenerator, epochs=10, validation_data=valgenerator)

plt.plot(hist.history["accuracy"],label = "Train", color = "blue")
plt.plot(hist.history["val_accuracy"],label = "Validation", color = "red", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 1: Training vs Test Accuracy", color = "darkred", size = 13)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()

plt.figure()
plt.plot(hist.history["loss"],label = "Train", color = "blue")
plt.plot(hist.history["val_loss"],label = "Validation", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model Loss", color = "darkred", size = 13)
plt.legend()
plt.show()

model.evaluate(testgenerator)

#CNN
model2 = tf.keras.models.Sequential()

#input layer
model2.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model2.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size = (2,2), padding='valid')) 
model2.add(Conv2D(64,(3, 3), padding ='same', activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2), padding='valid')) 
model2.add(Conv2D(128,(3, 3), padding ='same', activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2), padding='valid')) 


#Flatten
model2.add(Flatten())

#Fully-Connected Layer/hidden layer
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))


#Compile
model2.compile(loss=tf.losses.BinaryCrossentropy(), optimizer="rmsprop", metrics=['accuracy'])

model2.summary()
tf.keras.utils.plot_model(
    model2,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96
)

hist2 = model2.fit(traingenerator, epochs=10, validation_data=valgenerator)

#Plot accuracy
plt.plot(hist2.history["accuracy"],label = "Train", color = "blue")
plt.plot(hist2.history["val_accuracy"],label = "Validation", color = "red", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 2: Training vs Test Accuracy", color = "blue", size = 13)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()

#Plot loss
plt.figure()
plt.plot(hist2.history["loss"],label = "Train", color = "blue")
plt.plot(hist2.history["val_loss"],label = "Validation", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 2: Training vs Test Loss", color = "blue", size = 13)
plt.legend()
plt.show()

#Evaluate 2nd model 
model2.evaluate(testgenerator)

model3 = tf.keras.models.Sequential()
model3.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 activation='relu', input_shape=(32, 32, 3)))

#### Convolutional Layers ####
model3.add(Conv2D(32, (3,3), activation='relu'))
model3.add(MaxPooling2D((2,2)))  
model3.add(Dropout(0.2)) 


model3.add(Conv2D(64, (3,3), activation='relu'))
model3.add(MaxPooling2D((2,2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(128,(3,3 ), activation='relu'))
model3.add(Activation('relu'))
model3.add(MaxPooling2D((2,2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(256,(3, 3), padding ='same', activation = 'relu'))
model3.add(MaxPooling2D(pool_size = (2, 2), padding='valid')) 
model3.add(Dropout(0.2))

#### Fully-Connected Layer ###
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid'))

#Compile
model3.compile(loss=tf.losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy'])
model3.summary()
tf.keras.utils.plot_model(
    model3,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96
)

hist3 = model3.fit(traingenerator, epochs=10, validation_data=valgenerator)

#Plot accuracy
plt.plot(hist3.history["accuracy"],label = "Train", color = "blue")
plt.plot(hist3.history["val_accuracy"],label = "Validation", color = "red", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 3: Training vs Test Accuracy", color = "blue", size = 13)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()

#Plot loss
plt.figure()
plt.plot(hist3.history["loss"],label = "Train", color = "blue")
plt.plot(hist3.history["val_loss"],label = "Validation", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 3: Training vs Test Loss", color = "blue", size = 13)
plt.legend()
plt.show()

#Evaluate 3rd model 
model3.evaluate(testgenerator)

model4 = tf.keras.models.Sequential()
model4.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 activation='relu', input_shape=(32, 32, 3)))

#### Convolutional Layers ####
model4.add(Conv2D(64, (3,3), activation='relu'))
model4.add(BatchNormalization())
model4.add(MaxPooling2D((2,2),padding='valid'))
model4.add(Dropout(0.4))

model4.add(Conv2D(128,(3,3 ), activation='relu'))
model4.add(BatchNormalization())
model4.add(Activation('relu'))
model4.add(MaxPooling2D((2,2),padding='valid'))
model4.add(Dropout(0.4))

model4.add(Conv2D(256, (3,3), activation='relu'))
model4.add(BatchNormalization())
model4.add(MaxPooling2D((2,2),padding='valid'))  
model4.add(Dropout(0.4)) 

model4.add(Conv2D(512,(3, 3), padding ='same', activation = 'relu'))
model4.add(BatchNormalization())
model4.add(MaxPooling2D((2,2), padding='valid')) 
model4.add(Dropout(0.4))


#### Fully-Connected Layer ###
model4.add(Flatten())
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.4))
model4.add(Dense(1, activation='sigmoid'))

#Compile
model4.compile(loss=tf.losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy'])
model4.summary()
tf.keras.utils.plot_model(
    model4,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96
)

hist4 = model4.fit(traingenerator, epochs=10, validation_data=valgenerator)
#Plot accuracy
plt.plot(hist4.history["accuracy"],label = "Train", color = "blue")
plt.plot(hist4.history["val_accuracy"],label = "Validation", color = "red", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 4: Training vs Test Accuracy", color = "blue", size = 13)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend()

#Plot loss
plt.figure()
plt.plot(hist4.history["loss"],label = "Train", color = "blue")
plt.plot(hist4.history["val_loss"],label = "Validation", color = "darkred", linestyle="dashed",markeredgecolor = "purple", markeredgewidth = 2)
plt.title("Model 4: Training vs Test Loss", color = "blue", size = 13)
plt.legend()
plt.show()
result4 = model4.evaluate(testgenerator)



#model configuration
nfolds = 5
kf = KFold(nfolds, shuffle=True)

acc_per_fold = []
loss_per_fold = []
losshist=[]
acchist=[]
vallosthist=[]
valacchist=[]
k = 1

for train, val in kf.split(df):
  trainkf, testkf = df.iloc[train], df.iloc[val]
  traingeneratorkf = generator.flow_from_dataframe(dataframe = trainkf,
                                               directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 

  testgeneratorkf = generator.flow_from_dataframe(dataframe = testkf,
                                              directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {k} ...')
  hist5 = model.fit(traingeneratorkf, epochs=5, validation_data=testgeneratorkf)
  losshist.append(hist5.history['loss'])
  vallosthist.append(hist5.history['val_loss'])
  acchist.append(hist5.history['accuracy'])
  valacchist.append(hist5.history['val_accuracy'])

  # Generate generalization metrics
  scores = model.evaluate(testgeneratorkf)
  print(f'Score for fold {k}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  k = k + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

colorset = ["#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

#Plotloss
plt.figure()
for i in range(len(losshist)):
    plt.plot(losshist[i], label=f'Fold {i+1} train',color=colorset[i])
    plt.plot(vallosthist[i], label=f'Fold {i+1} val', color=colorset[i],linestyle="dashed")


plt.title("Kfold: Training vs Validation Loss", color = "blue", size = 13)
plt.legend()
plt.show()

#Plotaccuracy
plt.figure()
for i in range(len(acchist)):
    plt.plot(acchist[i], label=f'Fold {i+1} train',color=colorset[i])
    plt.plot(valacchist[i], label=f'Fold {i+1} val', color=colorset[i],linestyle="dashed")


plt.title("Kfold: Training vs Validation Accuracy", color = "blue", size = 13)
plt.legend()
plt.show()

acc_per_fold2 = []
loss_per_fold2 = []
losshist2=[]
acchist2=[]
vallosthist2=[]
valacchist2=[]
k = 1

for train, val in kf.split(df):
  trainkf, testkf = df.iloc[train], df.iloc[val]
  traingeneratorkf = generator.flow_from_dataframe(dataframe = trainkf,
                                               directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 

  testgeneratorkf = generator.flow_from_dataframe(dataframe = testkf,
                                              directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {k} ...')
  hist6 = model4.fit(traingeneratorkf, epochs=5, validation_data=testgeneratorkf)
  losshist2.append(hist6.history['loss'])
  vallosthist2.append(hist6.history['val_loss'])
  acchist2.append(hist6.history['accuracy'])
  valacchist2.append(hist6.history['val_accuracy'])

  # Generate generalization metrics
  scores = model4.evaluate(testgeneratorkf)
  print(f'Score for fold {k}: {model4.metrics_names[0]} of {scores[0]}; {model4.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold2.append(scores[1] * 100)
  loss_per_fold2.append(scores[0])

  # Increase fold number
  k = k + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold2)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold2[i]} - Accuracy: {acc_per_fold2[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold2)} (+- {np.std(acc_per_fold2)})')
print(f'> Loss: {np.mean(loss_per_fold2)}')
print('------------------------------------------------------------------------')

#Plotloss
plt.figure()
for i in range(len(losshist2)):
    plt.plot(losshist2[i], label=f'Fold {i+1} train',color=colorset[i])
    plt.plot(vallosthist2[i], label=f'Fold {i+1} val', color=colorset[i],linestyle="dashed")


plt.title("Kfold: Training vs Validation Loss", color = "blue", size = 13)
plt.legend()
plt.show()

#Plotaccuracy
plt.figure()
for i in range(len(acchist)):
    plt.plot(acchist2[i], label=f'Fold {i+1} train',color=colorset[i])
    plt.plot(valacchist2[i], label=f'Fold {i+1} val', color=colorset[i],linestyle="dashed")


plt.title("Kfold: Training vs Validation Accuracy", color = "blue", size = 13)
plt.legend()
plt.show()

#Hyperparameters tuning 
from keras_tuner.tuners import RandomSearch

# Define the model-building function
def build_model(hp):
    model5 = keras.Sequential()

    # Add the convolutional layers with hyperparameter search space
    #Conv1
    model5.add(keras.layers.Conv2D(
        filters=hp.Int("filters_1", min_value=32, max_value=64, step=32),
        kernel_size=3,
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3)
    ))
    model5.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))
    #Conv2
    model5.add(keras.layers.Conv2D(
        filters=hp.Int("filters_2", min_value=64, max_value=128, step=32),
        kernel_size=3,
        activation="relu",
        padding="same",
    ))
    model5.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))

    #Conv3
    model5.add(keras.layers.Conv2D(
        filters=hp.Int("filters_3", min_value=128, max_value=256, step=32),
        kernel_size=3,
        activation="relu",
        padding="same",
    ))
    model5.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))

    #Conv4
    model5.add(keras.layers.Conv2D(
        filters=hp.Int("filters_4", min_value=256, max_value=512, step=32),
        kernel_size=3,
        activation="relu",
        padding="same",
    ))
    model5.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))

    model5.add(keras.layers.Flatten())
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))

    # Add the dense layers with hyperparameter search space
    model5.add(keras.layers.Dense(
        units=hp.Int("dense_1", min_value=32, max_value=256, step=32),
        activation="relu",
    ))
    

    model5.add(keras.layers.Dense(
        units=hp.Int("dense_2", min_value=32, max_value=256, step=32),
        activation="relu"
    ))
    model5.add(keras.layers.Dropout(rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)))
    model5.add(Dense(1, activation='sigmoid'))


    # Compile the model
    model5.compile(loss=tf.losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy'])
    return model5

# Set up the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10
)

# Search for hyperparameters
tuner.search(traingenerator, epochs=10, validation_data=valgenerator)

# Get the best model
model5 = tuner.get_best_models(num_models=1)[0]

# Fit the best model with the selected hyperparameters
model5.summary()
hist5 = model5.fit(traingenerator, epochs=10, validation_data=valgenerator)
model5.evaluate(testgenerator)

acc_per_fold3 = []
loss_per_fold3 = []
losshist3=[]
acchist3=[]
vallosthist3=[]
valacchist3=[]
k = 1

nfolds = 5
kf = KFold(nfolds, shuffle=True)
for train, val in kf.split(df):
  trainkf, testkf = df.iloc[train], df.iloc[val]
  traingeneratorkf = generator.flow_from_dataframe(dataframe = trainkf,
                                               directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 

  testgeneratorkf = generator.flow_from_dataframe(dataframe = testkf,
                                              directory = None,
                                               x_col = 'image',
                                               y_col ='label',
                                               target_size=(32, 32),
                                               batch_size=64,
                                               class_mode='binary',
                                               shuffle=True) 
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {k} ...')
  hist7 = model5.fit(traingeneratorkf, epochs=5, validation_data=testgeneratorkf)
  losshist3.append(hist7.history['loss'])
  vallosthist3.append(hist7.history['val_loss'])
  acchist3.append(hist7.history['accuracy'])
  valacchist3.append(hist7.history['val_accuracy'])

  # Generate generalization metrics
  scores = model5.evaluate(testgeneratorkf)
  print(f'Score for fold {k}: {model5.metrics_names[0]} of {scores[0]}; {model5.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold3.append(scores[1] * 100)
  loss_per_fold3.append(scores[0])

  # Increase fold number
  k = k + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold3)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold3[i]} - Accuracy: {acc_per_fold3[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold3)} (+- {np.std(acc_per_fold3)})')
print(f'> Loss: {np.mean(loss_per_fold3)}')
print('------------------------------------------------------------------------')


