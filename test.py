# -*- coding: utf-8 -*-
"""
@author: jagriti
"""

import tensorflow as tf
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
#from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_json(project_dir):
    train_metadata = pd.read_json(project_dir +"/train_sample_videos/metadata.json", orient='index')
#    print(train_metadata.head())
    print(train_metadata['label'].value_counts(normalize=True))
    print(train_metadata['split'].value_counts(normalize=True))
    print(train_metadata.columns)
    train_metadata = train_metadata.reset_index()
    train_metadata.rename(columns = {'index':'video'}, inplace = True)
    train_metadata = train_metadata.drop(['split'], axis=1)
    print(train_metadata.head())
    videos = train_metadata['video']
    labels = train_metadata['label']
    originals = train_metadata['original']
    print(videos.head())
    print(labels.head())
    print(originals.head())    
    
    return 
, videos, labels, originals


def load_test_videos(project_dir):
    test_videos_names = pd.read_csv(project_dir +"/test_videos/test_videos_names.csv")
    videos = test_videos_names['video']
    print(test_videos_names.head())
    
    
    return test_videos_names, videos
    
def break_to_frames_train(project_dir, videos, labels, width, height):
    
    # storing the frames from training videos
    for i in tqdm(range(len(videos))):
        count = 0
        # capturing the video from the given path
        cap = cv2.VideoCapture(project_dir +"/train_sample_videos/"+videos[i])   
        frameRate = cap.get(5) #frame rate
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                #resize 
                frame = cv2.resize(frame, (width,height),interpolation=cv2.INTER_CUBIC)
                # storing the frames in a new folder named train_1
                filename =project_dir + '/train_1/' +labels[i] +"_"+ videos[i] +"_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        
        cap.release()


def break_to_frames_test(project_dir, videos, width, height):
    
    # storing the frames from training videos
    for i in tqdm(range(len(videos))):
        count = 0
        cap = cv2.VideoCapture(project_dir +"/test_videos/"+videos[i])   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                #resize
                frame = cv2.resize(frame, (width,height),interpolation=cv2.INTER_CUBIC)

                # storing the frames in a new folder named train_1
                filename =project_dir + '/test_1/'+ videos[i] +"_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        
        cap.release()    


def make_dataframe_train(project_dir):

    # getting the names of all the images
    images = glob(project_dir+"/train_1/*.jpg")
    train_image = []
    train_class = []
   
    for i in tqdm(range(len(images))):
        #images = "detection-challenge/train_1\FAKE_aagfhgtpmv.mp4_frame0.jpg"
        
        label_name = images[i].split('_')[1].split('\\')[1]
        image_name_parts = images[i].split('_')[2:]
        image_name = "_".join(image_name_parts)
        
        # print(label_name)
        # print(image_name)
        
        train_image.append(image_name)
        # creating the class of image
        train_class.append(label_name)       
       
    # storing the images and their class in a dataframe
    train_data = pd.DataFrame()
    train_data['image'] = train_image
    train_data['class'] = train_class
    
    print(train_data.head())
    # converting the dataframe into csv file 
    train_new_csv = train_data.to_csv(project_dir+'/train_new.csv',header=True, index=False)
    
    return train_new_csv
   

def make_dataframe_test(project_dir):

    # getting the names of all the images
    images = glob(project_dir+"/test_1/*.jpg")
    test_image = []
    # train_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        # label_name = images[i].split('/')[0]
        image_name = images[i]
        
        test_image.append(image_name)
        # creating the class of image
        # train_class.append(label_name)       

    # storing the images and their class in a dataframe
    test_data = pd.DataFrame()
    test_data['image'] = test_image
    # train_data['class'] = train_class
    
    # converting the dataframe into csv file 
    test_new_csv = test_data.to_csv(project_dir+'/test_new.csv',header=True, index=False)
    
    return test_new_csv

def get_Xy(project_dir,train_new_csv, width, height, depth):
    train = pd.read_csv(project_dir+'/train_new.csv')
    
    train.head()
    
    # creating an empty list
    train_image = []
    
    # for loop to read and store frames
    for i in tqdm(range(train.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        mylist = [train['class'][i],train['image'][i]]
        img_name_in_folder = "_".join(mylist)
        img = tf.keras.preprocessing.image.load_img(project_dir+'/train_1/'+img_name_in_folder, target_size=(width,height,depth))
        # converting it to array
        img = tf.keras.preprocessing.image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        train_image.append(img)
        
    # converting the list to numpy array
    X = np.array(train_image)
    
    # shape of the array
    print(X.shape)
    
    # separating the target
    y = train['class']    
    
    # creating the training and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
    y_train_original = y_train
    y_test_original = y_test
    # creating dummies of target variable for train and validation set
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    # don't write X_train.head(), as its not a pandas dataframe
    # print("X_train ", X_train)
    # print("X_test ", X_test)
    # print("y_train ", y_train)
    # print("y_test ", y_test)
    
    return X_train, y_train, X_test, y_test, train, y_train_original, y_test_original 
   

def vgg16Model(X_train, X_test, width, height, depth, classes):
    # creating the base model of pre-trained VGG16 model
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape =(width, height,depth))
    # extracting features for training frames
    X_train = base_model.predict(X_train)
    print(X_train.shape)
    # extracting features for validation frames
    X_test = base_model.predict(X_test)
    print(X_test.shape)
    # reshaping the training as well as validation frames in single dimension
    X_train = X_train.reshape(59075, 7*7*512)
    X_test = X_test.reshape(14769, 7*7*512)
    
    # # normalizing the pixel values
    # max = X_train.max()
    # X_train = X_train/max
    # X_test = X_test/max
    # # shape of images
    # X_train.shape
    
    #defining the model architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(25088,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(101, activation='sigmoid'))
    
    
    print(model.summary())
    return model
   

def plotAccLoss(H, NUM_EPOCHS):

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    
def evaluate(model, test_features, test_labels):
    
    y_preds = model.predict(test_features)

    # Accuracy Score using score method
    accuracy_score = model.score(test_features, test_labels)
    print ("Accuracy ", model, accuracy_score)
    
    # Confusion Matrix
    # Accuracy in machine learning algorithm is measured as:
    #               𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠+𝑇𝑟𝑢𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒𝑠
    #   --------------------------------------------------      
    # 𝑇𝑟𝑢𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠+𝐹𝑎𝑙𝑠𝑒𝑃𝑜𝑠𝑖𝑡𝑖𝑣𝑒𝑠+𝑇𝑟𝑢𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒𝑠+𝐹𝑎𝑙𝑠𝑒𝑁𝑒𝑔𝑎𝑡𝑖𝑣𝑒𝑠

    
    # F1 score
    f1score = f1_score(test_labels,y_preds, average = 'micro') #average = binary doesn't work
    print("f1score is", f1score)
    
    # Classification Report which shows Precision, Recall, F1Score and Support
    class_report = classification_report(test_labels,y_preds) #make sure to put test_labels in the method instead of test_features
    print("class_report is", class_report)
    
    # Measures of error, accuracy 
    
    errors = abs(y_preds - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error', np.mean(errors))
    print('Accuracy', accuracy)
    
    return accuracy    
 
def main():
    
    width = 128
    height = 128
    depth = 3
    classes = 2
    NUM_EPOCHS = 50
    
    
    #initialize the optimizer and model
    opt = tf.keras.optimizers.SGD(lr=0.01)
    
    project_dir = "deepfake-detection-challenge"
    #train_metadata, train_videos, labels, originals = load_json(project_dir)
    train_sub_dir = "/train_sample_videos/"
    dest_train_1 = '/train_1/'
    #break_to_frames_train(project_dir, train_videos, labels, width, height)
    
    #test_video_names, test_videos = load_test_videos(project_dir)
    test_sub_dir = "/test_videos/"
    dest_test_1 = '/test_1/'
    #break_to_frames_test(project_dir, test_videos, width, height)
  
    
    train_new_csv = make_dataframe_train(project_dir)
    test_new_csv = make_dataframe_test(project_dir)
    train_new_csv = '/train_new.csv'
    
    X_train, y_train, X_test, y_test, train, y_train_original, y_test_original  = get_Xy(project_dir,train_new_csv, width, height, depth)
    
    #Normalization
    X_train = X_train.astype("float")/ 255.0
    X_test = X_test.astype ("float")/ 255.0
    
    #One hot encode y
    
     
    choice = 4
    
    if choice == 1:  #not working 
        base_model = vgg16Model(X_train, X_test, width, height, depth, classes)
        
        # checkpointing to save the weights of best model
        mcp_save = tf.keras.callbacks.ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        # compiling the model
        base_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        # training the model
        H = base_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)
        print ("Base Model - Test Data Loss and Accuracy: ", model.evaluate(X_test, y_test))
        
        print("Final Plot ")
        plotAccLoss(H, NUM_EPOCHS)
        
    if choice == 2: 
        # Feature Extraction and Usage of Secondary Model
        vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(width, height, depth))
        print(vggModel.summary())
       
        X_train_new = vggModel.predict(X_train)
        X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
        X_val_new = vggModel.predict(X_test)
        X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
        secondary_model = 'random_forest'
        
        if (secondary_model == 'random_forest'):
            print("Secondary Model - Random Forest ")
            model = RandomForestClassifier(200)
            model.fit(X_train_new, y_train)
            # evaluate the model
            results = model.predict(X_val_new)
            print ("Random Forest Accuracy ", metrics.accuracy_score(results, y_test))
    
        if(secondary_model == 'naive_bayes'):
            print("Secondary Model - Using Naive Bayes")
            nBayes = GaussianNB()
            nBayes = nBayes.fit( X_train_new , y_train)
            accuracy = nBayes.score(X_val_new, y_test)
            print ("Naive Bayes Accuracy ", accuracy)
      
    if choice == 3: 
        # not working
        # FineTuning 
        inceptionV3Model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
        inceptionV3Model.trainable = False 
      
        model =tf.keras.models.Sequential()
        model.add (inceptionV3Model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout (0.5))
      
        model.add(tf.keras.layers.Dense (256, 'relu'))
        model.add(tf.keras.layers.Dense (classes, activation='sigmoid'))
        print (model.summary)
        NUM_EPOCHS =50
        opt = tf.keras.optimizers.SGD(lr=0.001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
      
        H = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
      
        plotAccLoss(H, NUM_EPOCHS)
      
        print ("\n Phase B  - Fine Tune Fully Connected Layer and Selected Convolutional Layers \n")
        inceptionV3Model.trainable = True
        trainableFlag = False
        for layer in inceptionV3Model.layers:
          if layer.name == 'block4_conv1':
            trainableFlag = True
          layer.trainable = trainableFlag
        opt = tf.keras.optimizers.SGD(lr=0.00001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        print (model.summary)
      
        H = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))
        print("Final Plot ")
        plotAccLoss(H, NUM_EPOCHS)


    if choice == 4:
        # works
        # Feature Extraction and Usage of Secondary Model
        inceptionV3Model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
        inceptionV3Model.trainable = False 
        
        print(inceptionV3Model.summary())
       
        X_train_new = inceptionV3Model.predict(X_train)
        X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
        X_val_new = inceptionV3Model.predict(X_test)
        X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
        secondary_model = 'random_forest'
        
        if(secondary_model == 'random_forest'):
            print("Secondary Model - Random Forest ")
            model = RandomForestClassifier(200)
            model.fit(X_train_new, y_train)
            # evaluate the model
            accuracy = evaluate(model, X_val_new, y_test)
            # results = model.predict(X_val_new)
            # print ("Random Forest Accuracy ", metrics.accuracy_score(results, y_test))
            print("Random Forest Accuracy ", accuracy)
    
        if(secondary_model == 'naive_bayes'):
            print("Secondary Model - Using Naive Bayes")
            nBayes = GaussianNB()
            nBayes = nBayes.fit( X_train_new , y_train)
            accuracy = nBayes.score(X_val_new, y_test)
            print ("Naive Bayes Accuracy ", accuracy)

    if choice == 41:
        # works
        # Feature Extraction and Usage of Secondary Model
        inceptionV3Model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
        inceptionV3Model.trainable = False 
        
        print(inceptionV3Model.summary())
       
        X_train_new = inceptionV3Model.predict(X_train)
        X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
        X_val_new = inceptionV3Model.predict(X_test)
        print("X_val_new b4 reshaping ", X_val_new)
        X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
        secondary_model = 'random_forest'
        
        if(secondary_model == 'random_forest'):
            print("Secondary Model - Random Forest ")
            model = RandomForestClassifier(200)
            model.fit(X_train_new, y_train)
            # evaluate the model
           
            predY = model.predict(X_val_new)
            #accuracy on the images
            print ("Images - Random Forest Accuracy ", metrics.accuracy_score(predY, y_test))
        
            
        #name of the video , label in X_val_new
        #collect all the images/group by all iamges with same irst name and count probability , if out of 11 frames atleast 3 are fake, then video is fake`
         # storing the images and their class in a dataframe
         
        # print("train.head() ", train.head(), train.shape )
        # print("y_test ", y_test, y_test.shape )
        # print("predY ", predY, predY.shape )
        # print("predY[:,0] ", predY[:,0])   #this a series
        # print("X_val_new ", X_val_new,X_val_new.shape )
        
        
        # pred_data_frame = train.copy(deep=True)
        # video_names = []
        # image_names = train['image']
        
        
        
        # for i in range(len(image_names)):
        #     #get the video name from the frame e.g.  aagfhgtpmv.mp4_frame0.jpg
        #     video_names.append(image_names[i].split("_")[0])
            
        # pred_data_frame['video'] =  video_names
        # print("pred_data_frame.head() ", pred_data_frame.head())
        # pred_data_frame['pred_image_fake'] = predY[:,0]
        # pred_data_frame['pred_image_real'] = predY[:,1]
        
        
        
        # pred_video_label1 = []        
        # # #sort the df based on video names
        # # pred_data_frame = pred_data_frame.sort_values(by=['video'])
        # pred_video_label = pred_data_frame.groupby(['video'])['pred_image_label'].count()
        # print(pred_video_label.head())


        # print ("Video Classification Accuracy ", metrics.accuracy_score(predY, y_test))

        # if(secondary_model == 'naive_bayes'):
        #     print("Secondary Model - Using Naive Bayes")
        #     nBayes = GaussianNB()
        #     nBayes = nBayes.fit( X_train_new , y_train)
        #     accuracy = nBayes.score(X_val_new, y_test)
        #     print ("Naive Bayes Accuracy ", accuracy)

            
    if choice == 5:
        
        #lstm
        model = Sequential()
        model.add(LSTM(256,dropout=0.2,input_shape=(train_data.shape[1],train_data.shape[2])))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))
        sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        #model.load_weights('video_1_LSTM_1_512.h5')
        callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
        nb_epoch = 500
        model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
        
        return model
    
    if choice ==6:
        #ensemble         
        vggModel= tf.keras.applications.VGG16 (weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
        model1 = tf.keras.models.Sequential()
        model1.add (vggModel)
        model1.add(tf.keras.layers.Flatten())
        model1.add(tf.keras.layers.Dropout (0.5))
        model1.add(tf.keras.layers.Dense (256, 'relu'))
        model1.add(tf.keras.layers.Dense (17, activation='softmax'))

        inceptionv3model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
    
        model2 = tf.keras.models.Sequential()
        model2.add(inceptionv3model)
        model2.add(tf.keras.layers.Flatten())
        model2.add(tf.keras.layers.Dropout (0.5))
        model2.add(tf.keras.layers.Dense (256, 'relu'))
        model2.add(tf.keras.layers.Dense (17, activation='softmax'))

       
        model_name = 'knn'
        if(model_name == 'randomforest'):
            model = RandomForestClassifier(200)
            model.fit(featuresTrain, trainY)
            # evaluate the model
            results = model.predict(featuresVal)
            print (metrics.accuracy_score(results, testY))
    
       
        if(model_name == 'knn'):
            print("using knn")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(featuresTrain, trainY)
            knn.predict(featuresVal)
            results = knn.predict(featuresVal)
            print (metrics.accuracy_score(results, testY))
    
      
        if(model_name == 'naive_bayes'):
            print("Using Naive Bayes")
            
            nBayes = GaussianNB()
            nBayes = nBayes.fit( featuresTrain , trainY)
            accuracy = nBayes.score(featuresVal, testY)
            print ("Naive Bayes Accuracy ", accuracy)
    
        
        if(model_name == 'svm'):
            print("Using SVM")
           
            svc = SVC(gamma='auto')
            svc = svc.fit(featuresTrain, trainY)
    #         accuracy = svc.score(test_features, test_labels)
            accuracy = evaluate(svc, featuresVal, testY)
            print ("SVM Accuracy ", accuracy)      
        # resnet50model = tf.keras.applications.resnet50(weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
        # model3 = tf.keras.models.Sequential()
        # model3.add(resnet50model)
        # model3.add(tf.keras.layers.Flatten())
        # model3.add(tf.keras.layers.Dropout (0.5))
        # model3.add(tf.keras.layers.Dense (256, 'relu'))
        # model3.add(tf.keras.layers.Dense (17, activation='softmax'))
    
         # Find the probabilities of all 17 classes in each instance of test data - should be 340 *17 
        predicted_vals1 = model1.predict(testX)
        print("predicted_vals1 shape ", predicted_vals1.shape )
        print("predicted_vals1 ", predicted_vals1 )
    
        predicted_vals2 = model2.predict(testX)
        print("predicted_vals2 shape ", predicted_vals2.shape )
        print("predicted_vals2 ", predicted_vals2 )
        
    
        # predicted_vals3 = model3.predict(testX)
        # print("predicted_vals3 shape ", predicted_vals3.shape )
        # print("predicted_vals3 ", predicted_vals3 )
    
        # element wise addition will help, as we want to add probabilities of each class for each image. Then takke average,
        # as I am using 3 models so 1/3 is multipled to the sum
        predY_sum = predicted_vals1+ predicted_vals2
        element_wise_sum_avg = predY_sum * (1/2)
    
        # Now doing np.argmax
    
        predY = np.argmax(element_wise_sum_avg, axis =1) 
    
        print("predY ", predY)
    
        print("Checking shapes of testY and predY ", testY.shape, " ", predY.shape)
    
        accuracy = accuracy_score(testY, predY)
    
        print(accuracy)
    
    if choice == 7:
        resnet101model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        print(resnet101model.summary())
   
        featuresTrain = resnet101model.predict(trainX)
        featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)
        featuresVal = resnet101model.predict(testX)
        featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

main()    