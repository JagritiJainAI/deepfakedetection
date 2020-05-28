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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

""" Load the json and csv files containing name of videos for training, validating and testing """
""" 400 training videos """
def load_json(project_dir):
    train_metadata = pd.read_json(project_dir +"/train_sample_videos/metadata.json", orient='index')
#    print(train_metadata.head())
    print(train_metadata['label'].value_counts(normalize=True))
    print(train_metadata['split'].value_counts(normalize=True))
    print(train_metadata.columns)
    train_metadata = train_metadata.reset_index()
    train_metadata.rename(columns = {'index':'video'}, inplace = True)
    train_metadata = train_metadata.drop(['split'], axis=1)
    
    print("train_metadata.json ", train_metadata.head())
    videos = train_metadata['video']
    labels = train_metadata['label']
    originals = train_metadata['original']
    
    """ Split this metadata into train and validation data and make sure to reset index as we are using random state, which retains the row no. """
    
    train_split = train_metadata.sample(frac=0.8,random_state=200)
    print("train_split without resetting index ", train_split.head())
    train_split = train_split.reset_index(drop=True)
    print("train_split after resetting index ", train_split.head())
    train_videos = train_split['video']
    train_labels = train_split['label']
    
    val_split=train_metadata.drop(train_split.index)
    val_split = val_split.reset_index(drop=True)
    print("val_split ", val_split.head())
    val_videos = val_split['video']
    val_labels = val_split['label']


    # print(videos.head())
    # print(labels.head())
    # print(originals.head())    
    
    return train_metadata, train_videos, train_labels, val_videos, val_labels, originals


""" 400 test videos """
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

def break_to_frames_val(project_dir, videos, labels, width, height):
    
    # storing the frames from training videos
    for i in tqdm(range(len(videos))):
        count = 0
        # capturing the video from the given path, 80 images based on video names 
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
                filename =project_dir + '/val_1/' +labels[i] +"_"+ videos[i] +"_frame%d.jpg" % count;count+=1
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

def make_dataframe_val(project_dir):

    # getting the names of all the images
    images = glob(project_dir+"/val_1/*.jpg")
    val_image = []
    val_class = []
   
    for i in tqdm(range(len(images))):
        #images = "detection-challenge/train_1\FAKE_aagfhgtpmv.mp4_frame0.jpg"
        
        label_name = images[i].split('_')[1].split('\\')[1]
        image_name_parts = images[i].split('_')[2:]
        image_name = "_".join(image_name_parts)
        
        # print(label_name)
        # print(image_name)
        
        val_image.append(image_name)
        # creating the class of image
        val_class.append(label_name)       
       
    # storing the images and their class in a dataframe
    val_data = pd.DataFrame()
    val_data['image'] = val_image
    val_data['class'] = val_class
    
    print(val_data.head())
    # converting the dataframe into csv file 
    val_new_csv = val_data.to_csv(project_dir+'/val_new.csv',header=True, index=False)
    
    return val_new_csv    

def make_dataframe_test(project_dir):

    # getting the names of all the images
    images = glob(project_dir+"/test_1/*.jpg")
    test_image = []
    # train_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        # label_name = images[i].split('/')[0]
        print("images[i]", images[i])
        break
        
        image_name = images[i].split("_")[1]
        video_name = images[i].split("_")[0]
        
        test_image.append(image_name)
        test_video.append(video_name)
        # creating the class of image
        # train_class.append(label_name)       

    # storing the images and their class in a dataframe
    test_data = pd.DataFrame()
    test_data['video'] = test_video
    test_data['image'] = test_image
    # train_data['class'] = train_class
    
    # converting the dataframe into csv file 
    test_new_csv = test_data.to_csv(project_dir+'/test_new.csv',header=True, index=False)
    print(" test_new_csv ", test_new_csv.head())
    return test_new_csv

def get_X_test(project_dir,test_new_csv, width, height, depth):
    
    test = pd.read_csv(project_dir+'/test_new.csv')
    test_image =[]
    video_names = []
    
     # for loop to read and store frames
    for i in tqdm(range(test.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        mylist = [test['video'][i], test['image'][i]]
        img_name_in_folder = "_".join(mylist)
        img = tf.keras.preprocessing.image.load_img(project_dir+'/test_1/'+img_name_in_folder, target_size=(width,height,depth))
        # converting it to array
        img = tf.keras.preprocessing.image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        test_image.append(img)
        video_names.append(test['video'][i])
        
    # converting the list to numpy array
    X_test = np.array(test_image)
    # just contains image names,  converted to numpy array
    print("X_test ", X_test)
    print("test_image ", test_image)
    print("video_names ", video_names)
    
    return X_test, test_image, video_names

def get_Xy(project_dir,train_new_csv, val_new_csv, width, height, depth):
    
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
    X_train = np.array(train_image)  
    # shape of the array
    print("X_train.shape ", X_train.shape)    
    # separating the target
    y_train = train['class']    
    
    val = pd.read_csv(project_dir+'/val_new.csv')
    val_image =[]
    
     # for loop to read and store frames
    for i in tqdm(range(val.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        mylist = [val['class'][i],val['image'][i]]
        img_name_in_folder = "_".join(mylist)
        img = tf.keras.preprocessing.image.load_img(project_dir+'/val_1/'+img_name_in_folder, target_size=(width,height,depth))
        # converting it to array
        img = tf.keras.preprocessing.image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        val_image.append(img)
        
    # converting the list to numpy array
    X_val = np.array(val_image)
    # shape of the array
    print("X_val.shape ", X_val.shape)    
    # separating the target
    y_val = val['class']    
    
    # # creating the training and validation set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
    y_train_original = y_train
    y_val_original = y_val
    # creating dummies of target variable for train and validation set
    y_train = pd.get_dummies(y_train)
    y_val = pd.get_dummies(y_val)
    
    return X_train, y_train, X_val, y_val, train, val, y_train_original, y_val_original 
   

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
 
def learning_curve(estimator, X, y):
    
    title = "Learning Curve"
    plt.title(title)
    ylim=(0.0, 1.01)
    
    
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
        
    train_sizes=np.linspace(.05, 1.0, 10)
    custom_cv = ShuffleSplit(n_splits=30, test_size=0.1, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv = custom_cv)
            
    # remember axis = 1 operates on the horizontal axis and calculates the mean below for       # each row. Each row corresponds to one tick and each value returned in a row refer to the 
    # accuracy for that fold. 
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()

def video_classification():
    print("y_val is ", y_val, y_val.shape)
    print("y_pred is ", y_pred, y_pred.shape)
    
    """ val is dataframe containing the names of frames and classes 100 videos * 11 = 1100 frames and 1100 classes"""
    new_val = val.copy(deep=True)      
    val_image_names = new_val['image']
    val_video_names = []
    
    for i in range(len(val_image_names)):
        #get the video name from the frame e.g.  aagfhgtpmv.mp4_frame0.jpg
        val_video_names.append(val_image_names[i].split("_")[0])
    
    new_val['video_name'] = val_video_names
    new_val['y_pred_fake'] = y_pred[:,0]
    new_val['y_pred_real'] = y_pred[:,1]
    
    print("new_val_dataframe ", new_val.head())
     
    """ export this dataframe """
    new_val_dataframe = new_val.to_csv(project_dir+'/new_val_dataframe.csv',header=True, index=False)
    
    """ group by the name of video, which is there in the image/frame name and count the labels/classes for fake """
    actual_video_class_fake =[]
    
    actual_video_class = new_val['class']
    for i in range(len(actual_video_class)):
        if actual_video_class[i] == "FAKE":
            actual_video_class_fake.append(1)
        else:     
            actual_video_class_fake.append(0)
                    
    new_val['actual_video_class_fake'] = actual_video_class_fake  
    
    print(" new_val['actual_video_class_fake'] ",  new_val['actual_video_class_fake'])
   
    new_val4 = new_val.groupby(['video_name'])['actual_video_class_fake', 'y_pred_fake'].sum()
        
        
    print(" new_val4['actual_video_class_fake'] ",  new_val4['actual_video_class_fake'])
   
    
    # actual_video_class =  
    """ export this dataframe """
    new_val4_dataframe = new_val4.to_csv(project_dir+'/new_val4_dataframe.csv',header=True, index=False)
    
    pred_video_class = []
    actual_video_class_groupby =[]
    
    y_pred_fake_series = new_val4['y_pred_fake']
    #Taking 27.27 % criteria, if 3 out of 11 frames are fake, then video is fake
    for i in range(len(y_pred_fake_series)):
        if y_pred_fake_series[i] >=3:  
            pred_video_class.append("FAKE")         
        else:
            pred_video_class.append("REAL")
    
    actual_video_class_fake_series = new_val4['actual_video_class_fake']
    for j in range (len(actual_video_class_fake_series)):         
        if actual_video_class_fake_series[j] ==11:
            actual_video_class_groupby.append("FAKE")
        else:
            actual_video_class_groupby.append("REAL")
            
         
    actual = pd.get_dummies(actual_video_class_groupby)
    pred = pd.get_dummies(pred_video_class)
    print("actual ", actual )
    print("pred ", pred)
    """ Video Classification Accuracy """
    """ Getting the original dataframe containing names and labels for validation videos and adding column of predicted labels"""

    print ("Video Classification Accuracy ", metrics.accuracy_score(pred, actual))
    
    class_report = classification_report(actual,pred) #make sure to put test_labels in the method instead of test_features
    print("class_report is", class_report)

    # Measures of error, accuracy 
    
    errors = abs(pred - actual)
    mape = 100 * np.mean(errors / actual)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error', np.mean(errors))
    
def main():
    
    """ Fixing size of image to 128*128*3 for consistency, Image resolution has been decreased to increase speed of running algorithm"""
    width = 128
    height = 128
    depth = 3
    classes = 2
    NUM_EPOCHS = 50
    
    #initialize the optimizer and model
    opt = tf.keras.optimizers.SGD(lr=0.01)
    project_dir = "deepfake-detection-challenge"
    preprocessing_done = 1
    
    if preprocessing_done == 0:
    
        train_metadata, train_videos, train_labels, val_videos, val_labels, originals = load_json(project_dir)
        print("train videos ", train_videos.shape)
        print("val videos ", val_videos.shape)
        
        train_sub_dir = "/train_sample_videos/"
        dest_train_1 = '/train_1/'
        break_to_frames_train(project_dir, train_videos, train_labels, width, height)
        
        dest_val_1 = '/val_1/'
        break_to_frames_val(project_dir, val_videos, val_labels, width, height)
        
        test_video_names, test_videos = load_test_videos(project_dir)
        test_sub_dir = "/test_videos/"
        dest_test_1 = '/test_1/'
        break_to_frames_test(project_dir, test_videos, width, height)
  
        train_new_csv = make_dataframe_train(project_dir)
        test_new_csv = make_dataframe_test(project_dir)
        val_new_csv = make_dataframe_val(project_dir)
    
    train_new_csv = '/train_new.csv'
    val_new_csv = '/val_new.csv'
    test_new_csv = '/train_new.csv'

    X_train, y_train, X_val, y_val, train, val, y_train_original, y_val_original  = get_Xy(project_dir,train_new_csv, val_new_csv, width, height, depth)
    
    # X_test, test_image_names, video_names = get_X_test(project_dir,test_new_csv, width, height, depth) 
    
    choice = 8
    
    if choice == 8:
        # works
        # Feature Extraction and Usage of Secondary Model
        inceptionV3Model= tf.keras.applications.VGG16(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
        inceptionV3Model.trainable = False 
        
        print(inceptionV3Model.summary())
       
        X_train_new = inceptionV3Model.predict(X_train)
        X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
        X_val_new = inceptionV3Model.predict(X_val)
        print("X_val_new b4 reshaping ", X_val_new.shape)
        X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
        secondary_model = 'random_forest'
        
        """ Image Classification Accuracy """
        if(secondary_model == 'random_forest'):
            print("Secondary Model - Random Forest ")
            model = RandomForestClassifier(200)
            model.fit(X_train_new, y_train)
            # evaluate the model
           
            y_pred = model.predict(X_val_new)
            #accuracy on the images
            print ("Images - Random Forest Accuracy ", metrics.accuracy_score(y_pred, y_val))
        
        
        # if(secondary_model == 'naive_bayes'):
        #     print("Secondary Model - Using Naive Bayes")
        #     nBayes = GaussianNB()
        #     nBayes = nBayes.fit( X_train_new , y_train)
        #     y_pred = model.predict(X_val_new)
        #     accuracy = nBayes.score(y_pred, y_val)
        #     print ("Images - Naive Bayes Accuracy ", accuracy)
        
        
        print("y_val is ", y_val, y_val.shape)
        print("y_pred is ", y_pred, y_pred.shape)
        
        """ val is dataframe containing the names of frames and classes 100 videos * 11 = 1100 frames and 1100 classes"""
        new_val = val.copy(deep=True)      
        val_image_names = new_val['image']
        val_video_names = []
        
        for i in range(len(val_image_names)):
            #get the video name from the frame e.g.  aagfhgtpmv.mp4_frame0.jpg
            val_video_names.append(val_image_names[i].split("_")[0])
        
        new_val['video_name'] = val_video_names
        new_val['y_pred_fake'] = y_pred[:,0]
        new_val['y_pred_real'] = y_pred[:,1]
        
        print("new_val_dataframe ", new_val.head())
         
        """ export this dataframe """
        new_val_dataframe = new_val.to_csv(project_dir+'/new_val_dataframe.csv',header=True, index=False)
        
        """ group by the name of video, which is there in the image/frame name and count the labels/classes for fake """
        actual_video_class_fake =[]
        
        actual_video_class = new_val['class']
        for i in range(len(actual_video_class)):
            if actual_video_class[i] == "FAKE":
                actual_video_class_fake.append(1)
            else:     
                actual_video_class_fake.append(0)
                        
        new_val['actual_video_class_fake'] = actual_video_class_fake  
        
        print(" new_val['actual_video_class_fake'] ",  new_val['actual_video_class_fake'])
       
        new_val4 = new_val.groupby(['video_name'])['actual_video_class_fake', 'y_pred_fake'].sum()
        
        
        print(" new_val4['actual_video_class_fake'] ",  new_val4['actual_video_class_fake'])
       
        
        # actual_video_class =  
        """ export this dataframe """
        new_val4_dataframe = new_val4.to_csv(project_dir+'/new_val4_dataframe.csv',header=True, index=False)
        
        pred_video_class = []
        actual_video_class_groupby =[]
        
        y_pred_fake_series = new_val4['y_pred_fake']
        #Taking 27.27 % criteria, if 3 out of 11 frames are fake, then video is fake
        for i in range(len(y_pred_fake_series)):
            if y_pred_fake_series[i] >=3:  
                pred_video_class.append("FAKE")         
            else:
                pred_video_class.append("REAL")
        
        actual_video_class_fake_series = new_val4['actual_video_class_fake']
        for j in range (len(actual_video_class_fake_series)):         
            if actual_video_class_fake_series[j] ==11:
                actual_video_class_groupby.append("FAKE")
            else:
                actual_video_class_groupby.append("REAL")
                
             
        actual = pd.get_dummies(actual_video_class_groupby)
        pred = pd.get_dummies(pred_video_class)
        print("actual ", actual )
        print("pred ", pred)
        """ Video Classification Accuracy """
        """ Getting the original dataframe containing names and labels for validation videos and adding column of predicted labels"""

        print ("Video Classification Accuracy ", metrics.accuracy_score(pred, actual))
        
        class_report = classification_report(actual,pred) #make sure to put test_labels in the method instead of test_features
        print("class_report is", class_report)
    
        # Measures of error, accuracy 
        
        errors = abs(pred - actual)
        mape = 100 * np.mean(errors / actual)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error', np.mean(errors))
        
        
        #Predictions on Test data
        # X_test_image = get_X_test(project_dir,test_new_csv, width, height, depth)
        # y_pred_image_test = model.predict(X_test)
        # print ("Predictions - Image Classification ", y_pred_image_test)
        # print("X_test_image ", X_test_image.shape)
        # Form a dataframe of X_test and y_pred
        
        #test_image_names, video_names
        
        
    # if choice == 1:  #not working 
    #     base_model = vgg16Model(X_train, X_test, width, height, depth, classes)
        
    #     # checkpointing to save the weights of best model
    #     mcp_save = tf.keras.callbacks.ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    #     # compiling the model
    #     base_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    #     # training the model
    #     H = base_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)
    #     print ("Base Model - Test Data Loss and Accuracy: ", model.evaluate(X_test, y_test))
        
    #     print("Final Plot ")
    #     plotAccLoss(H, NUM_EPOCHS)
        
    # if choice == 2: 
    #     # Feature Extraction and Usage of Secondary Model
    #     vggModel = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(width, height, depth))
    #     print(vggModel.summary())
       
    #     X_train_new = vggModel.predict(X_train)
    #     X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
    #     X_val_new = vggModel.predict(X_test)
    #     X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
    #     secondary_model = 'random_forest'
        
    #     if (secondary_model == 'random_forest'):
    #         print("Secondary Model - Random Forest ")
    #         model = RandomForestClassifier(200)
    #         model.fit(X_train_new, y_train)
    #         # evaluate the model
    #         results = model.predict(X_val_new)
    #         print ("Random Forest Accuracy ", metrics.accuracy_score(results, y_test))
    
    #     if(secondary_model == 'naive_bayes'):
    #         print("Secondary Model - Using Naive Bayes")
    #         nBayes = GaussianNB()
    #         nBayes = nBayes.fit( X_train_new , y_train)
    #         accuracy = nBayes.score(X_val_new, y_test)
    #         print ("Naive Bayes Accuracy ", accuracy)
      
    # if choice == 3: 
    #     # not working
    #     # FineTuning 
    #     inceptionV3Model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
    #     inceptionV3Model.trainable = False 
      
    #     model =tf.keras.models.Sequential()
    #     model.add (inceptionV3Model)
    #     model.add(tf.keras.layers.Flatten())
    #     model.add(tf.keras.layers.Dropout (0.5))
      
    #     model.add(tf.keras.layers.Dense (256, 'relu'))
    #     model.add(tf.keras.layers.Dense (classes, activation='sigmoid'))
    #     print (model.summary)
    #     NUM_EPOCHS =50
    #     opt = tf.keras.optimizers.SGD(lr=0.001)
    #     model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
      
    #     H = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
      
    #     plotAccLoss(H, NUM_EPOCHS)
      
    #     print ("\n Phase B  - Fine Tune Fully Connected Layer and Selected Convolutional Layers \n")
    #     inceptionV3Model.trainable = True
    #     trainableFlag = False
    #     for layer in inceptionV3Model.layers:
    #       if layer.name == 'block4_conv1':
    #         trainableFlag = True
    #       layer.trainable = trainableFlag
    #     opt = tf.keras.optimizers.SGD(lr=0.00001)
    #     model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    #     print (model.summary)
      
    #     H = model.fit(trainX, trainY, epochs=NUM_EPOCHS, batch_size=32, validation_data=(testX, testY))
    #     print("Final Plot ")
    #     plotAccLoss(H, NUM_EPOCHS)


    # if choice == 4:
    #     # works
    #     # Feature Extraction and Usage of Secondary Model
    #     inceptionV3Model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(width, height,depth))
    #     inceptionV3Model.trainable = False 
        
    #     print(inceptionV3Model.summary())
       
    #     X_train_new = inceptionV3Model.predict(X_train)
    #     X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
    #     X_val_new = inceptionV3Model.predict(X_val)
    #     X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
         
    #     secondary_model = 'random_forest'
        
    #     if(secondary_model == 'random_forest'):
    #         print("Secondary Model - Random Forest ")
    #         model = RandomForestClassifier(200)
    #         model.fit(X_train_new, y_train)
    #         # evaluate the model
    #         accuracy = evaluate(model, X_val_new, y_val)
    #         # results = model.predict(X_val_new)
    #         # print ("Random Forest Accuracy ", metrics.accuracy_score(results, y_test))
    #         print("Random Forest Accuracy ", accuracy)
    
    #     if(secondary_model == 'naive_bayes'):
    #         print("Secondary Model - Using Naive Bayes")
    #         nBayes = GaussianNB()
    #         nBayes = nBayes.fit( X_train_new , y_train)
    #         accuracy = nBayes.score(X_val_new, y_val)
    #         print ("Naive Bayes Accuracy ", accuracy) 

            
    # if choice == 5:
        
    #     #lstm
    #     model = Sequential()
    #     model.add(LSTM(256,dropout=0.2,input_shape=(train_data.shape[1],train_data.shape[2])))
    #     model.add(Dense(1024, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(5, activation='softmax'))
    #     sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
    #     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #     #model.load_weights('video_1_LSTM_1_512.h5')
    #     callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
    #     nb_epoch = 500
    #     model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
        
    #     return model
    
    # if choice ==6:
    #     #ensemble         
    #     vggModel= tf.keras.applications.VGG16 (weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
    #     model1 = tf.keras.models.Sequential()
    #     model1.add (vggModel)
    #     model1.add(tf.keras.layers.Flatten())
    #     model1.add(tf.keras.layers.Dropout (0.5))
    #     model1.add(tf.keras.layers.Dense (256, 'relu'))
    #     model1.add(tf.keras.layers.Dense (2, activation='sigmoid'))

    #     inceptionv3model= tf.keras.applications.InceptionV3(weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
    
    #     model2 = tf.keras.models.Sequential()
    #     model2.add(inceptionv3model)
    #     model2.add(tf.keras.layers.Flatten())
    #     model2.add(tf.keras.layers.Dropout (0.5))
    #     model2.add(tf.keras.layers.Dense (256, 'relu'))
    #     model2.add(tf.keras.layers.Dense (2, activation='sigmoid'))
        
    #     resnet101model = tf.keras.applications.ResNet101(weights = 'imagenet',include_top =False, input_shape =(128, 128,3))
    #     model3 = tf.keras.models.Sequential()
    #     model3.add(resnet101model)
    #     model3.add(tf.keras.layers.Flatten())
    #     model3.add(tf.keras.layers.Dropout (0.5))
    #     model3.add(tf.keras.layers.Dense (256, 'relu'))
    #     model3.add(tf.keras.layers.Dense (2, activation='sigmoid'))
        
        
    #     # X_train_new = inceptionV3Model.predict(X_train)
    #     # X_train_new = X_train_new.reshape(X_train_new.shape[0], -1)
    #     # X_val_new = inceptionV3Model.predict(X_val)
    #     # print("X_val_new b4 reshaping ", X_val_new.shape)
    #     # X_val_new = X_val_new.reshape(X_val_new.shape[0], -1)
        
    #     # model_name = 'knn'
        
    #     # if(model_name == 'randomforest'):
    #     #     model = RandomForestClassifier(200)
    #     #     model.fit(X_train_new, y_train)
    #     #     # evaluate the model
           
    #     #     y_pred = model.predict(X_val_new)
            
            
    #     #     print (metrics.accuracy_score(y_pred, testY))
    
       
    #     # if(model_name == 'knn'):
    #     #     print("using knn")
    #     #     model = KNeighborsClassifier(n_neighbors=3)
    #     #     model.fit(X_train_new, y_train)
    #     #     # evaluate the model
           
    #     #     y_pred = model.predict(X_val_new)
            
           
    #     #     print (metrics.accuracy_score(y_pred, testY))
    
      
    
        
    #     # if(model_name == 'svm'):
    #     #     print("Using SVM")
           
    #     #     model = SVC(gamma='auto')
    #     #     model.fit(X_train_new, y_train)
    #     #     # evaluate the model
           
    #     #     y_pred = model.predict(X_val_new)

    #     #     print ("SVM Accuracy ", metrics.accuracy_score(y_pred, testY))      
        
        
    
    #       # Find the probabilities of all 17 classes in each instance of test data - should be 340 *17 
    #     predicted_vals1 = model1.predict(X_train)
    #     print("predicted_vals1 shape ", predicted_vals1.shape )
    #     print("predicted_vals1 ", predicted_vals1 )
    
    #     predicted_vals2 = model2.predict(X_train)
    #     print("predicted_vals2 shape ", predicted_vals2.shape )
    #     print("predicted_vals2 ", predicted_vals2 )
        
    
    #     predicted_vals3 = model3.predict(X_train)
    #     print("predicted_vals3 shape ", predicted_vals3.shape )
    #     print("predicted_vals3 ", predicted_vals3 )
    
    #     # element wise addition will help, as we want to add probabilities of each class for each image. Then takke average,
    #     # as I am using 3 models so 1/3 is multipled to the sum
    #     predY_sum = predicted_vals1+ predicted_vals2 + predicted_vals3
    #     element_wise_sum_avg = predY_sum * (1/3)
    
    #     # Now doing np.argmax
    
    #     predY = np.argmax(element_wise_sum_avg, axis =1) 
    
    #     print("predY ", predY)
    #     y_val_num =[]
    #     #Considering 0 for fake and 1 for original as per the predY outputted
    #     for i in range(len(y_val_original)):
    #         if y_val_original[i] == "FAKE":
    #             y_val_num.append(0)
    #         else:
    #             y_val_num.append(1)
        
    #     y_val_num = np.array(y_val_num) 
    #     print("Checking shapes of testY and predY ", y_val_num.shape, " ", predY.shape)
    
    #     accuracy = accuracy_score(y_val_num, predY)
    
    #     print("Image Classification Accuracy is ", accuracy)
        
        
        
    #     # print("y_val is ", y_val, y_val.shape)
    #     # print("y_pred is ", y_pred, y_pred.shape)
        
    #     """ val is dataframe containing the names of frames and classes 100 videos * 11 = 1100 frames and 1100 classes"""
    #     new_val = val.copy(deep=True)      
    #     val_image_names = new_val['image']
    #     val_video_names = []
        
    #     for i in range(len(val_image_names)):
    #         #get the video name from the frame e.g.  aagfhgtpmv.mp4_frame0.jpg
    #         val_video_names.append(val_image_names[i].split("_")[0])
        
    #     new_val['video_name'] = val_video_names
    #     new_val['y_pred'] = predY

        
    #     print("new_val_dataframe ", new_val.head())
         
    #     """ export this dataframe """
    #     new_val_dataframe = new_val.to_csv(project_dir+'/new_val_dataframe.csv',header=True, index=False)
        
    #     """ group by the name of video, which is there in the image/frame name and count the labels/classes for fake """
        
                        
    #     new_val['y_val_num'] = y_val_num  
        
              
    #     new_val4 = new_val.groupby(['video_name'])['y_val_num', 'y_pred'].sum()
        
        
    #     print(" new_val4['actual_video_class_fake'] ",  new_val4['actual_video_class_fake'])
       
        
    #     # actual_video_class =  
    #     """ export this dataframe """
    #     new_val4_dataframe = new_val4.to_csv(project_dir+'/new_val4_dataframe.csv',header=True, index=False)
        
       
        
    #     #Taking 27.27 % criteria, if 3 out of 11 frames are fake, then video is fake
    #     for i in range(len(y_pred)):
    #         if y_pred_fake_series[i] <=3:  
    #             pred_video_class.append("FAKE")         
    #         else:
    #             pred_video_class.append("REAL")
        
    #     actual_video_class_fake_series = new_val4['actual_video_class_fake']
    #     for j in range (len(y_val_num)):         
    #         if actual_video_class_fake_series[j] ==11:
    #             actual_video_class_groupby.append("FAKE")
    #         else:
    #             actual_video_class_groupby.append("REAL")
                
             
    #     actual = pd.get_dummies(actual_video_class_groupby)
    #     pred = pd.get_dummies(pred_video_class)
    #     print("actual ", actual )
    #     print("pred ", pred)
    #     """ Video Classification Accuracy """
    #     """ Getting the original dataframe containing names and labels for validation videos and adding column of predicted labels"""

    #     print ("Video Classification Accuracy ", metrics.accuracy_score(pred, actual))
        
    #     class_report = classification_report(actual,pred) #make sure to put test_labels in the method instead of test_features
    #     print("class_report is", class_report)
    
    #     # Measures of error, accuracy 
        
    #     errors = abs(pred - actual)
    #     mape = 100 * np.mean(errors / actual)
    #     accuracy = 100 - mape
    #     print('Model Performance')
    #     print('Average Error', np.mean(errors))
        
        
        
    
    # if choice == 7:
    #     resnet101model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    #     print(resnet101model.summary())
   
    #     featuresTrain = resnet101model.predict(trainX)
    #     featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)
    #     featuresVal = resnet101model.predict(testX)
    #     featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)

main()    