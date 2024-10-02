import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
#import pandas as pd
import os
import sys
import time
import tensorflow as tf
import pathlib
from pathlib import Path
import random
import glob
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.metrics import classification_report
import load_npy
import load_dataset
import load_dataset_tif
#import resnet
import dcnn
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input
import ROC_curve
import efficientnet.tfkeras as efn
import CNN0927
import vgg16_1208
from tensorflow.keras.callbacks import ReduceLROnPlateau
import multi_channel

#from keras.backend.tensorflow_backend import set_session

def setup(foldername, model_type):
    print(tf.__version__)
    #print(tf.enable_eager_execution())
    print(tf.test.is_gpu_available())
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #config.gpu_options.allow_growth = True
    #set_session(tf.Session(config=config))
    
    ### Generate Results
    try:
        os.mkdir("./Results/outcome_pictures/" + foldername)
    except FileExistsError:
        print('File exist!!')
    classification_report_out = "./Results/outcome_pictures/" + foldername + "/2023_" + model_type + "_Original.txt" # vgg/resnet
    try:
        f = open(classification_report_out, 'w')
    except FileExistsError:
        os.mkdir(classification_report_out)
        f = open(classification_report_out, 'w')
    
    
    return f

def model_setting(conv_base, class_num):
    layers=tf.keras.layers # define
    
    #model=conv_base
    
    model = tf.keras.models.Sequential()
    model.add(conv_base)
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(class_num,activation='softmax')) #改成分類數量
    #model.add(layers.Dense(class_num, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer = tf.keras.regularizers.l2(0.001)))
    
    conv_base.trainable=False # 注意 False
    return model

def print_spend_time(start_time, end_time, state, f, model_type):
    mins = (end_time - start_time) // 60
    secs = (end_time - start_time) % 60
    if state == 'training':
        print(model_type + " Training Time: {}:{:.2f}".format(mins, secs)) # vgg/resnet
        f.write(model_type + " Training Time: {}:{:.2f}".format(mins, secs)+"\n") # vgg/resnet
    if state == 'testing':
        print(model_type + " Testing Time: {}:{:.2f}".format(mins, secs))
        f.write(model_type + " Testing Time: {}:{:.2f}".format(mins, secs)+"\n") # vgg/resnet

def plot_learning_curves(history, epoch_no, foldername, model_type):
	# plot the training loss and accuracy
    plt.figure(figsize=(8,5))
    plt.title('Training / Validation Loss and Accuracy on ' + model_type + ' (Original Images)',y=1.05) # vgg/resnet
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.gca().set_xlim(0,epoch_no)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    #plt.xticks(np.linspace(0, epoch_no, 7))
    plt.plot(history.history["loss"], label="training_loss")
    plt.plot(history.history["val_loss"], label="validation_loss")
    plt.plot(history.history["acc"], label="training_accuracy")
    plt.plot(history.history["val_acc"], label="validation_accuracy")
    plt.legend(loc="upper right",fontsize=10)
    plt.savefig('./Results/outcome_pictures/' + foldername + '/' + model_type + '_Original_curves.png') # vgg/resnet
    #plt.show()

def print_confusion_matrix(model, test_dataset, test_count, BATCH_SIZE, target_names, f): # test set
    x, y_true = [], []
    i = 0
    for element in test_dataset:
        i += 1
        _x, _y = element	
        x.append(_x.numpy())
        y_true.append(_y.numpy())
        if i==test_count//BATCH_SIZE:
            break

    x = np.concatenate(x, axis=0)
    y_true = np.concatenate(y_true)
    y_pred = model.predict(x, verbose=0)
    y_pred = np.argmax(y_pred, axis=-1)
    confmatrix = confusion_matrix(y_true, y_pred)
    print(confmatrix)
    print(classification_report(y_true, y_pred, labels=range(len(target_names)), target_names=target_names))
    f.write(classification_report(y_true, y_pred, labels=range(len(target_names)), target_names=target_names))
    f.close()
    return confmatrix

def ConfusionMatrixPlot(confmatrix_Input, foldername, class_num, target_names, model_type):    
    #pd.DataFrame(confmatrix_Input).to_csv('confusion_matrix.csv')
    clsnames = np.arange(0, class_num)
    tick_marks = np.arange(len(clsnames))
    plt.figure(figsize=(class_num+1, class_num+1))
    plt.title('Confusion matrix of ' + model_type + ' ',fontsize=8,pad=10) # vgg/resnet
    iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (confmatrix_Input.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confmatrix_Input[i, j]), fontsize=8, va='center', ha='center')  # 显示对应的数字

    plt.gca().set_xticks(tick_marks + 0.5, minor=True)
    plt.gca().set_yticks(tick_marks + 0.5, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.imshow(confmatrix_Input, interpolation='nearest', cmap='cool')  # 按照像素显示出矩阵
    plt.xticks(tick_marks,target_names) # 横纵坐标类别名
    plt.yticks(tick_marks,target_names)
   
    plt.ylabel('Actual Results',labelpad=-5,fontsize=10)
    plt.xlabel('Predicted Results',labelpad=10,fontsize=10)
    plt.ylim([len(clsnames) - 0.5, -0.5])
    plt.tight_layout()

    cb=plt. colorbar()# heatmap
   #cb.set_label('Numbers of Predict',fontsize = 15)
    plt.savefig('./Results/outcome_pictures/' + foldername + '/' + model_type + '_Original.png') # vgg/resnet

def early_stop_reshape(AroundPercentageInput, class_num):
    new = list(AroundPercentageInput)
    AroundPercentageInput_new = []
    for ns in new:
        #print(list(ns))
        count = len(list(ns))
        #print(count, type(count))
        ns = list(ns)
        ns = ns + [0.0] * (class_num-count)
        #print(ns)
        ns = np.array(ns)
        AroundPercentageInput_new.append(ns)

    col_count = class_num - len(AroundPercentageInput_new)
    for i in range(col_count):
        ns = [0.0] * (class_num)
        AroundPercentageInput_new.append(ns)

    #print(AroundPercentageInput_new)
    AroundPercentageInput_new = np.array(AroundPercentageInput_new)
    #print(AroundPercentageInput_new)
    AroundPercentageInput_new = AroundPercentageInput_new.reshape((class_num, class_num))
    return AroundPercentageInput_new

# include top = false 時才指定input_shape
def choose_model_type(model_type, weight, class_num, band_num, w, h):
    input_shape=Input(shape=(w, h, band_num))
    if model_type == 'VGG19':
        conv_base=keras.applications.VGG19(weights=weight,include_top=False, input_shape = (w, h, band_num))
    elif model_type == 'VGG16':
        conv_base=keras.applications.VGG16(weights=weight,include_top=False, input_shape = (w, h, band_num))
    elif model_type == 'AlexNet':
        x = dcnn.dcnn_net(input_shape, True)
        conv_base = keras.Model(inputs = input_shape, outputs = x, name = 'Alexnet')
    elif model_type == 'Resnet50':
        conv_base=keras.applications.resnet50.ResNet50(weights=weight, include_top=False, input_shape = (w, h, band_num))
    elif model_type == 'EfficientNet':
        conv_base=efn.EfficientNetB7(input_shape = (w, h, band_num), weights=weight, include_top=False)
    elif model_type == 'Xception':
        conv_base = keras.applications.Xception(input_shape = (w, h, band_num), weights=weight, include_top=False)
    elif model_type == 'MobileNet':
        conv_base = keras.applications.MobileNet(input_shape = (w, h, band_num), weights=weight, include_top=False)
    elif model_type == 'VGG16np':
        #x = CNN0927.VGG8(input_shape, True)
        conv_base = vgg16_1208.vgg16_net(input_shape, True, [False, False, False, False, True])
        vgg16base=keras.applications.VGG16(weights=weight,include_top=False)
        weight = vgg16base.get_weights()
        conv_base.set_weights(weight)
    
    model = model_setting(conv_base, class_num)
    model.summary()
    return model

def adapt_learning_rate(epoch):
    return 0.001 * epoch

def prep_data(path, x, batch_size, band_num):
    
    n = len(x)
    i = 0
    np.random.seed(1111)
    np.random.shuffle(x)
    np.random.seed(None)
    
    while True:
        train_data = []
        train_label = []
        for _ in range(batch_size):
            if i == 0:
               np.random.shuffle(x)
            name = x[i].split(' ')[0]
            
            # prepare image information
            if band_num != 3:
                img = multi_channel.merge_test(name)
                train_data.append(img)
            else:
                img = np.load(name)
                np.resize(img, (WIDTH, HEIGHT))
                #img = np.array(img)
                img = img/256
                train_data.append(img)
            
            train_label.append(int(x[i].split(' ')[1]))
            i = (i+1) % n

        yield (np.array(train_data), np.array(train_label)) # make this function to a generator

def main():

# python new_vgg19.py airphoto_lc_256_256_DeSample 1 256 256 32 65535 EfficientNet
    parser = argparse.ArgumentParser()
    parser.add_argument("band_set", help="這是第 1 個引數，請輸入執行dataset")
    parser.add_argument("band_num", help="這是第 2 個引數，請輸入整數", type = int)
    parser.add_argument("height", help="這是第 3 個引數，請input height", type = int)
    parser.add_argument("width", help="這是第 4 個引數，請input width", type = int)
    parser.add_argument("batch_size", help="這是第 5 個引數，請輸入batch size", type = int)
    parser.add_argument("norm", help="這是第 6 個引數，請正規化數值", type = int)
    parser.add_argument("model_type", help="這是第 7 個引數，請模型類別", type = str)
    
    args = parser.parse_args()
    print(args)
    #**************************************#
    folder_path = Path(args.band_set)

    foldername = folder_path.name
    model_type = args.model_type
    HEIGHT = args.height
    WIDTH = args.width
    batch_size = args.batch_size
    norm = args.norm
    model_type = args.model_type
    #**************************************#
    start = time.time()
    f = setup(foldername, model_type)
    epoch_no = 30 # setting
    #BATCH_SIZE=4 # setting
    #train_dataset, test_dataset, val_dataset, train_count, test_count, val_count = load_npy.call()
    '''
    if args.band_num == 3:
        train_dataset, test_dataset, val_dataset, train_count, test_count, val_count, target_names = load_dataset.call(foldername)
    else:
        train_dataset, test_dataset, val_dataset, train_count, test_count, val_count, target_names = load_dataset_tif.call(foldername)
    '''
    
    train_x, label_dict = load_dataset_tif.read_img(str(folder_path) + "/train_set/")
    test_x, label_dict = load_dataset_tif.read_img(str(folder_path) + "/test_set/")
    val_x, label_dict = load_dataset_tif.read_img(str(folder_path) + "/validation_set/")
    
    #np.random.seed(1111)
    #np.random.shuffle(train_datas)
    #np.random.seed(None)
    
    train_num = len(train_x)
    vali_num = len(val_x)
    
    label_list = []
    for d, k in label_dict.items():
        print(d, k)
        label_list.append(k)
   
    class_num = len(label_list)
    target_names = label_list
    print('train start!!!')
    
    #train_dataset=train_dataset.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
    #val_dataset=val_dataset.batch(BATCH_SIZE)
    #test_dataset=test_dataset.batch(BATCH_SIZE)
    keras=tf.keras
    # here need modify
    
    if args.band_num == 3: # resnet
        #input_shape=Input(shape=(256, 256, 3))
        #conv_base=keras.applications.Alexnet(weights='imagenet',include_top=False) # original version 
        #x = dcnn.dcnn_net(input_shape, True)
        #conv_base = keras.Model(inputs = input_shape, outputs = x, name = 'Alexnet')
        model = choose_model_type(model_type, 'imagenet', class_num, args.band_num, WIDTH, HEIGHT)
    else:
        #conv_base=keras.applications.Alexnet(weights=None,include_top=False, input_shape=(256, 256, args.band_num))
        model = choose_model_type(model_type, None, class_num, args.band_num, WIDTH, HEIGHT)
    
    model.summary()
        
    
    
    
    
    
    opt0 = keras.optimizers.Adam(lr=0.0015) #  optimizers setting
    opt1 = keras.optimizers.SGD(lr=0.001)
    model.compile(#optimizer='adam',
                        optimizer=opt0,
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc']                   
    )
    #steps_per_epoch=train_count//BATCH_SIZE # setting
    #validation_steps=val_count//BATCH_SIZE # setting
    early_stopping = EarlyStopping( # setting
    monitor='val_acc',
    min_delta=0.0001,
    patience=10
    )
    start_time = time.time()
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)
    rlrop = ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, mode='auto', verbose=1, patience=3)
    
    #batch_size = BATCH_SIZE
    history=model.fit_generator(prep_data(foldername, train_x, batch_size, args.band_num),  
                            steps_per_epoch=max(1, train_num // batch_size),
                            validation_data=prep_data(foldername, val_x, batch_size, args.band_num),  
                            validation_steps=max(1, vali_num // batch_size),
                            epochs=epoch_no,
                            initial_epoch=0,
                            callbacks=[rlrop, early_stopping]) 
    '''                        
    history=model.fit(train_dataset,
                            epochs=epoch_no,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            callbacks=[early_stopping, rlrop])
    '''
    end_time = time.time()
    print_spend_time(start_time, end_time, 'training', f, model_type)
    plot_learning_curves(history, epoch_no, foldername, model_type)
    start_time1=time.time()
    
    test_dataset = load_dataset_tif.process_ds_and_label(test_x, norm, WIDTH, HEIGHT)
    #一定要加上batch那一步 而且只能一次 loss acc
    #model.evaluate(test_dataset, verbose=0)
    end_time1 = time.time()
    print_spend_time(start_time1, end_time1, 'training', f, model_type)
    
    test_count = len(test_x)
    
    model.save('./Results/outcome_pictures/' + foldername + '/model_'  + model_type + '_'+  foldername + '.h5') #save model weights
    roc_report_out = r"./Results/outcome_pictures/" + foldername + "/roc_" + model_type + "_Original.png"
    report_folder = r"./Results/outcome_pictures/" + foldername + "/"
    confmatrix = print_confusion_matrix(model, test_dataset, test_count, batch_size, target_names, f) # test set 
    
    #ROC_curve.print_AUC_matrix(model, test_dataset, test_count, BATCH_SIZE, target_names, roc_report_out, report_folder)
    
    ###############################################################################
    matrixInput = np.array(confmatrix)
    PercentageInput = (matrixInput.T / matrixInput.astype(np.float).sum(axis=1)).T
    AroundPercentageInput = np.around(PercentageInput, decimals=3)
    print (AroundPercentageInput)
    ###############################################################################
    AroundPercentageInput = early_stop_reshape(AroundPercentageInput, class_num)
    
    '''
    json_string = model.to_json()
    j_path = 'model_cancer_' +  foldername + '.json'
    with open(j_path, 'w', newline='') as jfile:
        json.dump(json_string, jfile)
    '''
    #ConfusionMatrixPlot(AroundPercentageInput)
    ConfusionMatrixPlot(AroundPercentageInput, foldername, class_num, target_names, model_type) # 做圖
    
    end = time.time() #結束時間

    print(end - start) # 輸出耗時
    
    
    
    
    


if __name__ == '__main__':
    print('hello tensorflow!!')
    main()