import os
import numpy as np
import tensorflow as tf


def load_np(path):
    ds = []
    labels = []
    folders = os.listdir(path)
    i = 0 
    for folder in folders:
        files = os.listdir(path + folder)
        for file in files:
            data = np.load(path + folder + '/' + file)
            #ds.append(tf.convert_to_tensor(data))
            ds.append(data)
            print(path + folder + '/' + file)
            labels.append(i)
        i += 1
    return np.array(ds), np.array(labels) 

def load_preprocess_image(img_path_train):
    max = 31744
    min = -3912
    #img2numpy = multi_channel.merge_test(img_path_train)
    #print(type(img_path_train))
    #print(img_path_train.shape)
    img_tensor_train= tf.convert_to_tensor(img_path_train)

    img_tensor_train=tf.image.resize(img_tensor_train, [256, 256]) # 將圖片 resize 成 256*256
    img_tensor_train=tf.cast(img_tensor_train,tf.float32) # 將 tensor 轉成 float32 的型態
    print(type(img_tensor_train))
    
    #標準化
    #img_tensor_train=img_tensor_train-min/max-min
    return img_tensor_train

def process_ds_and_label(x, y):
    lists = []
    t = 0
    for i in x:
        lists.append(load_preprocess_image(i))
        print(t)
        t += 1
    data = tf.data.Dataset.from_tensor_slices(lists) # 將 x 轉成 tf.dataset 型態
    #data = data.map(load_preprocess_image) # 利用 load_preprocess_image 轉換資料型態 路徑轉成 tensor 
    label = tf.data.Dataset.from_tensor_slices(y) # 將 y 轉成 tf.dataset 型態
    dataset = tf.data.Dataset.zip((data, label)) # 將 data 和 label 整合成一個 dataset
    return dataset

def call():
    path = 'Transverse_0707/'
    print('hello python!!')
    train_set, train_label = load_np(path + 'train_set/')
    train_count = len(train_set)
    test_set, test_label = load_np(path + 'test_set/')
    test_count = len(test_set)
    validation_set, validation_label = load_np(path + 'validation_set/')
    validation_count = len(validation_set)
    #print(len(train_set))
    ds_train = process_ds_and_label(train_set, train_label)
    ds_test = process_ds_and_label(test_set, test_label)
    ds_validation = process_ds_and_label(validation_set, validation_label)
    print(ds_train, ds_test, ds_validation, train_count, test_count, validation_count)
    #print(ds_train)
    return ds_train, ds_test, ds_validation, train_count, test_count, validation_count

if __name__ == '__main__':
    call()