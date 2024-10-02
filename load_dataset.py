import tensorflow as tf
import skimage
import skimage.io
from PIL import Image
import os
import numpy as np


def read_img(path):
    datas = os.listdir(path)
    #img_list = []
    lists_x = [] #data
    lists_y = [] #label
    label_dict = {}
    i = 0
    for data in datas:
        imgs = os.listdir(path + data)
        label_dict[i] = str(data)
        for img in imgs:
            print(path, '+', data, '/', img)
            img_path =  path + data + '/' + img
            if img_path.endswith(".jpg"):
                #img_matrix = np.asarray(Image.open(img_path).resize((128, 128))) #利用 PIL 讀取圖片並轉成numpy格式
                #print(img)
                #print(img_path)
                lists_x.append(img_path) # 將資料新增到 list_x
                lists_y.append(i) # 將 label 新增到 list_y
        #print(data, 'have', len(lists), 'data')
        i+=1 #用於建立 label 的 one-hot encoding 幾個 type 就到多少 從 0 開始
    return lists_x, lists_y, label_dict
    
def load_preprocess_image(img_path_train):
    img_raw_train=tf.io.read_file(img_path_train)  # 讀取 img_path_train 路徑的圖片
    
    #img_tensor_train=
    
    img_tensor_train=tf.image.decode_jpeg(img_raw_train,channels=3) # 讀取 .jpg 檔, 讀取頻道 = 3 (RGB) # tif 檔從這裡改
    
    print(img_tensor_train) # 3-band-tensor
    img_tensor_train=tf.image.resize(img_tensor_train,[256,256]) # 將圖片 resize 成 256*256
    
    #增加數據 data augmentation
    #img_tensor_train=tf.image.random_flip_left_right(img_tensor_train) # 將圖片左右旋轉
    #img_tensor_train=tf.image.random_flip_up_down(img_tensor_train) # 將圖片上下旋轉 
    
    #轉換 data type
    img_tensor_train=tf.cast(img_tensor_train,tf.float32) # 將 tensor 轉成 float32 的型態
    #標準化
    img_train=img_tensor_train/255 
    return img_train


def process_ds_and_label(x, y):
    data = tf.data.Dataset.from_tensor_slices(x) # 將 x 轉成 tf.dataset 型態
    data = data.map(load_preprocess_image) # 利用 load_preprocess_image 轉換資料型態 路徑轉成 tensor 
    label = tf.data.Dataset.from_tensor_slices(y) # 將 y 轉成 tf.dataset 型態
    dataset = tf.data.Dataset.zip((data, label)) # 將 data 和 label 整合成一個 dataset
    return dataset


def call(path):
    #path = 'Coronal_0707/'
    if not path.endswith('/'):
        path = path + '/'
    train_x, train_y, label_dict = read_img(path + "train_set/")
    train_count = len(train_x)
    test_x, test_y, label_dict = read_img(path + "test_set/")
    test_count = len(test_x)
    val_x, val_y, label_dict = read_img(path + "validation_set/")
    val_count = len(val_x)
    
    dataset_train = process_ds_and_label(train_x, train_y)
    
    dataset_test = process_ds_and_label(test_x, test_y)
    
    dataset_val = process_ds_and_label(val_x, val_y)
    
    #print(len(label_dict))
    label_list = []
    for d, k in label_dict.items():
        print(d, k)
        label_list.append(k)
    #print(type(dataset_train))
    #print(type(dataset_test))
    #print(type(dataset_val))
    #BATCH_SIZE = 32
    
    #train_dataset=train_dataset.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
    #val_dataset=val_dataset.batch(BATCH_SIZE)
    #test_dataset=test_dataset.batch(BATCH_SIZE)
    
    return dataset_train, dataset_test, dataset_val, train_count, test_count, val_count, label_list
    

def main():
    call()

if __name__ == '__main__':
    main()