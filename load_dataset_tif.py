import tensorflow as tf
import skimage
import skimage.io
from PIL import Image
import os
import numpy as np
import multi_channel 

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
            #print(path, '+', data, '/', img)
            img_path =  path + data + '/' + img
            if img_path.endswith(".tif"):# .tif or .jpg
                #img_matrix = np.asarray(Image.open(img_path).resize((128, 128))) #利用 PIL 讀取圖片並轉成numpy格式
                #print(img)
                path_and_label = img_path +  ' ' + str(i)
                #print(path_and_label)
                lists_x.append(path_and_label) # 將資料新增到 list_x
                #lists_x.append('1')
                #lists_y.append(i) # 將 label 新增到 list_y
        #print(data, 'have', len(lists), 'data')
        i+=1 #用於建立 label 的 one-hot encoding 幾個 type 就到多少 從 0 開始
    return lists_x, label_dict

def load_tif_file(file_path):
    print("file_path: ", bytes.decode(file_path), type(bytes.decode(file_path)))
    return file_path

def load_preprocess_image(img_path_train, w, h, norm):
    #img_raw_train=tf.io.read_file(img_path_train)  # 讀取 img_path_train 路徑的圖片
    
    #print(img_path_train)
    #print(type(img_path_train))
    img2numpy = multi_channel.merge_test(img_path_train)
    #img2numpy = multi_channel.ran_merge_test()
    #print(img2numpy.ndim)
    #print(img2numpy.shape)
    #print(type(img2numpy))
    img_tensor_train= tf.convert_to_tensor(img2numpy)
    
    #img_tensor_train = tf.random.uniform(shape=[256, 256, 3], minval=0, maxval=255, dtype=tf.int64) # random tensor test
    #img_tensor_train=tf.image.decode_jpeg(img_raw_train,channels=3) # 讀取 .jpg 檔, 讀取頻道 = 3 (RGB) # tif 檔從這裡改
    #print('before resize:')
    #print(img_tensor_train) # 3-band-tensor
    img_tensor_train=tf.image.resize(img_tensor_train, [w, h]) # 將圖片 resize 成 256*256
    #print(img_tensor_train.shape.as_list())
    
    #增加數據
    #img_tensor_train=tf.image.random_flip_left_right(img_tensor_train) # 將圖片左右旋轉
    #img_tensor_train=tf.image.random_flip_up_down(img_tensor_train) # 將圖片上下旋轉 
    img_tensor_train=img_tensor_train/norm
    #print('before tensor2float :')
    #print(img_tensor_train)
    #轉換 data type
    img_tensor_train=tf.cast(img_tensor_train,tf.float32) # 將 tensor 轉成 float32 的型態
    #print(type(img_tensor_train))
    
    #標準化
    img_train=img_tensor_train
    return img_train


def prep_data(path, x, y, batch_size):
    
    n = len(data)
    i = 0
    while True:
        train_data = []
        train_label = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(data)
            name = x[i]
            
            # prepare image information
            img = multi_channel.merge_test(name)
            #img = Image.open(path + name)
            
            #img = img.resize((256, 256, band_num))
            #img = np.array(img)
           
            #img = img/255
            
            train_data.append(img)
            
            
            #name = (data[i].split(' ')[1]).replace('\n', '')
            # prepare mask data
            #img = Image.open(path + name)
            #img = img.resize((int(WIDTH/2), int(HEIGHT/2)))
            #img = np.array(img)
            '''
            seg_labels = np.zeros((int(WIDTH/2), int(HEIGHT/2), NCLASSES))
            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img[:, :, 0] == c).astype(int) 
                #if img pixel value equal to class type, seg_labels value is 1 # make one hot encoding
            #seg_labels = np.reshape(seg_labels, (-1, NCLASSES)) #(128, 128, c) -> (128*128, c)
            '''
            train_label.append(y[i])
            i = (i+1) % n
        #print(np.shape(train_data))
        #print(np.shape(train_label))
        yield (np.array(train_data), np.array(train_label)) # make this function to a generator

def process_ds_and_label(xx, norm, w, h):
    lists = []
    #print(type(x))
    x = []
    y = []
    for j in xx:
        x.append(load_preprocess_image(j.split(' ')[0], w, h, norm))
        y.append(int(j.split(' ')[1]))
    
    x = tf.expand_dims(x, 0)
    print(len(x))
    #test_x, test_y, label_dict = read_img(path + "test_set/")
    #data = tf.data.Dataset.from_tensor_slices(x) # 將 x 轉成 tf.dataset 型態
    #data = data.map(load_preprocess_image) # 利用 load_preprocess_image 轉換資料型態 路徑轉成 tensor 
    data = tf.data.Dataset.from_tensor_slices(x) # 將 x 轉成 tf.dataset 型態
    print(data)
    #print(y)
    #data = data.map(lambda x: tf.py_func(load_tif_file, [x], [tf.string])) # 利用 load_preprocess_image 轉換資料型態 路徑轉成 tensor 
    #print(y)
    y = np.expand_dims(np.array(y), axis=0)
    label = tf.data.Dataset.from_tensor_slices(y) # 將 y 轉成 tf.dataset 型態
    #for l in label:
    #    print(l)
    #print(label)
    dataset = tf.data.Dataset.zip((data, label)) # 將 data 和 label 整合成一個 dataset
    print(dataset)
    return dataset

#def save_dataset(dataset):


def call(path):
    print('in load dataset!!')
    #os.environ['PROJ_LIB'] = r'C:\Program Files\GDAL\projlib'
    #train_x, train_y, label_dict = read_img(path + "/train_set/")
    #train_count = len(train_x)
    #print('train count is:')
    #print(train_count)
    test_x, label_dict = read_img(path + "/test_set/")
    test_count = len(test_x)
    #val_x, val_y, label_dict = read_img(path + "/validation_set/")
    #val_count = len(val_x)
    
    print('read image complete!!')
    #print(train_x)
    #dataset_train = process_ds_and_label(train_x, train_y)
    #print(dataset_train)
    dataset_test = process_ds_and_label(test_x, 255, 255, 65536)
    
    #dataset_val = process_ds_and_label(val_x, val_y)
    
    label_list = []
    for d, k in label_dict.items():
        #print(d, k)
        label_list.append(k)
    #print(type(dataset_train))
    #print(type(dataset_test))
    #print(type(dataset_val))
    #BATCH_SIZE = 32
    
    #train_dataset=train_dataset.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
    #val_dataset=val_dataset.batch(BATCH_SIZE)
    #test_dataset=test_dataset.batch(BATCH_SIZE)
    print('complete!!!')
    return dataset_train, dataset_test, dataset_val, train_count, test_count, val_count, label_list
    

def main():
    path = './airphoto_lc_256_256_DeSample'
    call(path)
    #img = load_preprocess_image('z')
    #print('in main:')
    #print(img)

if __name__ == '__main__':
    main()