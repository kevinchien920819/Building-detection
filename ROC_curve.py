import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
import tensorflow as tf
import load_dataset
import argparse
from tensorflow import keras
from scipy import interp
import json


#def load_weight(path):
#    w = keras.models.load_weights(path)
#    return w
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def load_data(foldername):
    train_dataset, test_dataset, val_dataset, train_count, test_count, val_count, target_names = load_dataset.call(foldername)
    return test_dataset, test_count, target_names

def find_youden_index(tpr, fpr, threshold):
    y = tpr - fpr
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [fpr[Youden_index], tpr[Youden_index]]
    return optimal_threshold, point

def predict_by_threshold(prob, threshold, foldername):
    new_pred = []
    print(threshold)
    for i in prob:
        if i[1] >= threshold:
            new_pred.append(1)
        else:
            new_pred.append(0)
    
    print(len(new_pred))
    #print(new_pred)
    filename = foldername + '_result.json'
    with open(filename, 'w', newline='') as  jfile:
        json.dump(new_pred, jfile)
    
    
    return new_pred

def print_AUC_matrix(model, test_dataset, test_count, BATCH_SIZE, target_names, path, foldername): # test set
    print('in ROC')
    print(path) # test set
    #print('in ROC'))
    x, y_true = [], []
    i = 0
    for element in test_dataset:
        i += 1
        _x, _y = element	
        x.append(_x.numpy())
        y_true.append(_y.numpy())
        if i==test_count//BATCH_SIZE:
            break
    #print(x)
    #print(y_true)
    
    x = np.concatenate(x, axis=0)
    y_true = np.concatenate(y_true)
    y_prob = model.predict_proba(x, verbose=0) # predict 
    
    if len(target_names) == 2:
        y_pred = []
        for y in y_prob:
            #print(y[1])
            y_pred.append(y[1])
        #y_pred = np.argmax(y_pred, axis=-1)
        #print(y_true)
        #print(y_pred)
       
        
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        #print(fpr, tpr, threshold)
        
        
        auc1 = auc(fpr, tpr)
        op_th, op_point = find_youden_index(tpr, fpr, threshold)
        
        #new_pred = predict_by_threshold(y_prob, op_th, foldername)
        
        plt.cla()
        plt.title('ROC')
        plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)
        plt.plot(op_point[0], op_point[1], color='blue', marker = 'o')
        plt.text(op_point[0], op_point[1], f'Threshold:{op_th:.2f}')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        #plt.xlim([0, 1])
        #plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig(path)
        #plt.show()
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_pred = []
        #print(y_true)
        y_true0 = np.eye(len(target_names))[y_true]
        print(y_true0)
        nny_true = []
        for y in y_true0:
            yl = list(y)
            yll = []
            for i in yl:
                yll.append(int(i))
            nny_true.append(yll)
        
        #print(y_true[0][0])
        for y in y_prob:
            #print(y[1])
            y_pred.append(list(y))
        
        #print(y_pred)
        
        ny_true = []
        ny_pred = []
        for i in range(len(nny_true[0])):
            ny_truel = []
            ny_predl = []
            for x, y in zip(nny_true, y_pred):
                ny_truel.append(x[i])
                ny_predl.append(y[i])
            ny_true.append(ny_truel)
            ny_pred.append(ny_predl)
        
        
        #print(ny_true)
        #print(ny_pred)
        
        for i in range(len(target_names)):
            fpr[i], tpr[i], _ = roc_curve(ny_true[i], ny_pred[i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        
        #fpr["micro"], tpr["micro"] , _ = roc_curve(y_true.ravel(), y_prob.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(target_names)):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr = mean_tpr / len(target_names)
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        
        lw = 2
        plt.cla()
        plt.title('ROC multi_class')
        #plt.plot(fpr["micro"], tpr["micro"], color='orange', label='micro-average AUC = %0.2f' % roc_auc["micro"])
        plt.plot(fpr["macro"], tpr["macro"], color='navy', label='macro-average AUC = %0.2f' % roc_auc["macro"])
        cmap = get_cmap(len(target_names))
        for i in range(len(target_names)):
            plt.plot(fpr[i], tpr[i], lw=lw, color = cmap(i), label='ROC of {0}, AUC = {1:02f}' .format (target_names[i], roc_auc[i]))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig(path)
    
def model_setting(conv_base, class_num): #
    layers=tf.keras.layers # define
    model=keras.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(class_num,activation='softmax')) #改成分類數量
    conv_base.trainable=False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("band_set", help="這是第 1 個引數，請輸入執行dataset")
    parser.add_argument("modelPath", help="引數2，輸入model路徑")
    parser.add_argument("batch", help="引數3，輸入batch大小")
    #parser.add_argument("band_num", help="這是第 2 個引數，請輸入整數", type = int)
    args = parser.parse_args()
    print(tf.__version__)
    print(tf.enable_eager_execution())
    print(tf.test.is_gpu_available())
    foldername = args.band_set
    model_path = args.modelPath
    batch_size = int(args.batch)
    
    td, tc, target_names = load_data(foldername)
    class_num = len(target_names)
    try:
        model = tf.keras.models.load_model(model_path)
    except ValueError:
        conv_base=keras.applications.VGG19(weights='imagenet',include_top=False) # original version 
        model = model_setting(conv_base, class_num)
        model.summary()
        opt0 = keras.optimizers.Adam(lr=0.001) #  optimizers setting
        opt1 = keras.optimizers.SGD(lr=0.001)
        model.compile(#optimizer='adam',
                            optimizer=opt1,
                            loss='sparse_categorical_crossentropy',
                            metrics=['acc']                   
        )
        
        w = tf.keras.models.Model.load_weights(model, model_path)
        
    
    td=td.batch(batch_size)
    model.evaluate(td,verbose=0)
    file_name = foldername + '_test.png'
    print_AUC_matrix(model, td, tc, batch_size, target_names, file_name, foldername)
    

if __name__ == '__main__':
    main()


## Transverse_0911
## E:\Obstetrics\dcm_reader\Results\outcome_pictures\Transverse_0911\model_cancer_Transverse_0911.h5
## 4
## E:/Obstetrics/dcm_reader/Results/outcome_pictures/Transverse_1227/model_cancer_VGG16np_Transverse_1227.h5
## Transverse_1227



## [0.         0.18900077 1.        ] [0.         0.67106284 1.        ] [2 1 0] auc = 0.74
## [0.         0.20681642 1.        ] [0.         0.66951125 1.        ] [2 1 0] auc=0.74
