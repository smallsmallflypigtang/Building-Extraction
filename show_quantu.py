import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_all = (np.load("/media/admin324/000DAA61000D71C5/inria_data_set/TRAIN_DATA_RESIZE_vienna5.npy"))/255.
label_all = np.load("/home/admin324/PycharmProjects/linlin/inria_FCN/re_labels_vienna5.npy")
test_data = data_all[0:1805]
test_label = label_all[0:1805]
# test_data = (np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_data.npy"))/255.0
# test_label = np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_re_label.npy")
print(test_label .shape)
predict = np.load("/media/admin324/000DAA61000D71C5/inria_data_set/predict_0_vienna.npy")
print(predict.shape)

data_all = np.zeros((5,4864,4864,3))
for p in range(5):
    for m in range(19):
        for n in range(19):
            data_all[p,m*256:(m+1)*256,n*256:(n+1)*256,...] = test_data[p*361+m*19+n]
print(data_all.shape)


label_all = np.zeros((5,4864,4864))
for p in range(5):
    for m in range(19):
        for n in range(19):
            label_all[p,m*256:(m+1)*256,n*256:(n+1)*256,...] = test_label[p*361+m*19+n]
print(label_all.shape)


predict_all = np.zeros((4,4864,4864))
for p in range(4):
    for m in range(19):
        for n in range(19):
            predict_all[p,m*256:(m+1)*256,n*256:(n+1)*256,...] = predict[p*361+m*19+n]
print(predict_all.shape)



for i in range(4):
    m = [data_all[i],label_all[i],predict_all[i]]
    n = ['image', 'gt' ,'predict']
    for j in range(1,4):
        plt.subplot(1, 3, j)
        plt.title(n[j-1])
        plt.axis('off')
        plt.imshow(m[j-1])
    plt.show()