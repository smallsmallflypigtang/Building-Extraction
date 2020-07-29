import numpy as np
import matplotlib.pyplot as plt


# data_all = (np.load("/media/admin324/000DAA61000D71C5/inria_data_set/TRAIN_DATA_RESIZE_chicago2.npy"))
# label_all = np.load("/home/admin324/PycharmProjects/linlin/inria_FCN/re_labels_chicago2.npy")
# test_data = data_all[0:1805]
# test_label = label_all[0:1805]
#
# print(test_label .shape)
predict = np.load("/media/admin324/000DAA61000D71C5/inria_data_set/predict_Segnet_tyrol_w.npy")
print(predict.shape)


plt.imshow(predict[356])
plt.axis('off')
plt.show()


# for i in range(300,test_data .shape[0]):
#     print(i)
#     m = [test_data[i],test_label[i],predict[i]]
#     n = ['image', 'gt' ,'predict']
#     for j in range(1,4):
#         plt.subplot(1, 3, j)
#         plt.title(n[j-1])
#         plt.axis('off')
#         plt.imshow(m[j-1])
#     plt.show()
