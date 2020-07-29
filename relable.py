import numpy as np

train_label = np.load("F:/Python/inria_dataprocess/TRAIN_RESIZE_gt_austin1.npy")
print(train_label.shape)

#train815  test302
re_labels = np.zeros((12996,256,256))
labels = np.zeros((256,256))
for m in range(train_label.shape[0]):
    a = train_label[m]
    for i in range(256):
        for j in range(256):
            if a[i, j, 0] == 255 and a[i, j, 1] == 255 and a[i, j, 2] == 255:
                labels[i, j] = 1
            else:
                labels[i, j] = 0
    re_labels[m,...] = labels
re_labels = np.array(re_labels)
print(re_labels.shape)
np.save("TRAIN_RESIZE_relabel_austin1",re_labels)
