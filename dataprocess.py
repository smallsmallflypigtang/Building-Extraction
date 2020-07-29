import numpy as np
import os
import cv2

# path = ''
# filename = os.listdir(path)
# img = cv2.imread(filename)
# print(img)

#import image set
path = 'E:/aerialimagelabeling/NEW2-AerialImageDataset/AerialImageDataset/train/gt'
filenames = os.listdir(path)
names = filenames[72:108]
print("names"+str(names))
names.sort(key=lambda x:int(x[6:-4]))
print("names"+str(names))


#img_paths = []  # 所有图片路径
resizes = []
for name in names:
    #name = names[m]
    img_path = os.path.join(path, name)
    img = cv2.imread(img_path)  # 读取图片5000*5000*3
    print(img.shape)
    for i in range(19):
        for j in range(19):
            resize = img[(i*256):(i*256+256),(j*256):(j*256+256),...]   #直接切割
            resizes.append(resize)
    print(np.array(resizes).shape)
resizes = np.array(resizes)
print(resizes.shape)
np.save("TRAIN_RESIZE_gt_austin1",resizes)



