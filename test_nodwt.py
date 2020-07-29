# from linlin.DWTFCN.dwtfcnmodel import  *
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

acc = np.load('/home/admin324/PycharmProjects/linlin/DWTFCN/dwt_acc.npy')
print(acc)

#导入数据
#inria building
data_all = (np.load("/media/admin324/000DAA61000D71C5/inria_data_set/TRAIN_DATA_RESIZE_vienna5.npy"))/255.0
label_all = np.load("/home/admin324/PycharmProjects/linlin/inria_FCN/re_labels_vienna5.npy")
test_data = data_all[0:1805]
test_label = label_all[0:1805]


# test_data = (np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_data.npy"))/255.0
# test_label = np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_re_label.npy")

batch_size = 4

fp = 0
tp = 0
fn = 0

import time
start_time = time.time()

with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('/home/admin324/PycharmProjects/linlin/DWTFCN/model32.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/admin324/PycharmProjects/linlin/DWTFCN'))

    graph = tf.get_default_graph()
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

    x = graph.get_tensor_by_name("Placeholder:0")
    y = graph.get_tensor_by_name("Placeholder_1:0")
    is_training = graph.get_tensor_by_name("Placeholder_2:0")
    dropout_value = graph.get_tensor_by_name("Placeholder_3:0")


    output = graph.get_tensor_by_name("prediction:0")

    test_accuracy_list = []

    # test
    predict = []
    test_total_batch = int(len(test_label) / batch_size)

    for j in range(0, test_total_batch):
        start = j * batch_size
        end = min(start + batch_size, test_data.shape[0])
        batch_x = test_data[start:end]
        batch_y = test_label[start:end]

        train_feed_dict1 = {x: batch_x, y:batch_y,is_training :False ,dropout_value :0.7}
        correct_pred1 = tf.equal(output,y)
        batch_acc1 = sess.run(tf.reduce_mean(tf.cast(correct_pred1, tf.float32)), feed_dict=train_feed_dict1)
        print("batch"+str(j)+"....."+str(batch_acc1) )

        test_accuracy_list.append(batch_acc1)

    #     # tp,fp,fn求P R F1
    #     batch_y = np.array(batch_y)
    #     result = sess.run(output, feed_dict=train_feed_dict1)
    #     predict.append(result)
    #     result = np.int32(result)
    #     for i in range(result.shape[0]):
    #         for m in range(result.shape[1]):
    #             for n in range(result.shape[2]):
    #
    #                 if result[i, m, n] == 1 and batch_y[i, m, n] == 1:
    #                     tp += 1
    #                 elif result[i, m, n] == 0 and batch_y[i, m, n] == 1:
    #                     fp += 1
    #                 elif result[i, m, n] == 1 and batch_y[i, m, n] == 0:
    #                     fn += 1
    # print("tp"+str(tp))
    # print("fp"+str(fp))
    # print("fn"+str(fn))
    # P = tp / (tp + fp)
    # R = tp / (tp + fn)
    # F1 = (2 * P * R) / (P + R)
    # IOU = tp / (fp + tp + fn)
    #
    print('test_acc:' + str(np.mean(test_accuracy_list)))
    # print('test_precision' + str(P))
    # print('test_recall' + str(R))
    # print('test_F1score' + str(F1))
    # print('IOU'+ str(IOU))
    #
    # predict = np.array(predict)
    # predict = predict.reshape([-1, 256, 256])
    #np.save("/media/admin324/000DAA61000D71C5/inria_data_set/predict_mul34_vienna", predict)
end_time = time.time()
dtime = (end_time - start_time)
print("程序运行时间：%.4s s" % dtime)