from linlin.DWTFCN.FCN_nodwt import  *
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
start_time = time.time()

batch_size = 4
epoch_size = 40


#导入数据
# data = (np.load("/media/admin324/000DAA61000D71C5/massachusetts/train_data.npy"))/255.0
# label = np.load("/media/admin324/000DAA61000D71C5/massachusetts/train_re_label.npy")
data_all = (np.load("/media/admin324/000DAA61000D71C5/inria_data_set/TRAIN_DATA_RESIZE_vienna5.npy"))/255.0
label_all = np.load("/home/admin324/PycharmProjects/linlin/inria_FCN/re_labels_vienna5.npy")
# label_1 = label_1.astype("int")
# label = np.eye(6)[label_1].astype("float32")

# #massachusetts building
# test_data = (np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_data.npy"))/255.0
# test_label = np.load("/media/admin324/000DAA61000D71C5/massachusetts/test_re_label.npy")
#aerial building
data = data_all[1805:]
test_data = data_all[0:1805]
label = label_all[1805:]
test_label = label_all[0:1805]


result ,logits= dwtfcn(x,is_training,dropout_value  )
#logits= dwtfcn(x,is_training,dropout_value  )


#loss

with tf.name_scope("loss"):
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
    #loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

    var = tf.trainable_variables()
    # weights_decays = tf.nn.l2_loss(item for item in var)
    weights_decays = 0
    for item in var:
        weights_decays = tf.nn.l2_loss(item)
        weights_decays += weights_decays

    loss = loss1 + 0.004 * weights_decays
#optimazier
# rate = tf.placeholder(tf.float32,[None])
with tf.name_scope('optimazier'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002,beta1= 0.9,beta2= 0.999,epsilon= 1e-08).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#accuracy
with tf.name_scope('accuracy'):
    pred = tf.equal(result, y)
    #pred = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

epoch_acc = []
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    for epoch in range(1,epoch_size+1):
        idx = np.arange(data .shape[0])
        np.random.shuffle(idx)
        current_data = data[idx]
        current_label = label[idx]
        #
        # RATE = rate * 0.6

        for step in range(0, data.shape[0] // batch_size):
            start = step * batch_size
            end = min(start + batch_size, data.shape[0])
            batch_x = current_data[start:end]
            batch_y = current_label[start:end]

            feed_dict1 = {x:batch_x ,y:batch_y,is_training :True,dropout_value :0.5 }
            _,batch_loss,batch_accuracy = sess.run([optimizer ,loss,accuracy ],feed_dict= feed_dict1 )
            line = "inriaepoch: %d/%d, start:%d ,end:%d,train_loss: %.4f, train_acc: %.4f\n" % (
                epoch, epoch_size, start, end, batch_loss, batch_accuracy)
            print(line)

        if epoch % 10 == 0:
            test_accuracy_list = []
            test_total_batch = int(len(test_label) / batch_size)
            for j in range(0, test_total_batch):
                test_start = j * batch_size
                test_end = min(test_start + batch_size, test_data.shape[0])
                test_x = test_data[test_start:test_end]
                test_y = test_label[test_start:test_end]

                test_feed_dict1 = {x: test_x, y: test_y,is_training :False ,dropout_value :1.0}
                correct_pred1 = tf.equal(result, y)
                test_acc = sess.run(tf.reduce_mean(tf.cast(correct_pred1, tf.float32)), feed_dict=test_feed_dict1)

                # test_acc = sess.run(accuracy , feed_dict=test_feed_dict1)

                test_accuracy_list.append(test_acc)
            print('test_acc:' + str(np.mean(test_accuracy_list)))

            epoch_acc.append(np.mean(test_accuracy_list))
            np.save("dwt_acc", epoch_acc)

            saver = tf.train.Saver()
            save_path = saver.save(sess,"./model32")
end_time = time.time()
dtime = (end_time - start_time) / 60
print("程序运行时间：%.4s m" % dtime)

