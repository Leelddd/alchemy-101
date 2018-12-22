# _*_encoding=utf-8_*_

import tensorflow as tf
import generateds

batch_size = 32
all_poch = 2000
train_num = 1000
test_num = 360

# 训练个数一共有1360张图片
ox17_image_width = 224
ox17_image_height = 224
ox17_num_labels = 5  # 17个类别

# 前1000张图片作为训练集，后360张图片作为测试集

images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 5])
drop_prob = tf.placeholder(dtype=tf.float32, shape=[])


# 分别获得名字和你的shape大小
def print_activation(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def full_weight(shape):
    return tf.Variable((tf.random_normal(shape=shape, stddev=1e-4, dtype=tf.float32)))


def full_bias(shape):
    return (tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape)))


# 定义网络
def Net(images, drop):
    parameters = []
    l2_loss = tf.constant(0.1)
    #  layer 1  前两层使用了lrn方法防止过拟合操纵了
    with tf.name_scope("conv1") as scopes:
        weight = tf.Variable(tf.truncated_normal([11, 11, 3, 64], stddev=0.1, dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(images, weight, strides=[1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]), name="biases")
        bias = tf.add(conv, biases)
        conv1_output = tf.nn.relu(bias)
        print_activation(conv1_output)
    lrn1 = tf.nn.lrn(conv1_output, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
    pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME", name="pool1")
    print_activation(pool1)
    parameters += [weight, biases]

    # layer 2
    with tf.name_scope("conv2") as scopes:
        weight = tf.Variable(tf.truncated_normal([5, 5, 64, 192], stddev=0.1, dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(pool1, weight, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[192]), name="biases")
        bias = tf.add(conv, biases)
        conv2_output = tf.nn.relu(bias)
        print_activation(conv2_output)
    lrn2 = tf.nn.lrn(conv2_output, 4, 1.0, 0.001 / 9, 0.75, "lrn2")
    pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME", name="pool2")
    print_activation(pool2)
    parameters += [weight, biases]

    # layer 3
    with tf.name_scope("conv3") as scopes:
        weight = tf.Variable(tf.truncated_normal([3, 3, 192, 384], stddev=0.1, dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(pool2, weight, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384]), name="biases")
        bias = tf.add(conv, biases)
        conv3_output = tf.nn.relu(bias)
        print_activation(conv3_output)
    parameters += [weight, biases]

    # layer 4
    with tf.name_scope("conv4") as scopes:
        weight = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1, dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(conv3_output, weight, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256]), name="biases")
        bias = tf.add(conv, biases)
        conv4_output = tf.nn.relu(bias)
        print_activation(conv4_output)
    parameters += [weight, biases]

    # layer 5
    with tf.name_scope("conv5") as scopes:
        weight = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1, dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(conv4_output, weight, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256]), name="biases")
        bias = tf.add(conv, biases)
        conv5_output = tf.nn.relu(bias)
        print_activation(conv5_output)
    parameters += [weight, biases]
    pool5 = tf.nn.max_pool(conv5_output, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME", name="pool5")
    print_activation(pool5)

    with tf.name_scope("full_layer") as scopes:
        full_input = tf.reshape(pool5, [-1, 7 * 7 * 256])
        W1 = full_weight([7 * 7 * 256, 4096])
        b1 = full_bias([4096])
        l2_loss += tf.nn.l2_loss(W1)
        l2_loss += tf.nn.l2_loss(b1)
        full_mid1 = tf.nn.relu(tf.matmul(full_input, W1) + b1)
        full_drop1 = tf.nn.dropout(full_mid1, keep_prob=drop)

        W2 = full_weight([4096, 4096])
        b2 = full_bias([4096])
        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(b2)
        full_mid2 = tf.nn.relu(tf.matmul(full_drop1, W2) + b2)
        full_drop2 = tf.nn.dropout(full_mid2, keep_prob=drop)

        W3 = full_weight([4096, 17])
        b3 = full_bias([17])
        l2_loss += tf.nn.l2_loss(W3)
        l2_loss += tf.nn.l2_loss(b3)
        output = tf.matmul(full_drop2, W3) + b3

    return output, l2_loss


out, l2_loss = Net(images, drop_prob)

cross_entry = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
# cross_entry = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out),reduction_indices=1))
optimizer = tf.train.AdamOptimizer(0.00001).minimize(cross_entry)

correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
# Calculate accuracy

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

img_batch, label_batch = generateds.get_tfrecord(batch_size, True)

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 训练200次
        for epoch in range(all_poch):
            ave_cost = 0

            xs, ys = sess.run([img_batch, label_batch])
            cost, _ = sess.run([cross_entry, optimizer], feed_dict={images: xs, y: ys, drop_prob: 0.75})

            ave_cost += cost / batch_size
            if (epoch + 1) % 100 == 0:
                print("epoch : %04d" % (epoch + 1), " ", "cost :{:.9f}".format(ave_cost))
                accur = sess.run(accuracy, feed_dict={images: xs, y: ys, drop_prob: 1})
                print('accuracy after %d step: %f' % (epoch, accur))
        # num1 = int(test_dataset_ox17.shape[0] / batch_size)
        #
        # pred = 0.0
        # for i in range(num1):
        #     start = (i * batch_size) % test_dataset_ox17.shape[0]
        #     end = min(start + batch_size, test_dataset_ox17.shape[0])
        #     accu = sess.run(accuracy, feed_dict={images: test_dataset_ox17[start:end], y: test_labels_ox17[start:end],
        #                                          drop_prob: 1})
        #     # print(train_dataset_[start:end].shape)
        #     pred += accu / num1
        # # print(accuracy.eval(feed_dict={images: test_dataset_ox17[start:end], y: test_labels_ox17[start:end]}))
        # print("accuracy:{:.9f} ".format(pred))
