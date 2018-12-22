import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random

image_train_path = './data/flower_photos/'
tfRecord_train = './tf/flower_train.tfrecords'
tfRecord_test = './tf/flower_test.tfrecords'
data_path = './tf'
image_shape = [1, 224, 224, 3]
image_size = (224, 224)

# image_shape = [1, 512, 512, 1]

label_encoder = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


def write_tfRecord(train_name, test_name, image_path):
    train_writer = tf.python_io.TFRecordWriter(train_name)
    test_writer = tf.python_io.TFRecordWriter(test_name)
    train_cnt = 0
    test_cnt = 0

    for flower_name in os.listdir(image_path):
        if flower_name == 'LICENSE.txt':
            continue
        dir_path = os.path.join(image_path, flower_name)
        for img_name in os.listdir(dir_path):
            img = Image.open(os.path.join(dir_path, img_name))
            img = img.resize(image_size, Image.ANTIALIAS)
            label = label_encoder[flower_name]
            labels = [0] * 5
            labels[label] = 1

            example = tf.train.Example(features=tf.train.Features(feature={
                'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                'Y': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
            if random.random() < 0.7:
                train_writer.write(example.SerializeToString())
                train_cnt += 1
            else:
                test_writer.write(example.SerializeToString())
                test_cnt += 1

    train_writer.close()
    test_writer.close()
    print('write %d image for train, % image for test' % (train_cnt, test_cnt))
    print("write tfrecord successful")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("create data dir")
    else:
        print('data dir already exists')
    write_tfRecord(tfRecord_train, tfRecord_test, image_train_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'X': tf.FixedLenFeature([], tf.string),
                                           'Y': tf.FixedLenFeature([5], tf.int64)
                                       })
    X = tf.decode_raw(features['X'], tf.uint8)
    X.set_shape([image_size[0] * image_size[1] * 3])
    X = tf.reshape(X, image_shape)
    X = tf.cast(X, tf.float32) / 128 - 1
    Y = tf.cast(features['Y'], tf.float32)

    return X, Y


def get_tfrecord(num, isTrain=True):
    # image_shape[0] = num
    tfRecord_path = tfRecord_train if isTrain else tfRecord_test

    X, Y = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([X, Y], batch_size=num, num_threads=2, capacity=5000,
                                                    min_after_dequeue=1000)
    return img_batch, label_batch
    # return X, Y


# rebuild image to check
def est_get_tfrecord():
    x, y = get_tfrecord(1, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in [1, 2, 3]:
            xs, ys = sess.run([x, y])
            arr = ((xs + 1) * 128).astype(np.uint8)
            arr = arr.reshape([224, 224, 3])
            print(arr.shape)
            print(arr.dtype)
            img = Image.fromarray(arr)
            img.save('%d.jpg' % i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # generate_tfRecord()
    est_get_tfrecord()
