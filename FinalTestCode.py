#Final Test Model(TheFaceShot201906)
import tensorflow as tf
import numpy as np

# data input
input_data = 'TEST96x96_0521.csv'
ckptName = "/1559844664.7698438GrayScale_rate0.0003_epoch70_batch150.ckpt"

# X, Y definition
X = tf.placeholder(tf.float32, [None, 96 * 96 * 1], name='X-input')
Y = tf.placeholder(tf.int32, [None, 1], name='Y-input')
Y_one_hot = tf.one_hot(Y, 7)
X_img = tf.reshape(X, [-1, 96, 96, 1])

# Layer1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.local_response_normalization(L1, depth_radius=4, alpha=5e-05, beta=0.15, bias=1.3)
L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

# Layer2
W2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.local_response_normalization(L2, depth_radius=4, alpha=5e-05, beta=0.15, bias=1.3)
L2 = tf.nn.max_pool(L2, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

# Layer3
W3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.max_pool(L3, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

# Layer4
W4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.1))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.max_pool(L4, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

# Layer5
W5 = tf.Variable(tf.random_normal([6, 6, 512, 512], stddev=0.1))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 2, 2, 1], padding='SAME')
L5 = tf.nn.max_pool(L5, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

# Layer6
W6 = tf.get_variable("W6", shape=[512 * 1 * 1, 496], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([496]))
L6 = tf.nn.relu(tf.matmul(b6, W6))

# Layer7
W7 = tf.get_variable("W7", shape=[496, 100], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([100]))
L7 = tf.nn.relu(tf.matmul(b7, W7))

# Layer8
W8 = tf.get_variable("W8", shape=[100, 7], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([7]))
logits = tf.matmul(L7, W8) + b8

# define test parameters
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# test
falseList = [0, 0, 0, 0, 0, 0, 0]
trueList = [0, 0, 0, 0, 0, 0, 0]
trueCnt = 0
falseCnt = 0
test_count = 0
test_N = 10000
data_xy = np.loadtxt(input_data, dtype=np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
while i <= test_N:
    batch_test_x = data_xy[i:(i+1500), 0:-1]
    batch_test_y = data_xy[i:(i+1500), 0:[-1]]
    for p, y in zip(prediction, batch_test_y.flatten()):
        if p == int(y):
            trueCnt = trueCnt + 1
            trueList[int(y)] = trueList[int(y)] + 1
        else:
            falseCnt = falseCnt + 1
            falseList[int(y)] = falseList[int(y)] + 1
        test_count = test_count + 1
    i += 1500
sess.close()

print('Test_Accuracy:', float(trueCnt / test_count))
for i in range(0, 7, 1):
    print('face %d miss classification: %d, accuracy: %f' % (i, falseList[i], float(trueList[i] / (trueList[i] + falseList[i]))))