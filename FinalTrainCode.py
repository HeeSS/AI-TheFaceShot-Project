#Final Train Model(TheFaceShot201906)
import tensorflow as tf
import numpy as np

# data input & initializer
learning_rate = 0.003
training_epochs = 70
input_data = 'TRAIN96x96_0521.csv'

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

# define train parameters
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
prediction = tf.argmax(logits, 1, name='output')
tf.to_int32(prediction, name='int32_output')
tf.summary.scalar('output', prediction)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
train_N = 30000
Training_Cost = 0
data_xy = np.loadtxt(input_data, dtype=np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
dirName = "./"

for epoch in range(training_epochs):
    i = 0
    while i <= train_N:
        if (i + 1500) > train_N:
            batch_train_x = data_xy[i:(i + (train_N % 1500)), 0:-1]
            batch_train_y = data_xy[i:(i + (train_N % 1500)), 0:[-1]]
        else:
            batch_train_x = data_xy[i:(i + 1500), 0:-1]
            batch_train_y = data_xy[i:(i + 1500), 0:[-1]]

        train_feed_dict = {X: batch_train_x, Y: batch_train_y}
        Training_Cost = sess.run([cost, optimizer], feed_dict=train_feed_dict)
        i += 1500

saver.save(sess, dirName + "save.ckpt")
tf.train.write_graph(sess.graph.as_graph_def(), dirName + "graph.pb")
sess.close()