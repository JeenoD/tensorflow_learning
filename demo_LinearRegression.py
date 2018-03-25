"""
# @Author : Jeeno
# @Time   : 2018/3/25 15:54
# @File   : demo_LinearRegression.py
# 代码出处： https://www.youtube.com/watch?v=k3O0VCHxw10&t=120s
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机坐标点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]     # 在[-0.5, 0.5]中平均生成200个点，作为横坐标
noise = np.random.normal(0, 0.02, x_data.shape)         # 生成200个随机噪声
y_data = np.square(x_data) + noise                      # 纵坐标

# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))        # 权重值。输入层为1，中间层（隐层）为10
biases_L1 = tf.Variable(tf.zeros([1, 10]))                 # 偏执值。
Wx_plus_h_L1 = tf.matmul(x, Weights_L1) + biases_L1         # 加权求和
L1 = tf.nn.tanh(Wx_plus_h_L1)                               # 激活函数，隐藏层输出

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))        # 权重值。中间层为10，输出层为1
biases_L2 = tf.Variable(tf.zeros([1,1]))                   # 偏执值
Wx_plus_h_L2 = tf.matmul(L1, Weights_L2) + biases_L2        # 加权求和
prediction = tf.nn.tanh(Wx_plus_h_L2)                       # 激活函数，输出层输出，即拟合的结果值

# 损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 训练优化器

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 迭代20000次
    for _ in range(20000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图展示
    plt.figure()
    plt.scatter(x_data, y_data)     # 实际的坐标值
    plt.plot(x_data, prediction_value, "r-", lw=5)  # 拟合的结果
    plt.show()