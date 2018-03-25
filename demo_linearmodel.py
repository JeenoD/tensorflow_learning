"""
# 这是一个线性函数逼近的例子
# -*- coding: utf-8 -*-
# @Author : Jeeno
# @Time   : 2018/3/25 15:03
# @File   : demo_linearmodel.py
"""
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.8        # 目标线性函数

k = tf.Variable(0.)
b = tf.Variable(0.)
y = x_data * k + b          # 拟合的线性函数

# 损失函数(平均平方误差)
loss = tf.reduce_mean(tf.square(y_data - y))
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 定义训练目标：最小化损失函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2001):
        sess.run(train)
        if _ % 100 == 0:
            print(_, "次迭代:", sess.run([k, b]))

