"""
# -*- coding: utf-8 -*-
# @Author : Jeeno
# @Time   : 2018/3/25 14:30
# @File   : demo_fetchAndFeed.py
"""
import tensorflow as tf

# 定义占位符
num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)
# 定义op
add = tf.add(num1, num2)
mul = tf.multiply(num1, num2)

with tf.Session() as sess:  # 定义会话，用完自动关闭
   print(sess.run([add, mul], feed_dict={num1: 3.0, num2: 5.}))

# 定义两个变量
input1 = tf.Variable(3.0, tf.float32)
input2 = tf.Variable(4., tf.float32)
# 定义加法op
input_add = tf.add(input1, input2)
# input1新的值_占位符
new_input1 = tf.placeholder(tf.float32)
# 更新input1值的op
update_input1 = tf.assign(input1, new_input1)

# 初始化全部变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 执行变量初始化
    sess.run(init)
    # 执行加法op
    print(sess.run(input_add))
    # 通过feed更新input1
    sess.run(update_input1, feed_dict={new_input1: 6.0})
    # 执行加法op
    print("更新后:", sess.run(input_add))