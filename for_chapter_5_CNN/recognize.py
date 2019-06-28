from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
def imageprepare(im): 
#     im = Image.open('C:/Users/考拉拉/Desktop/.png') #读取的图片所在路径，注意是28*28像素
#    im = im.convert('L')
 #  plt.imshow(im)  #显示需要识别的图片
#   plt.show()
    tv = list(im.getdata()) 
#     tva = [(255-x)*1.0/255.0 for x in tv] 
    return tv

array_of_img = [] #储存图片数组

# 读取图片函数
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    files=os.listdir(r"./"+directory_name)
    files.sort(key=lambda x:int(x[:-5]))
#文件按顺序读取
    for filename in files:
        #print(filename) #just for test
        #img is used to store the image data 
        img = Image.open(directory_name + "/" + filename)
        array_of_img.append(img)
#       print(filename,"/n")
#       print(array_of_img)
read_directory("test_images")
#im=Image.open("./test_images/train_11.bmp")
#result=imageprepare(im)
learning_rate = 1e-4
keep_prob_rate = 0.7 # 
max_epoch = 2000
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    # 每一维度  滑动步长全部是 1， padding 方式 选择 same
    # 提示 使用函数  tf.nn.conv2d
    
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    # 滑动步长 是 2步; 池化窗口的尺度 高和宽度都是2; padding 方式 请选择 same
    # 提示 使用函数  tf.nn.max_pool
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#  卷积层 1
## conv1 layer ##

W_conv1 =  weight_variable([7, 7, 1, 32])                      # patch 7x7, in size 1, out size 32
b_conv1 =  bias_variable([32])                    
h_conv1 =  tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                    # 卷积  自己选择 选择激活函数
h_pool1 =  max_pool_2x2(h_conv1)                     # 池化               

# 卷积层 2
W_conv2 =  weight_variable([5, 5, 32, 64])                      # patch 5x5, in size 32, out size 64
b_conv2 =  bias_variable([64])
h_conv2 =  tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                      # 卷积  自己选择 选择激活函数
h_pool2 =  max_pool_2x2(h_conv2)                      # 池化

#  全连接层 1
## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层 2
## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 交叉熵函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

saver=tf.train.Saver()
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    number=''
    saver.restore(sess,"./model.ckpt-1") #使用模型，参数和之前的代码保持一致
    for im in array_of_img:
        result=imageprepare(im)#预处理
        prediction_=tf.argmax(prediction,1)
        predint=prediction_.eval(feed_dict={xs: [result],keep_prob: 1.0}, session=sess)
        number+=str(predint[0])
    print('识别结果:')
    print(number)


