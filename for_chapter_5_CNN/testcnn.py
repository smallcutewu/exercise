
import tensorflow as tf
import os
import cv2
from PIL import Image, ImageFilter
def weight_init(shape, name):
    '''
    获取某个shape大小的参数
    '''
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

def bias_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

def conv2d(x,conv_w):
    return tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1,size,size,1], strides = [1,size,size,1], padding='SAME')



def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          
          'image_data': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_data'], tf.uint8)
  image = tf.reshape(image, [28, 28, 3])

  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  #image = tf.cast(image, tf.float32) 

  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(filename, batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=5000)

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=4*batch_size,
        min_after_dequeue=2*batch_size)

    return images, labels

 
def inference(input_data):
  '''
  定义网络结构、向前计算过程
  '''
  with tf.name_scope('conv1'):
    w_conv1 = weight_init([7,7,3,32], 'conv1_w')
    b_conv1 = bias_init([32], 'conv1_b')
  
    #卷积之后图片大小变成100-10+1 = 91
    h_conv1 = tf.nn.relu(conv2d(input_data, w_conv1) + b_conv1)
    #池化之后图片大小变成45
    h_pool1 = max_pool(h_conv1, 2) #140

  with tf.name_scope('conv2'):
    w_conv2 = weight_init([5,5,32,64], 'conv2_w')
    b_conv2 = bias_init([64], 'conv2_b')

    #卷积之后图片大小变成 45 -5+1 = 41
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    #池化之后图片大小变成20
    h_pool2 = max_pool(h_conv2, 2) #68

  
  with tf.name_scope('fc1'):
    w_fc1 = weight_init([7*7*64, 1024], 'fc1_w')
    b_fc1 = bias_init([1024], 'fc1_b')
        
    h_fc = tf.nn.relu(tf.matmul(tf.reshape(h_pool2,[-1,7*7*64]), w_fc1)  + b_fc1)

  #keep_prob = 0.8
  #h_fc_drop = tf.nn.dropout(h_fc,keep_prob)
  with tf.name_scope('fc2'):
    w_fc2 = weight_init([1024, 10], 'fc2_w')
    b_fc2 = bias_init([10], 'fc2_b')

    h_fc2 = tf.matmul(h_fc, w_fc2) + b_fc2

  return h_fc2
array_of_img = [] #储存图片数组

# 读取图片函数
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    files=os.listdir(r"./"+directory_name)
    files.sort(key=lambda x:int(x[:-4]))
#文件按顺序读取
    for filename in files:
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
#       print(filename,"/n")
#       print(array_of_img)
read_directory("test_images")

def imageprepare(im):
    image = tf.reshape(im, [-1,28, 28, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

def train():
  '''
  训练过程
  '''


#  batch_size = 10
#  train_images, train_labels = inputs("./train.tfrecords", batch_size )
  
#  train_images=tf.reshape(train_images,[-1,28,28,3])
 # train_labels_one_hot = tf.one_hot(train_labels, 10, on_value=1.0, off_value=0.0)
  
  #因为任务比较简单，故意把学习率调小了，以拉长训练过程。
  learning_rate = 0.0001
  
    

 
  
  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  

#  tf.summary.scalar('cross_entropy_loss', cross_entropy)
#  tf.summary.scalar('train_acc', train_accuracy)
#  summary_op = tf.summary.merge_all()
 
  
  
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model/model.ckpt-10000.meta')
    #sess.run(tf.global_variables_initializer())
    number=''
    
    saver.restore(sess,"./model/model.ckpt-10000") #使用模型，参数和之前的代码保持一致
    for im in array_of_img:
        train_image=imageprepare(im)#预处理
        train_y_conv = inference(train_image)
        prediction_=tf.argmax(train_y_conv,1)
        number+=str(prediction_[0])
    print('识别结果:')
    print(number)

if __name__ == '__main__':
     train()
