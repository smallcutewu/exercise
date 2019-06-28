import tensorflow as tf
import os
import cv2
'''
本文件由图片生成tfrecord文件，保存的信息有图片的像素信息，图片的高和宽，图片的通道数，
图片对应的label，也就是图片是1还是0.
会生成训练的tfrecord文件和验证的tfrecord文件，
还会生成一个用来展示卷积过程中图片经过卷积变化的图片的tfrecord文件。
'''

def gen_tfrecord(output_tfrecord_file):
  '''
  读取文件夹中的图片数据， 生成tfrecord格式的文件
  
  Args:
    zero_dir: 保存图片0的文件夹
	one_dir: 保存图片1的文件夹
	output_tfrecord_file: 输出的tfrecord文件
	
  Return:
  
  '''
  tf_writer = tf.python_io.TFRecordWriter(output_tfrecord_file)

  #为数字0的数据
  for file in os.listdir("./train_images/0"):
    file_path = os.path.join("./train_images/0", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 0

    example = tf.train.Example()

    feature = example.features.feature
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  
  #为数字1的数据
  for file in os.listdir("./train_images/1"):
    file_path = os.path.join("./train_images/1", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 1

    example = tf.train.Example()

    feature = example.features.feature
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())

  #为数字2的数据
  for file in os.listdir("./train_images/2"):
    file_path = os.path.join("./train_images/2", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 2

    example = tf.train.Example()

    feature = example.features.feature
   
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字3的数据
  for file in os.listdir("./train_images/3"):
    file_path = os.path.join("./train_images/3", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 3

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字4的数据
  for file in os.listdir("./train_images/4"):
    file_path = os.path.join("./train_images/4", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 4

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字5的数据
  for file in os.listdir("./train_images/5"):
    file_path = os.path.join("./train_images/5", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 5

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字6的数据
  for file in os.listdir("./train_images/6"):
    file_path = os.path.join("./train_images/6", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 6

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字7的数据
  for file in os.listdir("./train_images/7"):
    file_path = os.path.join("./train_images/7", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 7

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字8的数据
  for file in os.listdir("./train_images/8"):
    file_path = os.path.join("./train_images/8", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 8

    example = tf.train.Example()

    feature = example.features.feature
   
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())
  #为数字9的数据
  for file in os.listdir("./train_images/9"):
    file_path = os.path.join("./train_images/9", file)
    image_data = cv2.imread(file_path)
    image_bytes = image_data.tostring()
    rows = image_data.shape[0]
    cols = image_data.shape[1]
    channels = image_data.shape[2]
    label_data = 9

    example = tf.train.Example()

    feature = example.features.feature
    
    feature['image_data'].bytes_list.value.append(image_bytes)
    feature['label'].int64_list.value.append(label_data)

    tf_writer.write(example.SerializeToString())

  tf_writer.close()



  
if __name__ == "__main__":
  gen_tfrecord( "./train.tfrecords")
  #生成展示卷积网络中间过程图形化的测试数据
print("generate tfrecord data complete!")
