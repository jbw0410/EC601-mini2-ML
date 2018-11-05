import tensorflow as tf
import numpy as np
import os

train_dir = '/home/ece-student/deep learning/train/'

def get_files(file_dir):
    A5 = []
    label_A5 = []
    A6 = []
    label_A6 = []
    SEG = []
    label_SEG = []
    SUM = []
    label_SUM = []
    LTAX1 = []
    label_LTAX1 = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='A5':
            A5.append(file_dir+file)
            label_A5.append(1)
        elif name[0] == 'A6':
            A6.append(file_dir+file)
            label_A6.append(2)
        elif name[0]=='LTAX1':
            LTAX1.append(file_dir+file)
            label_LTAX1.append(3)
        elif name[0] == 'SEG':
            SEG.append(file_dir+file)
            label_SEG.append(4)
        else:
            SUM.append(file_dir+file)
            label_SUM.append(5)
    print('There are %d A5\nThere are %d A6\nThere are %d LTAX1\nThere are %d SEG\nThere are %d SUM' \
          %(len(A5),len(A6),len(LTAX1),len(SEG),len(SUM)))
    
    image_list = np.hstack((A5,A6,LTAX1,SEG,SUM))
    label_list = np.hstack((label_A5,label_A6,label_LTAX1,label_SEG,label_SUM))
    
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    
    return  image_list,label_list


def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
    input_queue = tf.train.slice_input_producer([image,label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = capacity)
    
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    
    

import matplotlib.pyplot as plt
BATCH_SIZE = 16
CAPACITY = 64
IMG_W = 208
IMG_H = 208

train_dir = '/home/ece-student/deep learning/train/'

image_list,label_list = get_files(train_dir)
image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

with tf.Session() as sess:
    i=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    try:
        while not coord.should_stop() and i<1:
            img,label = sess.run([image_batch,label_batch])
            
            for j in np.arange(BATCH_SIZE):
                print('label: %d'%label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
    
    
    
            
    
    
    
    
    
    