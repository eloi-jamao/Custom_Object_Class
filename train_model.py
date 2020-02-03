import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
tflogger = tf.get_logger()
tflogger.setLevel('ERROR')

def create_dataset(num_epochs, batch_size):
    """
    :param str dataset_path: The path to the CSV file describing the dataset. Each row should be (image_name, label)
    :param str images_dir: The directory holding the images. The full image path is then the concatenation of the
    images_dir and the image_name obtained from the CSV
    :param int num_epochs: Number of epochs to generate samples
    :param int batch_size: Number of samples per batch
    :return:
    """
    dataset = tf.data.Dataset.from_generator(generator = lambda: data_generator(),
                                             output_types = (tf.string, tf.string),
                                             output_shapes = (tf.TensorShape([]),
                                             tf.TensorShape([])))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(100)
    dataset = dataset.map(create_sample)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    return dataset

def data_generator():
    img_dir = (os.getcwd() + r'/images')
    objects = os.listdir(img_dir)
    for img_folder in objects:
        for img in os.listdir(img_dir + '/' + img_folder):
            img_name = img_dir + '/' + img_folder + '/' + img
            yield img_name, img_folder

def create_sample(image_path, label):
    with tf.name_scope('create_sample'):
        with tf.name_scope('read_image'):
            raw_image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(raw_image, channels=3)

        with tf.name_scope('preprocessing'):
            #mean_channel = [123.68, 116.779, 103.939]
            image = tf.cast(image, dtype=tf.float32)
            #image = tf.subtract(image, mean_channel, name='mean_substraction')
            image = tf.image.resize(image, size=(350, 350))

        with tf.name_scope('data_augmentation'):
            image = tf.image.random_crop(image, size=(224, 224, 3))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=20)
            image = tf.divide(image, tf.constant(255.0, dtype = tf.float32), name ='0_1_normalization')

    return image, label

def build_model(feat_ext_url = None, num_classes):
    if feat_ext_url == None:
        feature_extractor_url = 'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4'
    with tf.name_scope('Build model'):
        feat_ext = hub.KerasLayer(feature_extractor_url, trainable = False)
        dense1 = layers.Dense(1000, activation = 'relu')
        drop1 = layers.Dropout(0.5)
        dense2 = layers.Dense(1000, activation = 'relu')
        drop2 = layers.Dropout(0.5)
        dense3 = layers.Dense(num_classes, activation = 'relu')
        model = tf.keras.Sequential([feat_ext, dense1,
                                    drop1, dense2,
                                    drop2, dense3,
                                    ])

        model.build([None, 224, 224, 3])  # Batch input shape.
        print(model.summary)
    return model

def show_samples(images):

    fig = plt.figure()
    for i,img in enumerate(images[:4]):
        fig.add_subplot(2, 2, (i+1))
        plt.imshow((img*255.0).astype(np.uint8))
    print(f'Batch size : {len(images)}, Image size: {images[0].shape}, Images type = {type(images[0])}')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-o', '--objects', default='all', help='Objects to train')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-ld', '--logdir', default=(os.getcwd() + r'/tf_logs'), help='Location of saved tf.summary')
    args = parser.parse_args()

    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/cpu:0'):  # To force the graph operations of the input pipeline to be placed in the CPU
            with tf.name_scope('input_pipeline'):
                dataset = create_dataset(args.num_epochs, args.batch_size)
                iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
                batch = iterator.get_next()
                images, labels = batch

            with tf.device('/cpu:0'): # Here the ops should go to the GPU
                with tf.name_scope('Model training'):
                    model = build_model()
                    model.compile(
                                  optimizer=tf.keras.optimizers.Adam(),
                                  loss='categorical_crossentropy',
                                  metrics=['acc'])

                    steps= np.ceil(len(images)/len(args.batch_size))
                    history = model.fit_generator(image_data,
                                                  epochs=args.epochs,
                                                  steps_per_epoch=steps,
                                                  )
            with tf.device('/cpu:0'): # Here the ops should go to the GPU


    with tf.compat.v1.Session(graph=graph) as sess:
        try:
            images, labels = sess.run(batch)
            show_samples(images)

            #writer = tf.compat.v1.summary.FileWriter(args.logdir, graph = graph) #Uncomment to save graph

        except tf.errors.OutOfRangeError:
            pass
