from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Conv2D, Reshape, LeakyReLU, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal as RN, Constant

#from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10

from PIL import Image
import numpy as np
import argparse
import math

image_size = 32
iterations = 3000

#メインの関数を定義する
def main():
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)

#Discreminatorモデル
def D_model(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    
    x = Conv2D(32, (5, 5), padding='same', strides=(2,2), name='d_conv1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    #x = InstanceNormalization()(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv1_bn')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64, (5, 5), padding='same', strides=(2,2), name='d_conv2',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv2_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (5, 5), padding='same', strides=(2,2), name='d_conv3',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv3_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(256, (5, 5), padding='same', strides=(2,2), name='d_conv4',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv4_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    #x = Dense(2048, activation='relu', name='d_dense1',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Dense(1, activation='sigmoid', name='d_out',
        kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name='D')
    
    return model

#Generatorモデル
def G_model(Height, Width, channel=3):
    inputs = Input((100,))
    
    in_h = int(Height / 16)
    in_w = int(Width / 16)
    
    d_dim = 512
    
    x = Dense(in_h * in_w * d_dim, name='g_dense1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = Reshape((in_h, in_w, d_dim), input_shape=(d_dim * in_h * in_w,))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_dense1_bn')(x)

    # 1/8
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(512, (5, 5), name='g_conv1', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(256, (5, 5), padding='same', name='g_conv1',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)

    # 1/4
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(256, (5, 5), name='g_conv2', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(128, (5, 5), padding='same', name='g_conv2',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)

    # 1/2
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (5, 5), name='g_conv3', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(64, (5, 5), padding='same', name='g_conv3',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv3_bn')(x)

    # 1/1
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(channel, (5, 5), name='g_out', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02),  bias_initializer=Constant())(x)
    #x = Conv2D(channel, (5, 5), padding='same', activation='tanh', name='g_out',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('tanh')(x)
    model = Model(inputs=inputs, outputs=x, name='G')

    return model

def generator_containing_discremenator(dis, gen):
    model = Sequential()
    model.add(gen)
    dis.trainable = False
    model.add(dis)
    
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), 
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1], :] = img[:, :, :]
    
    return image

def train(BATCH_SIZE):
    #CIFAR10データ読み込み
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = X_train[:, :, :, None]
    #X_test = X_test[:, :, :, None]
        
    d_model = D_model(image_size, image_size, channel=3)
    g_model = G_model(image_size, image_size, channel=3)
    
    d_on_g = generator_containing_discremenator(d_model, g_model)
        
    d_opt = Adam(lr=0.0002, beta_1=0.5)
    g_opt = Adam(lr=0.0002, beta_1=0.5)
    
    g_model.compile(loss='binary_crossentropy', optimizer='SGD')
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_opt)
    d_model.trainable = True
    d_model.compile(loss='binary_crossentropy', optimizer=d_opt)
    
    for epoch in range(5000):
        print("Epoch is ", epoch)
        print("Number of batches ", int(X_train.shape[0] / BATCH_SIZE))
        
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index * BATCH_SIZE: (index+1) * BATCH_SIZE]
            generated_images = g_model.predict(noise, verbose=0)
            
            if index % 130 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                        str(epoch) + "_" + str(index) + ".png")
            
            #print(image_batch.shape)
            #print(generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d_model.train_on_batch(X, y)
            print("epoch %d batch %d d_loss : %f" % (epoch, index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d_model.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d_model.trainable = True
            print("epoch %d batch %d g_loss : %f" % (epoch, index, g_loss))
            if index % 10 == 9:
                g_model.save_weights('generator', True)
                d_model.save_weights('discremenator', True)

def generate(BATCH_SIZE, nice=False):
    g_model = G_model(image_size, image_size, channel=3)
    g_model.compile(loss='binary_crossentropy', optimizer="SGD")
    g_model.load_weights('generator')
    if nice:
        d_model = D_model(image_size, image_size, channel=3)
        d_model.compile(loss='binary_crossentropy', optimizer="SGD")
        d_model.load_weights('discremenator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g_model.predict(noise, verbose=1)
        d_pret = d_model.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3],
                               dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g_model.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    main()
