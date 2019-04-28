from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense, Conv2DTranspose, ELU, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16
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

#Discreminatorモデル（VGG16をFineTuning）
def D_model(Height, Width, channel=3):
    #Inputサイズを指定
    input_tensor = Input(shape=(image_size, image_size, 3))

    #VGG16の読み込み（全結合層なし、ImageNetで学習した重み使用、Inputサイズ指定)
    base_model = VGG16(
            include_top = False,
            weights = "imagenet",
            input_tensor=input_tensor
            )
    
    #VGG16の図の緑色の部分（FC層）の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    
    #VGG16とFC層を結合してモデルを作成
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output), name='D')
    return model

#Generatorモデル
def G_model(Height, Width, channel=3):
    noise_shape = (100,)
    
    model = Sequential()
    model.add(Dense(4 * 4 * 1024, input_shape=noise_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Reshape((4, 4, 1024)))
    
    model.add(Conv2DTranspose(512, (5, 5), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Conv2DTranspose(256, (5, 5), padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Conv2DTranspose(3, (5, 5), padding='same', strides=(2, 2), activation='tanh'))
    
    model = Model(model.input, model.output, name='G')
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
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]), 
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = img[:, :, 0]
    
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
            
#            print(image_batch.shape)
#            print(generated_images.shape)
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
