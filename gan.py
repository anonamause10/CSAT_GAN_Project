import os
import numpy as np
from collections import Counter
#yeet
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import keras

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=3):
        epoch = open("currepoch.txt","r+")
        self.currepoch = float(epoch.read())
        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """
        if(os.path.exists('pokegan.h5')):
            model = keras.models.load_model('pokegan.h5')
            model.name = "generator"
        else:
            model = Sequential()
            model.add(Dense(256, input_shape=(100,)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(self.width  * self.height * self.channels,  activation='tanh'))
            model.add(Reshape((self.width, self.height, self.channels)))
        print('generator')
        model.summary()
        return model

    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.int64((self.width * self.height * self.channels)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X, epochs=15000, batch = 32, save_interval = 100):
        
        train_datagen = ImageDataGenerator(rescale=1./255,)
        itr = train_generator = train_datagen.flow_from_directory('pokemon-images-dataset',
        target_size=(self.width,self.height),batch_size=batch, class_mode='sparse',color_mode='rgb')
        for cnt in range(epochs+1):
            X_train ,y= itr.next()
            #X_train = X
            ## train discriminator
            if(len(X_train)<batch):
                train_datagen = ImageDataGenerator(rescale=1./255,)
                itr = train_generator = train_datagen.flow_from_directory('pokemon-images-dataset',
                target_size=(self.width,self.height),batch_size=batch, class_mode='sparse',color_mode='rgb')
                X_train,y= itr.next()
            np.random.shuffle(X_train)
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            model = self.stacked_generator_discriminator
            g_loss = model.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (self.currepoch+cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=self.currepoch+cnt)
        epoch = open("currepoch.txt","r+")
        s = str(self.currepoch+cnt)
        print(type(s))
        epoch.write("\f"+s)
        self.G.save('pokegan.h5')

        


    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./newpokemans"):
            os.makedirs("./newpokemans")
        filename = "./newpokemans/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width,self.channels])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    
    (X_train, _), (_, _) = fashion_mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)


    gan = GAN()
    gan.train(X_train)