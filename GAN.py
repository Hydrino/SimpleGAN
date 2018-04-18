# here I try to implement a simple GAN
# both the networks would be vanilla neural nets(Dense)
# training data is 28x28 doodle images of apple (grayscale)

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Input, BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.regularizers import L1L2
import matplotlib.pyplot as plt


class GAN:
    def __init__(self):
        # shape of input image
        self.image_shape = (28, 28, 1)
        self.noise_shape = (20,)
        self.optimiser = Adam(0.0002, 0.5)

        # now we create both our networks

        self.generator = self.build_generator()
        self.generator.compile(optimizer=self.optimiser, loss='binary_crossentropy')

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=self.optimiser, loss='binary_crossentropy', metrics=['accuracy'])

        noise = Input(self.noise_shape)
        img = self.generator(noise)
        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(noise, validity)
        self.combined.compile(optimizer=self.optimiser, loss='binary_crossentropy')

    def build_generator(self):
        # the generator will be a fully connected network
        # with input noise vector of shape (20,)
        # the output of the generator will be a 28x28 image

        model = Sequential()
        model.add(Dense(256, input_shape=self.noise_shape))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
       # model.add(Dropout(0.50))

        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Dropout(0.50))

        model.add(Dense(784, activation='tanh'))
        model.add(Reshape(self.image_shape))
        print('Generator model : ')
        print(model.summary())

        noise = Input(self.noise_shape)
        gen_img = model(noise)

        return Model(noise, gen_img)

    def build_discriminator(self):
        # this is a simple FCN
        # input for the discriminator is a 28x28 image

        model = Sequential()
        model.add(Flatten(input_shape=self.image_shape))

        model.add(Dense(512))
        model.add(LeakyReLU())
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        model.add(Dense(256))
        model.add(LeakyReLU())
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.25))

        model.add(Dense(1, activation='sigmoid'))
        print('Discriminator model : ')
        print(model.summary())

        img = Input(self.image_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X, epochs=30000, batch_size=64):
        X_train = (X.astype(np.float32)-127.5) / 127.5
        X_train = X_train.reshape((X.shape[0], 28, 28, 1))

        for epoch in range(epochs + 1):
            # -----------train discriminator ------------

            idx = np.random.randint(0, X.shape[0], batch_size)
            real_imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 20))
            fakes = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fakes, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------train generator -------------------

            noise = np.random.normal(0, 1, (batch_size, 20))
            valid_y = [1] * batch_size

            g_loss = self.combined.train_on_batch(noise, valid_y)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % 1000 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 20))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    apples = np.load("apple.npy")
    gan.train(X=apples)
