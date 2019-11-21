import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

nz = 100
nf = 65
class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(nf * 8, (4, 4), strides=(4, 4), padding='same', use_bias=False, input_shape=(nz, 1, 1))
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(nf * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv3 = tf.keras.layers.Conv2DTranspose(nf * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    
    self.conv4 = tf.keras.layers.Conv2DTranspose(nf * 1, (4, 4), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm4 = tf.keras.layers.BatchNormalization()
    
    self.conv5 = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False)

  def call(self, x, training=True):
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv3(x)
    x = self.batchnorm3(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv4(x)
    x = self.batchnorm4(x, training=training)
    x = tf.nn.relu(x)

    x = tf.nn.tanh(self.conv5(x))  
    return x
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(nf, (4, 4), strides=(2, 2), padding='same', input_shape=(nf, nf, 3))
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(nf * 2, (4, 4), strides=(2, 2), padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv3 = tf.keras.layers.Conv2D(nf * 4, (4, 4), strides=(2, 2), padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.conv4 = tf.keras.layers.Conv2D(nf * 8, (4, 4), strides=(2, 2), padding='same')
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1)

  def call(self, x, training=True):
    x = tf.nn.leaky_relu(self.conv1(x), alpha=0.25)
    x = tf.nn.leaky_relu(self.conv2(x), alpha=0.25)
    x = self.bn1(x)
    x = tf.nn.leaky_relu(self.conv3(x), alpha=0.25)
    x = self.bn2(x)
    x = tf.nn.leaky_relu(self.conv4(x), alpha=0.25)
    x = self.bn3(x)
    x = self.flatten(x)
    x = self.fc1(x)
    return x

generator = Generator()
discriminator = Discriminator()
# Defun gives 10 secs/epoch performance boost
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)

discriminator_optimizer = tf.train.AdamOptimizer(0.00008, beta1=0.5)
generator_optimizer = tf.train.AdamOptimizer(0.00008, beta1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "properpokegan-hires")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore('./training_checkpoints/properpokegan-hires')

                                
random_vector_for_generation = tf.random_normal([16, 1, 1, nz])

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)
  print(predictions.shape[0])

  fig = plt.figure(figsize=(4,4))
  
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      p = (predictions[i].numpy())
      plt.imshow(p)
      print(np.amin(p),np.amax(p))
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

generate_and_save_images(generator,
                               5,
                               random_vector_for_generation)


