# Use pre-trained models by default
use_pretrained = False

# Dimensionality of latent space
latent_dim = 2

# Mini-batch size used for training
batch_size = 64

import keras

from keras import layers
from keras.models import Model

import pdb

def print_ops(layer, indent=""):
    print(indent + "Layer:", layer.name)
    pdb.set_trace()
    print(indent + "Input(s):", [node.name for node in layer._inbound_nodes])
    
    for sub_layer in layer.layers:
        print_ops(sub_layer, indent + "  ")

def create_predictor():
    '''
    Creates a classifier that predicts digit image labels 
    from latent variables.
    '''
    predictor_input = layers.Input(shape=(latent_dim,), name='t_mean')
    
    x = layers.Dense(128, activation='relu')(predictor_input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax', name='label_probs')(x)

    return Model(predictor_input, x, name='predictor')

from variational_autoencoder_opt_util import *

encoder = create_encoder(latent_dim)
decoder = create_decoder(latent_dim)
sampler = create_sampler()
predictor = create_predictor()

x = layers.Input(shape=image_shape, name='image')
t_mean, t_log_var = encoder(x)
t = sampler([t_mean, t_log_var])
t_decoded = decoder(t)
t_predicted = predictor(t_mean)

model = Model(x, [t_decoded, t_predicted], name='composite')

from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
y_train_cat = to_categorical(y_train)

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))
y_test_cat = to_categorical(y_test)  

from keras import backend as K
from keras.models import load_model

def vae_loss(x, t_decoded):
    '''
    Negative variational lower bound used as loss function for 
    training the variational autoencoder on the MNIST dataset.
    '''
    # Reconstruction loss
    rc_loss = K.sum(K.binary_crossentropy(
        K.batch_flatten(x), 
        K.batch_flatten(t_decoded)), axis=-1)

    # Regularization term (KL divergence)
    kl_loss = -0.5 * K.sum(1 + t_log_var \
                             - K.square(t_mean) \
                             - K.exp(t_log_var), axis=-1)
    
    return K.mean(rc_loss + kl_loss)

if use_pretrained:
    # Load VAE that was jointly trained with a 
    # predictor returned from create_predictor()
    model = load_model('models/vae-opt/vae-predictor-softmax.h5', 
                        custom_objects={'vae_loss': vae_loss})
else:
    model.compile(optimizer='rmsprop', 
                  loss=[vae_loss, 'categorical_crossentropy'],
                  loss_weights=[1.0, 20.0], run_eagerly=True)
    

    # print_ops(model)
    opt = model.predict(x_train[0:1])
    pdb.set_trace()
    vae_loss(x_train[0:1], opt[0])

    model.fit(x_train, [x_train, y_train_cat], epochs=1, batch_size=batch_size)
    # model.fit(x_train, [x_train, y_train_cat],
    #           shuffle=True,
    #           epochs=10,
    #           batch_size=batch_size,
    #           validation_data=(x_test, [x_test, y_test_cat]))
    
"""
KerasTensor(type_spec=TensorSpec(shape=(), dtype=tf.float32, name=None), 
name='Placeholder:0', description="created by layer 'tf.cast_1'")
"""