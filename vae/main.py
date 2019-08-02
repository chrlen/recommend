import tensorflow as tf
import vae.functions as fct

class VAE(tf.keras.Model):
    def __init__(self,
                 input_shape=(412, 297, 3),
                 intermediate_dim=512,
                 latent_dim=8,
                 output_shape=(412, 297, 3),
                 **kwargs):
        super(VAE, self).__init__()

        flattend_size = 0


        self.inputs = tf.keras.layers.Dense(units=3, input_shape=input_shape, name='encoder_input')
        self.flatten = tf.keras.layers.Reshape(target_shape=(-1,), input_shape=input_shape)#(self.inputs)
        self.intermediate = tf.keras.layers.Dense(intermediate_dim, activation='relu')
        self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')
        self.sampler = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')#([self.z_mean, self.z_log_var])
        #self.decoder_inputs = tf.keras.layers.Dense(units= 3, input_shape=(latent_dim,), name='z_sampling')
        self.decoder_input= tf.keras.layers.Dense(latent_dim, activation='relu')
        self.decoder_intermediate = tf.keras.layers.Dense(intermediate_dim,  activation='relu')#(self.decoder_inputs)
        self.decoder_output = tf.keras.layers.Dense(367092, activation='relu')#(self.decoder_intermediate)
        self.decoder_reshape = tf.keras.layers.Reshape(output_shape)

    @tf.function
    def encode(self, inputs):
        print(type(inputs))
        x = self.inputs(inputs)
        f = self.flatten(x)
        y = self.intermediate(f)
        return y

    @tf.function
    def sample(self, inputs):
        z_mean = self.z_mean(inputs)
        z_log_var = self.z_log_var(inputs)
        return self.sampler([z_mean, z_log_var])

    @tf.function
    def decode(self, inputs):
        x = self.decoder_input(inputs)
        y = self.decoder_intermediate(inputs)
        z = self.decoder_output(y)
        return self.decoder_reshape(z)

    def call(self, inputs, training=False):
        a = self.encode(inputs)
        latent = self.sample(a)
        #return latent
        return self.decode(latent)

    @tf.function
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = tf.keras.backend.shape(z_mean)[0]
        dim = tf.keras.backend.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), dtype=tf.double)
        exponent = tf.keras.backend.exp(z_log_var)* epsilon
        return z_mean + exponent
    """"""
