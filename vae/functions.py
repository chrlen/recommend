import tensorflow as tf

def get_encoder(input_shape, output_shape):
    def encoder(input):
        inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
        x = tf.keras.layers.Dense(output_shape, activation='relu')(inputs)
        return x

    return encoder

def get_var(n_means,input_shape,output_shape):
    def var(input):
        return input

    return var

def get_decoder(
        decoder_input_shape,
        decoder_output_shape
):
    def decoder(input):
        x = tf.keras.layers.Dense(decoder_input_shape, activation='relu')(input)
        outputs = tf.keras.layers.Dense(decoder_output_shape, activation='sigmoid')(x)
        return input

    return decoder

def generate_functions(
        encoder_input_shape,
        encoder_output_shape,
        n_means,
        decoder_input_shape,
        decoder_output_shaope,
        **kwargs
):
    return [
        get_encoder(
            encoder_input_shape,
            encoder_output_shape,
            **kwargs
        ),
        get_var(n_means, **kwargs),
        get_decoder(
            decoder_input_shape,
            decoder_output_shape,
            **kwargs
        )
    ]
