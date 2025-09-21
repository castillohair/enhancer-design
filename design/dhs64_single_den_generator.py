
import tensorflow
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as Kpy

import genesis
import genesis.generator

LATENT_SIZE = 100
SEQ_LENGTH = 145

def make_generator(
        batch_size,
        sequence_class,
        n_classes=1,
        seq_length=SEQ_LENGTH,
        supply_inputs=False
    ):
    """
    Constructs the generator model.

    The generator is a deconvolutional/convolutional neural network that takes a latent
    vector as input and produces a one hot encoded sequence as output. To ensure
    compatibility with DENs, the generator model constructed here takes two latent
    vectors as input and produces two sequences as output. Additionally, this function's
    signature matches that expected by DENs, even though some inputs are unused.

    The only possible sequence length with this architecture is 145bp. Model dimensions
    are hardcoded to ensure that the output sequence length is correct. If a different
    sequence length is desired, the model architecture must be modified.
    
    Parameters
    ----------
    batch_size : int
        The batch size to be used during training.
    sequence_class : int
        The class of sequences to be generated. This parameter is unused, but is included
        for compatibility with DENs.
    n_classes : int
        The number of sequence classes. This parameter is unused, but is included for
        compatibility with DENs.
    seq_length : int
        The length of sequences to be generated. The only supported value is 145. This
        parameter is included for compatibility with DENs.
    supply_inputs : bool
        Whether to supply inputs to the model. This parameter is unused, but is included
        for compatibility with DENs.

    Returns
    -------
    generator_inputs : list of keras.layers.Input
        The inputs to the generator model.
    generator_outputs : list of keras.layers.Layer
        The outputs of the generator model.
    generator_weights_initialization : list
        An empty list, included for compatibility with DENs.

    """

    # Check that the specified sequence length is correct
    if seq_length != SEQ_LENGTH:
        raise ValueError('The specified generator sequence length is not supported.')
    
    # Generator inputs
    latent_input_1 = Input(
        tensor=K.ones((batch_size, LATENT_SIZE)), name='noise_input_1')
    latent_input_2 = Input(
        tensor=K.ones((batch_size, LATENT_SIZE)), name='noise_input_2')

    latent_input_1_out = Lambda(
        lambda inp: inp * K.random_uniform(
            (batch_size, LATENT_SIZE), minval=-1.0, maxval=1.0),
        name='lambda_rand_input_1')(latent_input_1)
    latent_input_2_out = Lambda(
        lambda inp: inp * K.random_uniform(
            (batch_size, LATENT_SIZE), minval=-1.0, maxval=1.0),
        name='lambda_rand_input_2')(latent_input_2)
    
    seed_input_1 = latent_input_1_out
    seed_input_2 = latent_input_2_out
    
    # Generator network definition
    policy_dense_1 = Dense(
        31 * 384,
        activation='relu',
        kernel_initializer='glorot_uniform',
        name='policy_dense_1')
    
    policy_dense_1_reshape = Reshape((31, 1, 384))
    
    policy_deconv_0 = Conv2DTranspose(
        256,
        (8, 1),
        strides=(2, 1),
        padding='valid',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_deconv_0')
    
    policy_deconv_1 = Conv2DTranspose(
        192,
        (8, 1),
        strides=(2, 1),
        padding='valid',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_deconv_1')
    
    policy_deconv_2 = Conv2DTranspose(
        128,
        (7, 1),
        strides=(2, 1),
        padding='valid',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_deconv_2')
    
    policy_conv_3 = Conv2D(
        128,
        (8, 1),
        strides=(2, 1),
        padding='same',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_conv_3')

    policy_conv_4 = Conv2D(
        64,
        (8, 1),
        strides=(1, 1),
        padding='same',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_conv_4')

    policy_conv_5 = Conv2D(
        4,
        (8, 1),
        strides=(1, 1),
        padding='same',
        activation='linear',
        kernel_initializer='glorot_normal',
        name='policy_conv_5')
    
    batch_norm_0 = BatchNormalization(name='policy_batch_norm_0')
    relu_0 = Lambda(lambda x: K.relu(x))

    batch_norm_1 = BatchNormalization(name='policy_batch_norm_1')
    relu_1 = Lambda(lambda x: K.relu(x))

    batch_norm_2 = BatchNormalization(name='policy_batch_norm_2')
    relu_2 = Lambda(lambda x: K.relu(x))

    batch_norm_3 = BatchNormalization(name='policy_batch_norm_3')
    relu_3 = Lambda(lambda x: K.relu(x))

    batch_norm_4 = BatchNormalization(name='policy_batch_norm_4')
    relu_4 = Lambda(lambda x: K.relu(x))

    policy_out_1 = Reshape((seq_length, 4, 1))(policy_conv_5(
        relu_4(batch_norm_4(policy_conv_4(
            relu_3(batch_norm_3(policy_conv_3(
                relu_2(batch_norm_2(policy_deconv_2(
                    relu_1(batch_norm_1(policy_deconv_1(
                        relu_0(batch_norm_0(policy_deconv_0(
                            policy_dense_1_reshape(policy_dense_1(seed_input_1))
                            )))
                        )))
                    )))
                )))
            )))
        ))
    policy_out_2 = Reshape((seq_length, 4, 1))(policy_conv_5(
        relu_4(batch_norm_4(policy_conv_4(
            relu_3(batch_norm_3(policy_conv_3(
                relu_2(batch_norm_2(policy_deconv_2(
                    relu_1(batch_norm_1(policy_deconv_1(
                        relu_0(batch_norm_0(policy_deconv_0(
                            policy_dense_1_reshape(policy_dense_1(seed_input_2))
                            )))
                        )))
                    )))
                )))
            )))
        ))
    
    return [latent_input_1, latent_input_2], [policy_out_1, policy_out_2], []


def load_generator(model_path):
    """
    Load a generator model from file.

    Parameters
    ----------
    model_path : str
        The path to the saved generator model.

    Returns
    -------
    model : keras.Model
        The loaded generator model.

    """
    model = tensorflow.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "K": K,
            'Kpy': Kpy,
            'mask_pwm': genesis.generator.mask_pwm,
            'st_sampled_softmax': genesis.generator.st_sampled_softmax,
        }
    )

    return model
