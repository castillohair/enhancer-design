import numpy
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

def resblock(x, filters, kernel_size, dilation_rate=1, first_conv_activation='relu'):
    """
    Constructs a residual block with two convolutional layers and a skip connection.

    Parameters:
    -----------
    x : tf.Tensor
        Input tensor to the residual block.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int
        Size of the convolutional kernels.
    dilation_rate : int, optional
        Dilation rate for the convolutional layers. Default is 1.
    first_conv_activation : str, optional
        Activation function to use after the first convolution. Default is 'relu'.

    Returns:
    --------
    tf.Tensor
        Output tensor after applying the residual block.

    """
    conv_x = x

    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation(first_conv_activation)(conv_x)
    conv_x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        padding='same',
        activation='linear',
        dilation_rate=dilation_rate,
        kernel_initializer='glorot_normal',
    )(conv_x) 

    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.ReLU()(conv_x)
    conv_x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        padding='same',
        activation='linear',
        dilation_rate=dilation_rate,
        kernel_initializer='glorot_normal',
    )(conv_x)

    x = layers.add([conv_x, x])

    return x

def resgroup(
        x,
        n_blocks_per_group,
        filters,
        kernel_size,
        dilation_rate=1,
        first_conv_activation='relu',
    ):
    """
    Constructs a group of residual blocks.
    
    Parameters:
    -----------
    x : tf.Tensor
        Input tensor to the residual group.
    n_blocks_per_group : int
        Number of residual blocks in the group.
    filters : int
        Number of filters for the convolutional layers in each block.
    kernel_size : int
        Size of the convolutional kernels in each block.
    dilation_rate : int, optional
        Dilation rate for the convolutional layers in each block. Default is 1.
    first_conv_activation : str, optional
        Activation function to use after the first convolution in each block. Default is 'relu'.

    Returns:
    --------
    tf.Tensor
        Output tensor after applying the residual group.

    """
    for i in range(n_blocks_per_group):
        x = resblock(x, filters, kernel_size, dilation_rate, first_conv_activation)

    return x

def make_model(
        input_seq_length,
        groups=4,
        blocks_per_group=3,
        filters=128,
        kernel_size=13,
        dilation_rates=[1, 2, 4, 8],
        first_conv_activation='relu',
        n_outputs=8,
        output_activation='linear',
    ):
    """
    Constructs a residual neural network model with a DNA sequence input.

    Parameters:
    -----------
    input_seq_length : int
        Maximum length of the input DNA sequence.
    groups : int, optional
        Number of residual groups in the model.
    blocks_per_group : int, optional
        Number of residual blocks in each group.
    filters : int, optional
        Number of filters for the convolutional layers.
    kernel_size : int, optional
        Size of the convolutional kernels.
    dilation_rates : list of int, optional
        Dilation rates for each residual group.
    first_conv_activation : str, optional
        Activation function to use after the first convolution in each block.
    n_outputs : int, optional
        Number of output units in each final dense layer.
    output_activation : str or list of str, optional
        Activation function(s) for the final dense layer(s). Use 'linear' for regression
        and 'sigmoid' for binary classification.

    Returns:
    --------
    tf.keras.Model
        Residual neural network as a Keras model.

    """

    # Input
    model_input = layers.Input(shape=(input_seq_length, 4))

    skip_convs = []

    # 1x1 convolution w/ linear activation to get to "filters" channels
    # (output dim: batch_size x seq_len x filters)
    x = layers.Conv1D(
        filters,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_initializer='glorot_normal',
    )(model_input)
    y = layers.Conv1D(
        filters,
        kernel_size=1,
        padding='same',
    )(x)
    skip_convs.append(y)

    # First actual residual group: activation is different
    # (output dim: batch_size x seq_len x n_filters) <- maybe don't do this?
    x = resgroup(
        x,
        blocks_per_group,
        filters,
        kernel_size,
        dilation_rate=dilation_rates[0],
        first_conv_activation=first_conv_activation,
    )
    y = layers.Conv1D(
        filters,
        kernel_size=1,
        padding='same',
        kernel_initializer='glorot_normal',
    )(x)
    skip_convs.append(y)

    # Remaining residual groups
    # (output dim: batch_size x seq_len x n_filters)
    for i in range(1, groups):
        x = resgroup(
            x,
            blocks_per_group,
            filters,
            kernel_size,
            dilation_rate=dilation_rates[i],
            first_conv_activation='relu',
        )
        y = layers.Conv1D(
            filters,
            kernel_size=1,
            padding='same',
            kernel_initializer='glorot_normal',
        )(x)
        skip_convs.append(y)

    # Final convolutional layer, summed with all the skip connections from previous resgroups
    # (output dim: batch_size x seq_len x n_filters)
    x = layers.Conv1D(
        filters,
        kernel_size=1,
        padding='same',
        kernel_initializer='glorot_normal',
    )(x)

    for i in range(len(skip_convs)):
        x = layers.add([x, skip_convs[i]])

    # x is still (batch_size x seq_len x n_filters)
    # Average across sequence dimension
    # output is (batch_size x n_filters)
    x = layers.GlobalAvgPool1D()(x)

    # Final layers
    if type(output_activation) is not list:
        output_activations = [output_activation]
    else:
        output_activations = output_activation

    model_outputs = []
    for act in output_activations:
        output_layer = layers.Dense(
            n_outputs,
            name=f'dense_output_{act}',
            activation=act,
        )
        model_outputs.append(output_layer(x))

    if type(output_activation) is list:
        model_output = model_outputs
    else:
        model_output = model_outputs[0]

    model = models.Model(
        model_input,
        model_output,
    )

    return model

def load_model(model_path, output_transformation_fpath=None):
    """
    Load a pre-trained model from the specified path.

    If output_transformation_fpath is provided, it will be used to apply a linear
    transformation to the model outputs. Note that this assumes the model has a single output.

    Parameters:
    -----------
    model_path : str
        Path to the saved Keras model.
    output_transformation_fpath : str, optional
        Path to a .npy file containing the output transformation matrix.

    Returns:
    --------
    tf.keras.Model
        Loaded Keras model, possibly with transformed outputs.
        
    """
    # Load the model from the specified path
    model = tensorflow.keras.models.load_model(model_path)

    if output_transformation_fpath is None:
        # If no output transformation file is provided, return the model as is
        return model

    else:
    
        # create dummy input layer
        model_input = layers.Input(shape=(None, 4))

        # Get model output
        model_output = model(model_input)

        # Load model output mapping matrix
        output_transformation_mat = numpy.load(output_transformation_fpath)

        # Create a tf layer where the input is multiplied by a constant matrix
        output_transformation_layer = layers.Lambda(lambda x: tensorflow.matmul(x, output_transformation_mat), name='output_transformation_layer')
        model_transformed_output = output_transformation_layer(model_output)

        # Create a new model with the transformed output
        model_transformed = models.Model(
            model_input,
            model_transformed_output,
        )
        
        return model_transformed
