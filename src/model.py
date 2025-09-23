import numpy
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

from . import definitions


def resblock(
        x,
        filters,
        kernel_size,
        dilation_rate=1,
        first_conv_activation='relu',
    ):
    """
    Constructs a residual block with two convolutional layers and a skip connection.

    Parameters
    ----------
    x : tensorflow.Tensor
        Input tensor to the residual block.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int
        Size of the convolutional kernels.
    dilation_rate : int, optional
        Dilation rate for the convolutional layers. Default is 1.
    first_conv_activation : str, optional
        Activation function to use after the first convolution. Default is 'relu'.

    Returns
    -------
    tensorflow.Tensor
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
    
    Parameters
    ----------
    x : tensorflow.Tensor
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

    Returns
    -------
    tensorflow.Tensor
        Output tensor after applying the residual group.

    """
    for i in range(n_blocks_per_group):
        x = resblock(x, filters, kernel_size, dilation_rate, first_conv_activation)

    return x

def make_resnet(
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

    Parameters
    ----------
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

    Returns
    -------
    tensorflow.keras.Model
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

def load_model(model_path):
    """
    Load a pre-trained model from the specified path.
    
    Parameters
    ----------
    model_path : str
        Path to the saved Keras model.
        
    Returns
    -------
    tensorflow.keras.Model
        Loaded Keras model.
        
    """
    model = tensorflow.keras.models.load_model(model_path)
    return model

def select_output_head(model, output_head_idx):
    """
    Select a specific output head from a multi-output Keras model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Keras model with multiple output heads.
    output_head_idx : int
        Index of the output head to select.

    Returns
    -------
    tensorflow.keras.Model
        New Keras model with the selected output head.

    """

    # create dummy input layer
    model_input = layers.Input(shape=(None, 4))

    # Get the specified model output
    model_output = model(model_input)[output_head_idx]

    # Create a new model with the selected output
    model_to_return = models.Model(
        model_input,
        model_output,
    )
    
    return model_to_return

def apply_output_transformation(model, output_transmat_filepath):
    """
    Apply an output transformation matrix to a Keras model's outputs.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Keras model whose outputs will be transformed.
    output_transmat_filepath : str
        Path to a .npy file containing the output transformation matrix.
    
    Returns
    -------
    tensorflow.keras.Model
        New Keras model with transformed outputs.

    """

    # create dummy input layer
    model_input = layers.Input(shape=(None, 4))

    # Get original model output
    model_output = model(model_input)

    # Load the output transformation matrix
    output_transformation_mat = numpy.load(output_transmat_filepath)

    # Create a tf layer where the input is multiplied by a constant matrix
    output_transformation_layer = layers.Lambda(lambda x: tensorflow.matmul(x, output_transformation_mat))
    model_output = output_transformation_layer(model_output)

    # Create a new model with the transformed output
    model_to_return = models.Model(
        model_input,
        model_output,
    )
    
    return model_to_return

def make_model_ensemble(
        models_list,
        max_output_idx=None,
        min_output_idx=None,
        avg_output_idx=None,
        padded_input_length=None,
    ):
    """
    Create an ensemble model from a list of Keras models.

    The default behavior is to average the outputs of the models. Any output indices
    that are not specified for max or min will be averaged.

    Optionally, change the length of the input sequence by padding with zeros.

    Parameters
    ----------
    models_list : list of tensorflow.keras.Model or list of str
        List of Keras models or paths to saved Keras models to include in the ensemble.
    max_output_idx : int, optional
        Index of the output to take the maximum across models. Default is None.
    min_output_idx : int, optional
        Index of the output to take the minimum across models. Default is None.
    avg_output_idx : int, optional
        Index of the output to take the average across models. Default is None.
    input_seq_length : int, optional
        Length of the input sequence. If specified, input sequences will be padded
        with zeros to this length. Default is None, which uses the models' original
        input length.

    Returns
    -------
    tensorflow.keras.Model
        Ensemble Keras model.

    Raises
    ------
    ValueError
        If any of the specified output indices are out of range for the models' outputs.
        If the models have different output shapes.

    """
    
    # If models is a list of file paths, load each model
    if type(models_list[0]) is str:
        model_list = [load_model(m) for m in models_list]

    # Check that all models have the same output shape
    output_shapes = [m.output_shape for m in models_list]
    if len(set(output_shapes)) != 1:
        raise ValueError("All models must have the same output shape.")
    n_outputs = output_shapes[0][-1]

    # Figure out defaults for max, min, avg output indices
    if avg_output_idx is None:
        avg_output_idx = list(range(n_outputs))
    if max_output_idx is None:
        max_output_idx = []
    else:
        avg_output_idx = [i for i in avg_output_idx if i not in max_output_idx]
    if min_output_idx is None:
        min_output_idx = []
    else:
        avg_output_idx = [i for i in avg_output_idx if i not in min_output_idx]

    # Construct padded input if necessary
    if padded_input_length is None:
        model_input = layers.Input(shape=(None, 4))
    else:
        input_padding_layer = layers.Lambda(
            lambda x: tensorflow.pad(
                x,
                [[0, 0], [0, definitions.MODEL_INPUT_LENGTH - padded_input_length], [0, 0]],
                "CONSTANT",
                constant_values=0,
            ),
        )

        model_input = layers.Input(shape=(padded_input_length, 4))

    # Outputs of individual models
    models_individual_output = [m(input_padding_layer(model_input)) for m in models_list]
    
    # Select outputs
    mask_min = numpy.zeros((1, n_outputs))
    mask_min[:, min_output_idx] = 1
    mask_min = tensorflow.cast(mask_min, tensorflow.float32)
    mask_max = numpy.zeros((1, n_outputs))
    mask_max[:, max_output_idx] = 1
    mask_max = tensorflow.cast(mask_max, tensorflow.float32)
    mask_avg = numpy.zeros((1, n_outputs))
    mask_avg[:, avg_output_idx] = 1
    mask_avg = tensorflow.cast(mask_avg, tensorflow.float32)
    select_output_layer = layers.Lambda(
        lambda x: tensorflow.reduce_min(tensorflow.stack(x, axis=-1), axis=-1)*mask_min + \
            tensorflow.reduce_max(tensorflow.stack(x, axis=-1), axis=-1)*mask_max + \
            tensorflow.reduce_mean(tensorflow.stack(x, axis=-1), axis=-1)*mask_avg,
    )
    model_ensemble_output = select_output_layer(models_individual_output)

    model_ensemble = models.Model(
        model_input,
        model_ensemble_output,
    )

    return model_ensemble

class MSE_Cosine_Loss(tensorflow.keras.losses.Loss):
    """
    Combines mean squared error (MSE) and cosine similarity losses.
    
    Parameters
    ----------
    w_mse, w_cosine : float
        Weights for MSE and cosine similarity losses, respectively.

    """
    def __init__(self, w_mse, w_cosine):
        super().__init__()
        self.w_mse = w_mse
        self.w_cosine = w_cosine

    def call(self, y_true, y_pred):
        # MSE loss
        mse_loss = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
        # Cosine loss
        ym_true = tensorflow.math.reduce_mean(y_true, axis=1, keepdims=True)
        ym_pred = tensorflow.math.reduce_mean(y_pred, axis=1, keepdims=True)
        cosine_sim = tensorflow.keras.losses.cosine_similarity(y_true - ym_true, y_pred - ym_pred, axis=1)
        # cosine_loss = 1 - cosine_sim
        # Combine
        return self.w_mse*mse_loss + self.w_cosine*cosine_sim
    
class SequentialLearningScheduler(tensorflow.keras.callbacks.Callback):
    """
    Learning rate scheduler that reloads the best model after every plateau.

    Parameters
    ----------
    learning_rates : list of float
        List of learning rates to use sequentially.
    patience : int or list of int
        Number of epochs with no improvement to wait before reloading the best model
        and switching to the next learning rate. If an int is provided, the same patience
        will be used for all learning rates. If a list is provided, it should have the
        same length as learning_rates.

    """
    def __init__(self, learning_rates, patience=10):
        super(SequentialLearningScheduler, self).__init__()
        self.learning_rates = learning_rates
        if type(patience) == int:
            self.patience = [patience]*len(learning_rates)
        else:
            self.patience = patience
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0
        self.lr_index = 0  # Start with the first learning rate in the list

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience[self.lr_index]:
                if self.lr_index < len(self.learning_rates) - 1:
                    self.model.set_weights(self.best_weights)
                    self.lr_index += 1
                    new_lr = self.learning_rates[self.lr_index]
                    tensorflow.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    print(f'\nEpoch {epoch+1}: Plateau reached, reloading best model and setting learning rate to {new_lr}.')
                    self.wait = 0
                else:
                    print("\nReached the end of the learning rates list. Stopping training.")
                    self.model.stop_training = True
