'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import math

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length

debug = False

class ExpandDim(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0:-1] + (1,) + input_shape[-1:])

    def get_config(self):
        config = {}
        base_config = super(ExpandDim, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class RemoveDim(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.squeeze(inputs, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0:-2] + input_shape[-1:])

    def get_config(self):
        config = {}
        base_config = super(RemoveDim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Length(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(Length, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        assert inputs.get_shape()[-2] == self.num_classes, \
            'Error: Must have num_capsules = num_classes going into Length else have dimensions (batch size, atoms)'

        output = tf.norm(tensor=inputs, axis=-1)

        if debug:
            output = tf.compat.v1.Print(output, [output], message='\nLength: ', summarize=99999999)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(Length, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mask(layers.Layer):
    def __init__(self, n_classes=5, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.n_classes = n_classes

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if self.n_classes == 1:
            return inputs
        else:
            return K.batch_flatten(inputs * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            if len(input_shape[0]) == 3:
                return tuple([None, input_shape[0][1] * input_shape[0][2]])
            else:
                return input_shape[0][0:-2] + input_shape[0][-1:]
        else:  # no true label provided
            if len(input_shape) == 3:
                return tuple([None, input_shape[1] * input_shape[2]])
            else:
                return input_shape[0:-2] + input_shape[-1:]

    def get_config(self):
        config = {'n_classes': self.n_classes}
        base_config = super(Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FullCapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, num_atoms, activation='softmax', routings=3, leaky_routing=False,
                 kernel_initializer='he_normal', **kwargs):
        super(FullCapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.activation = activation
        self.activation = activation
        self.routings = routings
        self.leaky_routing = leaky_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_height * self.input_width * self.input_num_capsule,
                                        self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_shape = K.shape(input_tensor)

        input_tensor_reshaped = K.reshape(input_tensor, [
            input_shape[0], self.input_num_capsule * self.input_height * self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_num_capsule * self.input_height * self.input_width,
                                         self.input_num_atoms))

        input_tiled = tf.tile(K.expand_dims(input_tensor_reshaped, -1), [1, 1, 1, self.num_capsule * self.num_atoms])

        votes = tf.reduce_sum(input_tensor=input_tiled * self.W, axis=2)
        votes_reshaped = tf.reshape(votes, [-1, self.input_num_capsule * self.input_height * self.input_width,
                                            self.num_capsule, self.num_atoms])

        input_shape = K.shape(input_tensor)
        logit_shape = tf.stack([input_shape[0], self.input_num_capsule * self.input_height * self.input_width,
                                self.num_capsule])

        activations = _update_routing(
            votes=votes_reshaped,
            biases=self.b,
            logit_shape=logit_shape,
            num_dims=4,
            route_activ=self.activation,
            input_dim=self.input_num_capsule * self.input_height * self.input_width,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        if debug:
            activations = tf.compat.v1.Print(activations, [activations], message='FullCon Caps Activations: ', summarize=99999999)

        return activations

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.num_atoms])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'routings': self.routings,
            'leaky_routing': self.leaky_routing,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(FullCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', activation='softmax', routings=3,
                 kernel_initializer='he_normal', epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                 self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_shape = K.shape(input_tensor)
        _, in_height, in_width, _, _ = input_tensor.get_shape()

        input_transposed = tf.transpose(a=input_tensor, perm=[0, 3, 1, 2, 4])
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[3], input_shape[1], input_shape[2], self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, in_height, in_width, self.input_num_atoms))

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(conv)
        _, conv_height, conv_width, _ = conv.get_shape()
        # Reshape back to 6D by splitting first dimmension to batch and input_dim
        # and splitting last dimmension to output_dim and output_atoms.

        votes = K.reshape(conv, [input_shape[0], input_shape[3], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[0], input_shape[3], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = _update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            route_activ=self.activation,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _update_routing(votes, biases, logit_shape, num_dims, route_activ, input_dim, output_dim,
                    num_routing):
    """Sums over scaled votes and applies squash to compute the activations.

    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.

    Args:
    votes: tensor, The transformed outputs of the layer below.
    biases: tensor, Bias variable.
    logit_shape: tensor, shape of the logit to be initialized.
    num_dims: scalar, number of dimmensions in votes. For fully connected
      capsule it is 4, for convolutional 6.
    input_dim: scalar, number of capsules in the input layer.
    output_dim: scalar, number of capsules in the output layer.
    num_routing: scalar, Number of routing iterations.
    leaky: boolean, if set use leaky routing.

    Returns:
    The activation tensor of the output layer after num_routing iterations.
    """

    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 7:
        votes_t_shape = [6, 0, 1, 2, 3, 4, 5]
        r_t_shape = [1, 2, 3, 4, 5, 6, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(a=votes, perm=votes_t_shape)

    def _body(i, logits, activations):
        """Routing while loop."""
        if route_activ == 'softmax':
            route = tf.nn.softmax(logits, axis=-1)
        elif route_activ == 'sigmoid':
            route = tf.nn.sigmoid(logits)
        else:
            raise NotImplementedError('Must choose from sigmoid or softmax for capsule activation.')
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(a=preactivate_unrolled, perm=r_t_shape)
        preactivate = tf.reduce_sum(input_tensor=preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(input_tensor=votes * act_replicated, axis=-1)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
    if route_activ == 'softmax':
        logits = tf.fill(logit_shape, 0.0)
    elif route_activ == 'sigmoid':
        logits = tf.fill(logit_shape, 1.0)
    else:
        raise NotImplementedError('Must choose from sigmoid or softmax for capsule activation.')

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
      cond=lambda i, logits, activations: i < num_routing, body=_body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

    return K.cast(activations.read(num_routing - 1), dtype='float32')


def _squash(input_tensor):
    norm = tf.norm(tensor=input_tensor, axis=-1, keepdims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image
