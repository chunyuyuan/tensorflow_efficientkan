import tensorflow as tf
import numpy as np
import math

class KANLinear(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=tf.nn.silu,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            tf.range(-spline_order, grid_size + spline_order + 1, dtype=tf.float32) * h
            + grid_range[0]
        )
        self.grid = tf.Variable(grid, trainable=False)

        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(out_features, in_features),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
        )
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(out_features, in_features, grid_size + spline_order),
            initializer=tf.keras.initializers.RandomNormal(stddev=scale_noise / grid_size),
            trainable=True,
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = self.add_weight(
                name='spline_scaler',
                shape=(out_features, in_features),
                initializer=tf.keras.initializers.HeNormal(),
                trainable=True,
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps

    def b_splines(self, x):
        # Expand x to have an additional last dimension for proper broadcasting
        x = tf.expand_dims(x, -1)  # Shape: (batch_size, num_features, 1)

        # Ensure grid is broadcastable across batches and features
        # Adding a batch and feature dimension to grid
        grid = tf.expand_dims(tf.expand_dims(self.grid, 0), 0)  # Shape: (1, 1, num_grid_points)

        # Initialize bases using the first order conditions
        bases = tf.logical_and(
            tf.greater_equal(x, grid[:, :, :-1]),  # x compared to the start of each interval
            tf.less(x, grid[:, :, 1:])            # x compared to the end of each interval
        )
        bases = tf.cast(bases, x.dtype)  # Convert boolean to float32 or the appropriate dtype

        # Recursive calculation of B-spline basis functions
        for k in range(1, self.spline_order + 1):
            # Grid segments adapted for the order k of B-splines
            grid_start = grid[:, :, :-(k + 1)]
            grid_end = grid[:, :, k:-1]
            grid_next = grid[:, :, (k + 1):]
            grid_mid = grid[:, :, 1:-(k)]

            # Calculate new bases with ensured broadcasting
            term1 = (x - grid_start) / (grid_end - grid_start) * bases[:, :, :-1]
            term2 = (grid_next - x) / (grid_next - grid_mid) * bases[:, :, 1:]
            bases = term1 + term2

        return bases


    def curve2coeff(self, x, y):
        A = tf.transpose(self.b_splines(x), perm=[1, 0, 2])
        B = tf.transpose(y, perm=[1, 0, 2])
        solution = tf.linalg.lstsq(A, B).solution
        result = tf.transpose(solution, perm=[2, 0, 1])
        return result

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (tf.expand_dims(self.spline_scaler, -1) if self.enable_standalone_scale_spline else 1.0)

    def call(self, x):
        # Ensuring dynamic shape compatibility
        batch_size = tf.shape(x)[0]  # Dynamically get the batch size

        # Computing the base output
        base_output = tf.matmul(self.base_activation(x), self.base_weight, transpose_b=True)

        # Computing the spline output
        # Ensure the b_splines method handles dynamic shapes appropriately
        b_splines_output = self.b_splines(x)

        # When reshaping, use the dynamically obtained batch size instead of None
        reshaped_b_splines = tf.reshape(b_splines_output, [batch_size, -1])
        reshaped_spline_weights = tf.reshape(self.scaled_spline_weight, [self.out_features, -1])

        spline_output = tf.matmul(reshaped_b_splines, reshaped_spline_weights, transpose_b=True)

        # Combining the base and spline outputs
        output = base_output + spline_output
        return output


    def update_grid(self, x, margin=0.01):
        batch = x.shape[0]
        splines = self.b_splines(x)
        splines = tf.transpose(splines, perm=[1, 0, 2])
        orig_coeff = self.scaled_spline_weight
        orig_coeff = tf.transpose(orig_coeff, perm=[1, 2, 0])
        unreduced_spline_output = tf.einsum('ijk,jkl->ijl', splines, orig_coeff)
        unreduced_spline_output = tf.transpose(unreduced_spline_output, perm=[1, 0, 2])
        x_sorted = tf.sort(x, axis=0)
        grid_adaptive = tf.gather(x_sorted, tf.cast(tf.linspace(0, batch - 1, self.grid_size + 1), dtype=tf.int32))
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            tf.range(self.grid_size + 1, dtype=tf.float32)[:, tf.newaxis]
            * uniform_step
            + x_sorted[0] - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.grid.assign(
            tf.concat([
                grid[:1] - uniform_step * tf.range(self.spline_order, 0, -1, dtype=tf.float32)[:, tf.newaxis],
                grid,
                grid[-1:] + uniform_step * tf.range(1, self.spline_order + 1, dtype=tf.float32)[:, tf.newaxis],
            ], axis=0)
        )
        self.spline_weight.assign(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.spline_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p))
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

#### KAN Model in TensorFlow python
class KAN(tf.keras.Model):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=tf.nn.silu,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers_ = []
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers_.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def call(self, x, update_grid=False):
        for layer in self.layers_:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return tf.add_n([
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers_
        ])