
import numpy as np


def load_weights(variables, file, load_full_weights):

    with open(file, "rb") as f:
        # Skip first 5 values containing irrelevant info
        np.fromfile(f, dtype=np.float32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
        ptr = 0
        # Load weights for DarkNet53
        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, moving_average, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, moving_average, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                var.assign(var_weights)

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))

            ptr += num_params
            conv_var.assign(var_weights)
        # Load weights for YOLO Conv Block

        ranges = [range(0, 6), range(6, 12), range(12, 18)]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i

                conv_var = variables[current]
                gamma, beta, moving_average, variance = \
                    variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, moving_average, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    var.assign(var_weights)

                shape = conv_var.get_shape().as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                conv_var.assign(var_weights)

            # Load weights for YOLO detections
            if load_full_weights:
                bias = variables[360 + j * 2 + 1]
                shape = bias.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                bias.assign(var_weights)

                conv_var = variables[360 + j * 2]
                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))

                ptr += num_params
                conv_var.assign(var_weights)
            ptr_list = [56629087, 60898910]
            # Load weights for discrete DBLs
            if j <= 1:
                ptr = ptr_list[j]
                DBL_conv = variables[350 + j * 5]
                gamma, beta, moving_average, variance = \
                    variables[350 + j * 5 + 1:350 + j * 5 + 5]
                batch_norm_vars = [beta, gamma, moving_average, variance]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    var.assign(var_weights)

                shape = DBL_conv.get_shape().as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                DBL_conv.assign(var_weights)




