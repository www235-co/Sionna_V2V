import tensorflow as tf

def reconstruct(snr, num_vehicle):
    num_vehicle = num_vehicle 
    dim1, dim2, dim3, _  = snr.shape 
    snr = snr

    i, j = tf.meshgrid(tf.range(num_vehicle), tf.range(num_vehicle), indexing='ij')
    non_diag_mask = tf.not_equal(i, j)
    flat_indices = i * num_vehicle + j 
    flat_non_diag_indices = flat_indices[non_diag_mask] 

    target_shape = (dim1, dim2, dim3, num_vehicle*num_vehicle)
    target_tensor = tf.fill(target_shape, tf.constant(float('inf'))) 

    x, y, z = tf.meshgrid(tf.range(dim1), tf.range(dim2), tf.range(dim3), indexing='ij')
    flat_x = tf.reshape(x, [-1])
    flat_y = tf.reshape(y, [-1])
    flat_z = tf.reshape(z, [-1])

    num_prev_dims = dim1 * dim2 * dim3
    flat_non_diag_repeated = tf.tile(flat_non_diag_indices[tf.newaxis, :], [num_prev_dims, 1])

    scatter_indices = tf.stack([tf.repeat(flat_x, num_vehicle*(num_vehicle-1)),
                                tf.repeat(flat_y, num_vehicle*(num_vehicle-1)),
                                tf.repeat(flat_z, num_vehicle*(num_vehicle-1)),
                                tf.reshape(flat_non_diag_repeated, [-1])],
                                axis=1)

    flat_original = tf.reshape(snr, [-1])

    target_tensor = tf.tensor_scatter_nd_update(target_tensor, scatter_indices, flat_original)
    return target_tensor