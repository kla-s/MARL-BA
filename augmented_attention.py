import tensorflow as tf

from tensorpack import *


def shape_list(x):
    """Return list of dims, statically where possible."""
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret


def split_heads_2d(inputs, Nh):
    """Split channels into multiple heads."""
    B, H, W, d = shape_list(inputs)
    ret_shape = [B, H, W, Nh, d // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(inputs):
    """Combine heads (inverse of split_heads_2d)."""
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])  # [-1, H, W, Nh, dkh or dvh]
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return tf.reshape(transposed, ret_shape)  # [-1, H, W, Nh * (dkh or dvh)]


def self_attention_2d(inputs, dk, dv, Nh, relative=True):
    """2d relative self-attention."""
    _, H, W, _ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H * W, d])

    # Compute q, k, v
    kqv = tf.layers.conv2d(inputs, 2 * dk + dv, 1)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    q *= dkh ** -0.5  # scaled dot-product

    # After splitting, shape is [B, Nh, H, W, dkh or dvh]
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)
    # in: [-1, Nh, H * W, d] = [?, 4, 225, 1]
    logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)  # [B, Nh, HW, HW] = [?, 4, 225, 225]
    if relative:
        # ([B, Nh, HW, HW], [B, Nh, HW, HW]) = ([?, 4, 225, 225], [?, 4, 225, 225])
        rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh, dkh)
        logits += rel_logits_h
        logits += rel_logits_w
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_hw(v, dvh))  # [-1, Nh, H * W, dvh] = [?, 4, 225, 1]
    attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])  # [-1, Nh, H, W, dvh] = [?, 4, 15, 15, 1]
    attn_out = combine_heads_2d(attn_out)  # [-1, H, W, Nh * dvh] = [?, 15, 15, 4]
    # Project heads
    attn_out = tf.layers.conv2d(attn_out, dv, 1)  # [-1, H, W, Nh * dvh] = [?, 15, 15, 4]
    return attn_out


def rel_to_abs(x):
    """Converts tensor from relative to aboslute indexing.
    x_w: [B, Nh * H, W, 2 * W - 1]
    x_h: [B, Nh * W, H, 2 * H - 1]"""
    B, NhxH, L, _ = shape_list(x)  # [?, 60, 15, 29]
    # Pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((B, NhxH, L, 1))
    x = tf.concat([x, col_pad], axis=3)  # [?, 60, 15, 30]
    # w: [B, Nh*H, W, 2W]
    # h: [B, Nh*W, H, 2H]
    flat_x = tf.reshape(x, [B, NhxH, L * 2 * L])  # [?, 60, 450]
    flat_pad = tf.zeros((B, NhxH, L - 1))  # [?,60,14]
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)  # [?, 60, 464]
    # w: [B, Nh*H, 2W*W+(W-1)]
    # h: [B, Nh*W, 2H*H+(H-1)]
    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [B, NhxH, L + 1, 2 * L - 1])  # [?, 60, 16, 29]
    # w: [B, Nh*H, W+1, 2W-1]
    # h: [B, Nh*W, H+1, 2H-1]
    final_x = final_x[:, :, :L, L - 1:]  # [?, 60, 15, 15]
    # w: [B, Nh*H, W, W]
    # h: [B, Nh*W, H, H]
    return final_x


def relative_logits_1d(q, relk, H, W, Nh, transpose_mask):
    """Compute relative logits along one dimenion.
    q_w: [B, Nh, H, W, dkh]
    q_h: [B, Nh, W, H, dkh]
    """
    # out[b, h, x, y, m] = sum_d q[b, h, x, y, d] * relk[m, d]
    rel_logits = tf.einsum('bhxyd, md- > bhxym', q, relk)  # [?, 4, 15, 15, 29] = [B, Nh, H, W, 2W-1]
    # w: [B, Nh, H, W, 2W-1]
    # h: [B, Nh, W, H, 2H-1]
    # Collapse height and heads
    rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, 2 * W - 1])  # [?, 60, 15, 29]
    # w: [B, Nh * H, W, 2 * W - 1]
    # h: [B, Nh * W, H, 2 * H - 1]
    rel_logits = rel_to_abs(rel_logits)  # [?, 60, 15, 15]
    # w: [B, Nh*H, W, W]
    # h: [B, Nh*W, H, H]
    # Shape it and tile height times
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])  # [?, 4, 15, 15, 15]
    # w: [B, Nh, H1, W1, W2]
    # h: [B, Nh, W1, H1, H2]
    rel_logits = tf.expand_dims(rel_logits, axis=3)  # [?, 4, 15, 1, 15, 15]
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])  # [?, 4, 15, 15, 15, 15]= [-1, Nh,H1,H2,W1,W2]
    # w: [B, Nh, H1, H2, W1, W2]
    # h: [B, Nh, W1, W2, H1, H2]
    # Reshape for adding to the logits.
    rel_logits = tf.transpose(rel_logits, transpose_mask)  # [?, 4, 15, 15, 15, 15]
    # w: [B, Nh, H1, W1, H2, W2]; mask:[0, 1, 2, 4, 3, 5]
    # h: [B, Nh, H1, W1, H2, W2]; mask:[0, 1, 4, 2, 5, 3]
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H * W, H * W])  # [?, 4, 255, 255]
    return rel_logits


def relative_logits(q, H, W, Nh, dkh):
    """Compute relative logits.
    q: [?, 4, 15, 15, 1] = [B, Nh, H, W, dkh or dvh]"""
    # Relative logits in width dimension first.
    rel_embeddings_w = tf.get_variable('r_width', shape=(2 * W - 1, dkh), initializer=tf.random_normal_initializer(
        dkh ** -0.5))  # [29,1]
    # in: [B, Nh, H, W, dkh]
    rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh,
                                      [0, 1, 2, 4, 3, 5])  # [?, 4, 255, 255] = [B, Nh, HW, HW]
    # Relative logits in height dimension next.
    # For ease, we 1) transpose height and width,
    # 2) repeat the above steps and
    # 3) transpose to eventually put the logits
    # in their right positions.
    rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh), initializer=tf.random_normal_initializer(
        dkh ** -0.5))  # [29,1]
    # in: [B, Nh, W, H, dkh]
    rel_logits_h = relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]), rel_embeddings_h, W, H, Nh,
                                      [0, 1, 4, 2, 5, 3])  # out: [?, 4, 255, 255]
    return rel_logits_h, rel_logits_w


@layer_register(log_shape=True)
def augmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
    """
    X: 2d input
    Fout: Output filters
    k: kernel size
    dk: depth of keys
    dv: depth of values
    Nh: number heads, divides evenly by dk and dv
    relative: use relative?
    """
    convout = tf.layers.conv2d(inputs=X, filters=Fout - dv, kernel_size=k, padding='same')
    attnout = self_attention_2d(X, dk, dv, Nh, relative=relative)
    return tf.concat([convout, attnout], axis=3)


########## 3D #############################

def split_heads_3d(inputs, Nh):
    """Split channels into multiple heads. 3D"""
    B, H, W, D, d = shape_list(inputs)
    ret_shape = [B, H, W, D, Nh, d // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 4, 1, 2, 3, 5])  # [B, Nh, H, W, D, dkh]


def combine_heads_3d(inputs):
    """Combine heads (inverse of split_heads_3d).
    inputs: [-1, Nh, H, W, D, dkh or dvh]
    outputs: [-1, H, W, D, Nh * (dkh or dvh)]"""
    transposed = tf.transpose(inputs, [0, 2, 3, 4, 1, 5])  # [-1, H, W, D, Nh, dkh or dvh]
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return tf.reshape(transposed, ret_shape)  # [-1, H, W, D, Nh * (dkh or dvh)]


def rel_to_abs_3d(x):
    """Converts tensor from relative to aboslute indexing.
    x_w: (B, Nh*D, H, W, 2W-1)
    x_h: (B, Nh*W, D, H, 2H-1)
    x_d: (B, Nh*H, W, D, 2D-1)
    out: [B, D, Nh, L, L]"""
    B, NhxD, H, L, _ = shape_list(x)  # [?, 60, 15, 15, 29]
    # Pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((B, NhxD, H, L, 1))
    x = tf.concat([x, col_pad], axis=4)  # (?, 60, 15, 15, 30)
    # w: (B, Nh*D, H, W, 2W)
    # h: (B, Nh*W, D, H, 2H)
    # d: (B, Nh*H, W, D, 2D)
    flat_x = tf.reshape(x, [B, NhxD, H, L * 2 * L])  # (?, 60, 15, 450)
    # w: (B, Nh*D, H, W2W)
    # h: (B, Nh*W, D, H2H)
    # d: (B, Nh*H, W, D2D)
    flat_pad = tf.zeros((B, NhxD, H, L - 1))  # (?, 60, 15, 14)
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=3)  # (?, 60, 15, 464)
    # w: (B, Nh*D, H, W2W+(W-1))
    # h: (B, Nh*W, D, H2H+(H-1))
    # d: (B, Nh*H, W, D2D+(D-1))
    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [B, NhxD, H, L + 1, 2 * L - 1])  # (?, 60, 15, 16, 29)
    # w: (B, Nh*D, H, W+1, 2W-1)
    # h: (B, Nh*W, D, H+1, 2H-1)
    # d: (B, Nh*H, W, D+1, 2D-1)
    final_x = final_x[:, :, :, :L, L - 1:]  # (?, 60, 15, 15, 15)
    # w: (B, Nh*D, H, W, W)
    # h: (B, Nh*W, D, H, H)
    # d: (B, Nh*H, W, D, D)
    return final_x


def relative_logits_1d_3d(q, relk, H, W, D, Nh, transpose_mask):
    """Compute relative logits along one dimenion.
    q_w: (B, Nh, D, H, W, dkh)
    q_h: (B, Nh, W, D, H, dkh)
    q_d: (B, Nh, H, W, D, dkh)
    """
    # out[b, h, x, y, m] = sum_d q[b, h, x, y, d] * relk[m, d]
    rel_logits = tf.einsum('bhxyzd, md- > bhxyzm', q, relk)  # [?, 4, 15, 15, 15, 29]
    # w: (B, Nh, D, H, W, 2W-1)
    # h: (B, Nh, W, D, H, 2H-1)
    # d: (B, Nh, H, W, D, 2D-1)
    # Collapse height and heads
    rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, D, 2 * W - 1])  # (?, 60, 15, 15, 29)
    # w: (B, Nh*D, H, W, 2W-1)
    # h: (B, Nh*W, D, H, 2H-1)
    # d: (B, Nh*H, W, D, 2D-1)
    rel_logits = rel_to_abs_3d(rel_logits)  # (?, 60, 15, 15, 15)
    # Shape it and tile height times
    # w: (B, Nh*D, H, W, W)
    # h: (B, Nh*W, D, H, H)
    # d: (B, Nh*H, W, D, D)
    rel_logits = tf.reshape(rel_logits,
                            [-1, Nh, H, W, D, D])  # (?, 4, 15, 15, 15, 15, 15) = [-1, Nh, H1, W1, W2, D1, D2]
    # TODO wie H W D festlegen? alles durcheinander
    # w: (B, Nh, D, H, W, W)
    # h: (B, Nh, W, D, H, H)
    # d: (B, Nh, H, W, D, D)
    rel_logits = tf.expand_dims(rel_logits, axis=4)  # (?, 4, 15, 1, 15, 15, 15, 15) = (B, Nh, H1, 1, W1, W2, D1, D2)
    rel_logits = tf.tile(rel_logits,
                         [1, 1, 1, 1, W, 1, 1])  # (?, 4, 15, 15, 15, 15, 15, 15) = (B, Nh, H1, H2, W1, W2, D1, D2)
    # w: (B, Nh, D1, H1, H2, W1, W2)
    # h: (B, Nh, W1, D1, D2, H1, H2)
    # d: (B, Nh, H1, W1, W2, D1, D2)
    rel_logits = tf.expand_dims(rel_logits, axis=3)  # (?, 4, 15, 1, 15, 15, 15, 15) = (B, Nh, H1, 1, W1, W2, D1, D2)
    rel_logits = tf.tile(rel_logits,
                         [1, 1, 1, H, 1, 1, 1, 1])
        # 0  1   2    3  4   5   6   7
    # w: (B, Nh, D1, D2, H1, H2, W1, W2)
    # h: (B, Nh, W1, W2, D1, D2, H1, H2)
    # d: (B, Nh, H1, H2, W1, W2, D1, D2)
    # Reshape for adding to the logits.
    rel_logits = tf.transpose(rel_logits, transpose_mask)  # [?, 4, 15, 15, 15, 15]
    # w: (B, Nh, H1, W1, D1, H2, W2, D2) with mask: [0, 1, 4, 6, 2, 5, 7, 3]
    # h: (B, Nh, H1, W1, D1, H2, W2, D2) with mask: [0, 1, 6, 2, 4, 7, 3, 5]
    # d: (B, Nh, H1, W1, D1, H2, W2, D2) with mask: [0, 1, 2, 4, 6, 3, 5, 7]
    rel_logits = tf.reshape(rel_logits, [-1, Nh, H * W * D, H * W * D])  # [?, 4, 255, 255]
    return rel_logits


def relative_logits_3d(q, H, W, D, Nh, dkh):
    """Compute relative logits. 3D.
    q: (B, Nh, H, W, D, dkh)"""
    # Relative logits in width dimension first.
    rel_embeddings_w = tf.get_variable('r_width', shape=(2 * W - 1, dkh), initializer=tf.random_normal_initializer(
        dkh ** -0.5))  # (29,1) = (2 * W - 1, dkh)
    q_trans = tf.transpose(q, [0, 1, 4, 2, 3, 5])
    # in: (?, 4, 15, 15, 15, 1) = (B, Nh, D, H, W, dkh); out: [B, Nh, HWD, HWD]
    rel_logits_w = relative_logits_1d_3d(q_trans, rel_embeddings_w, D, H, W, Nh, [0, 1, 4, 6, 2, 5, 7, 3])  #
    # Relative logits in height dimension next.
    # For ease, we 1) transpose height and width,
    # 2) repeat the above steps and
    # 3) transpose to eventually put the logits
    # in their right positions.
    rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh), initializer=tf.random_normal_initializer(
        dkh ** -0.5))
    q_trans = tf.transpose(q, [0, 1, 3, 4, 2, 5])
    # in: (?, 4, 15, 15, 15, 1) = (B, Nh, W, D, H, dkh); out: [B, Nh, HWD, HWD]
    rel_logits_h = relative_logits_1d_3d(q_trans, rel_embeddings_h, W, D, H, Nh,
                                         [0, 1, 6, 2, 4, 7, 3, 5])
    # do the third analogous
    rel_embeddings_d = tf.get_variable('r_depth', shape=(2 * D - 1, dkh), initializer=tf.random_normal_initializer(
        dkh ** -0.5))
    # in: (?, 4, 15, 15, 15, 1) = (B, Nh, H, W, D, dkh); out: [B, Nh, HWD, HWD]
    rel_logits_d = relative_logits_1d_3d(q, rel_embeddings_d, H, W, D, Nh, [0, 1, 2, 4, 6, 3, 5, 7])
    return rel_logits_h, rel_logits_w, rel_logits_d


def self_attention_3d(inputs, dk, dv, Nh, relative=True, reuse=None):
    """2d relative self-attention."""
    _, H, W, D, _ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    flatten_hwd = lambda x, d: tf.reshape(x, [-1, Nh, H * W * D, d])

    # Compute q, k, v
    kqv = tf.layers.conv3d(inputs, filters=2 * dk + dv, kernel_size=1, reuse=reuse)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=4)
    # if dkh == 0.0:
    #     dkh += 0.000001
    q *= dkh ** -0.5  # scaled dot-product

    # After splitting, shape is [B, Nh, H, W, D, dkh or dvh]
    q = split_heads_3d(q, Nh)
    k = split_heads_3d(k, Nh)
    v = split_heads_3d(v, Nh)
    # [B, Nh, HWD, HWD]
    logits = tf.matmul(flatten_hwd(q, dkh), flatten_hwd(k, dkh), transpose_b=True)
    if relative:
        rel_logits_h, rel_logits_w, rel_logits_d = relative_logits_3d(q, H, W, D, Nh,
                                                                      dkh)  # TODO: here we need to get rel_logits_d
        logits += rel_logits_h
        logits += rel_logits_w
        logits += rel_logits_d
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_hwd(v, dvh))
    attn_out = tf.reshape(attn_out, [-1, Nh, H, W, D, dvh])
    attn_out = combine_heads_3d(attn_out)
    # Project heads
    attn_out = tf.layers.conv3d(attn_out, dv, 1, reuse=reuse)
    return attn_out


@layer_register(log_shape=True)
def augmented_conv3d(X, filters, kernel_size, dk, dv, Nh, relative, reuse):
    """
        X: 3d input
        Fout: Output filters
        k: kernel size
        dk: depth of keys
        dv: depth of values
        Nh: number heads, divides evenly by dk and dv
        relative: use relative?
        """
    convout = tf.layers.conv3d(inputs=X, filters=filters - dv, kernel_size=kernel_size, padding='same', reuse=reuse)
    attnout = self_attention_3d(X, dk, dv, Nh, relative=relative, reuse=reuse)
    return tf.concat([convout, attnout], axis=4)


########## 1D #####################

def split_heads_1d(inputs, Nh):
    """Split channels into multiple heads."""
    B, W, d = shape_list(inputs)
    ret_shape = [B, W, Nh, d // Nh]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 2, 1, 3])


def combine_heads_1d(inputs):
    """Combine heads (inverse of split_heads_2d)."""
    transposed = tf.transpose(inputs, [0, 2, 1, 3])  # [-1, W, Nh, dkh or dvh]
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return tf.reshape(transposed, ret_shape)  # [-1, W, Nh * (dkh or dvh)]


def self_attention_1d(inputs, dk, dv, Nh, relative=True):
    """1d relative self-attention."""
    _, W, _ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    flatten_w = lambda x, d: tf.reshape(x, [-1, Nh, W, d])

    # Compute q, k, v
    kqv = tf.layers.conv1d(inputs, 2 * dk + dv, 1)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    q *= dkh ** -0.5  # scaled dot-product

    # After splitting, shape is [B, Nh, H, W, dkh or dvh]
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)
    # in: [-1, Nh, H * W, d] = [?, 4, 225, 1]
    logits = tf.matmul(flatten_w(q, dkh), flatten_w(k, dkh), transpose_b=True)  # [B, Nh, HW, HW] = [?, 4, 225, 225]
    if relative:
        # ([B, Nh, HW, HW], [B, Nh, HW, HW]) = ([?, 4, 225, 225], [?, 4, 225, 225])
        rel_logits_h, rel_logits_w = relative_logits(q, W, Nh, dkh)
        logits += rel_logits_h
        logits += rel_logits_w
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_w(v, dvh))  # [-1, Nh, H * W, dvh] = [?, 4, 225, 1]
    attn_out = tf.reshape(attn_out, [-1, Nh, W, dvh])  # [-1, Nh, H, W, dvh] = [?, 4, 15, 15, 1]
    attn_out = combine_heads_2d(attn_out)  # [-1, H, W, Nh * dvh] = [?, 15, 15, 4]
    # Project heads
    attn_out = tf.layers.conv2d(attn_out, dv, 1)  # [-1, H, W, Nh * dvh] = [?, 15, 15, 4]
    return attn_out


@layer_register(log_shape=True)
def augmented_conv1d(X, Fout, k, dk, dv, Nh, relative):
    """
    X: 2d input
    Fout: Output filters
    k: kernel size
    dk: depth of keys
    dv: depth of values
    Nh: number heads, divides evenly by dk and dv
    relative: use relative?
    """
    convout = tf.layers.conv1d(inputs=X, filters=Fout - dv, kernel_size=k, padding='same')
    attnout = self_attention_1d(X, dk, dv, Nh, relative=relative)
    return tf.concat([convout, attnout], axis=3)
