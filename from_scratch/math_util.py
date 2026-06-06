import numpy as np

def softmax(z):
    """
    Compute numerically stable softmax
    Args:
        z: raw logits, shape (n_categories, batch_size)
    Returns:
        Probability distribution (sum to 1 along each column)
    """
    # z_max: take max value per column to avoid exp() numerical overflow
    # keepdims make z_max (1, batch_size)
    z_max = np.max(z, axis=0, keepdims=True)

    # Shift all values by column max: z - max(z) keeps inputs ≤ 0
    exp_z = np.exp(z - z_max)

    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))