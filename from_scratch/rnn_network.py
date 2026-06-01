import numpy as np
from softmax import softmax

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- numpy array of shape (n_a, n_x)
                        Waa -- numpy array of shape (n_a, n_a)
                        Wya -- numpy array of shape (n_y, n_a)
                        ba -- numpy array of shape (n_a, 1)
                        by -- numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- (a_next, a_prev, xt, parameters)
    """
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev)+ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya,a_next) + by)

    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- numpy array of shape (n_a, n_x)
                        Waa -- numpy array of shape (n_a, n_a)
                        Wya -- numpy array of shape (n_y, n_a)
                        ba -- numpy array of shape (n_a, 1)
                        by -- numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states, of shape (n_a, m, T_x)
    y_pred -- Predictions, of shape (n_y, m, T_x)
    caches -- (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    _, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    a_next = a0
    
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t],a_next,parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    
    caches = (caches, x)
    
    return a, y_pred, caches