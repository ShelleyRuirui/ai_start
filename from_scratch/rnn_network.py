import numpy as np
from from_scratch.math_util import sigmoid
from from_scratch.math_util import softmax

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


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", of shape (n_a, m)
    parameters -- python dictionary containing parameter metrics
                        
    Returns:
    a_next -- of shape (n_a, m)
    c_next -- of shape (n_a, m)
    yt_pred -- of shape (n_y, m)
    cache -- contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    """

    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate([a_prev, xt], axis=0)
    forget_gate = sigmoid(np.dot(Wf,concat) + bf)
    update_gate = sigmoid(np.dot(Wi,concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.multiply(forget_gate, c_prev) + np.multiply(update_gate, cct)
    output_gate = sigmoid(np.dot(Wo, concat) + bo)
    a_next = np.multiply(output_gate, np.tanh(c_next))

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy,a_next)+by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, forget_gate, update_gate, cct, output_gate, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing parameter metrics
                        
    Returns:
    a -- Hidden states of shape (n_a, m, T_x)
    y -- Predictions of shape (n_y, m, T_x)
    c -- The value of the cell state of shape (n_a, m, T_x)
    caches -- contains (list of all the caches, x)
    """

    caches = []
    
    Wy = parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m ,T_x))
    c = np.zeros((n_a, m ,T_x))
    y = np.zeros((n_y,m,T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    for t in range(T_x):
        xt = x[:,:,t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        y[:,:,t] = yt
        caches.append(cache)
        
    caches = (caches, x)

    return a, y, c, caches