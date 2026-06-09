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


def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- needed data from the forward pass of the current time-step

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    dtanh = da_next*(1-np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev)+ba)**2)
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients


def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- needed data from the forward pass of the current time-step
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- the input data gradient, numpy-array of shape (n_x, m, T_x)
                        da0 -- the initial hidden state gradient, numpy-array of shape (n_a, m)
                        dWax -- the input's weight matrix gradient, numpy-array of shape (n_a, n_x)
                        dWaa -- the hidden state's weight matrix gradient, numpy-arrayof shape (n_a, n_a)
                        dba -- the bias gradient, of shape (n_a, 1)
    """
        
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape 
    
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    
    for t in reversed(range(T_x)):
        # Compute gradients for time step t. Use da_prevt to pass the gradient of the hidden state to the previous time step
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:, :, t] = dxt  
        dWax += dWaxt  
        dWaa += dWaat  
        dba += dbat  
        
    da0 = da_prevt
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_a, m = a_next.shape
    
    tanh_c_next = np.tanh(c_next)
    dot = da_next * tanh_c_next * ot * (1 - ot)
    dc_next_total = dc_next + da_next * ot * (1 - tanh_c_next ** 2)
    dcct = dc_next_total * it * (1 - cct ** 2)
    dit = dc_next_total * cct * it * (1 - it)
    dft = dc_next_total * c_prev * ft * (1 - ft)
    
    concat = np.concatenate([a_prev, xt], axis=0)
    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    dconcat = (np.dot(parameters['Wf'].T, dft)
               + np.dot(parameters['Wi'].T, dit)
               + np.dot(parameters['Wc'].T, dcct)
               + np.dot(parameters['Wo'].T, dot))
    da_prev = dconcat[:n_a, :]
    dxt = dconcat[n_a:, :]
    dc_prev = dc_next_total * ft

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients


def lstm_backward(da, caches):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros_like(parameters['Wf'])
    dWi = np.zeros_like(parameters['Wi'])
    dWc = np.zeros_like(parameters['Wc'])
    dWo = np.zeros_like(parameters['Wo'])
    dbf = np.zeros_like(parameters['bf'])
    dbi = np.zeros_like(parameters['bi'])
    dbc = np.zeros_like(parameters['bc'])
    dbo = np.zeros_like(parameters['bo'])
    
    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        da_prevt = gradients['da_prev']
        dc_prevt = gradients['dc_prev']
        dx[:, :, t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']
    da0 = da_prevt
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients