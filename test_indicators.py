import numpy as np
def NSE(Q_sim, Q_obs):
    '''
    Calculate the Nash efficiency coefficient of the model
    :param Q_sim: runoff flow simulation value, type: list or numpy
    :param Q_obs: runoff flow observation value, type: list or numpy
    :return: nse
    '''
    Q_sim = np.array(Q_sim)
    Q_obs = np.array(Q_obs)
    m2 = Q_obs.shape[0]
    Q2 = np.zeros((m2,1))
    if len(Q_obs.shape) == 1:
        Q2[:,0] = Q_obs
    elif Q_obs.shape[1] != 1:
        Q2[:, 0] = Q_obs
    else:
        Q2[:, 0:1] = Q_obs
    if Q_sim[0].shape == ():
        Q1 = np.zeros((m2,1))
        Q1[:, 0] = Q_obs
    else:
        m1,n1 = Q_sim.shape
        if m1 == m2:
            Q1 = Q_sim
        elif n1 == m2:
            Q1 = Q_sim.T
        else:
            Q1 = None
    Q_mean = np.mean(Q2)
    Q2 = Q2*np.ones(Q1.shape)
    nse = 1-np.sum((Q1-Q2)**2, axis=0)/np.sum((Q2-Q_mean)**2, axis=0)
    return nse