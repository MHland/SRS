import numpy as np

def to_float_numpy(*args):
    l = []
    if isinstance(args[0], float) or isinstance(args[0], int):
        for i in range(len(args)):
            l.append(np.array([args[i] * 1.0]))
    else:
        for i in range(len(args)):
            l.append(np.array(args[i]) * 1.0)
    return tuple(l)

def to_float_number(*args):
    l = []
    if isinstance(args[0], float) or isinstance(args[0], int):
        for i in range(len(args)):
            l.append(float(args[i]))
    elif isinstance(args[0][0], float) or isinstance(args[0][0], int):
        for i in range(len(args)):
            l.append(float(args[i][0]))
    else:
        raise TypeError('Check the type of input please!')
    return tuple(l)

def transform_to_vector(s, *args):
    l = []
    for i in range(len(args)):
        if isinstance(args[i], dict):
            dict_value = {}
            for key in args[i]:
                value = np.array(args[i][key]) * np.ones((s, 1))
                dict_value[key] = value[:, 0]
            l.append(dict_value)
        else:
            l.append((np.array(args[i]) * np.ones((s, 1))).T)
    return tuple(l)

def warm_up_code(Prec, ETp, Q_obs, warm_up, warm_up_year):
    Prec = np.array(Prec)
    ETp = np.array(ETp)
    Q_obs = np.array(Q_obs)
    Q_obs_pre = Q_obs
    l = max(Prec.shape)
    warm_up_time = warm_up_year
    if warm_up == True:
        warm_up_time = 365 * warm_up_year
        if l < warm_up_time:
            warm_up_time = max(Prec.shape)
        Prec1 = Prec[0: warm_up_time]
        ETp1 = ETp[0: warm_up_time]
        Q_obs1 = Q_obs[0: warm_up_time]
        Prec = np.append(Prec1, Prec)
        ETp = np.append(ETp1, ETp)
        Q_obs = np.append(Q_obs1, Q_obs)
    return Prec, ETp, Q_obs, Q_obs_pre, warm_up_time, l

def warm_up_type1(srs, parameter):
    print('-' * 88)
    print('-' * 88)
    print('The result of SRS algorithm:')
    srs.type_result()
    print('*' * 88)
    print('NSE:%.8f'%(1 - srs.get_result()[1][0]))
    # print(1 - srs.get_result()[1][0])
    print('*' * 88)
    print('Parameter value: ')
    i = 0
    for key in parameter.keys():
        if i == 4:
            print(key + ': ' + '{:^4.4f}'.format(parameter[key]), end='\n')
            i = 0
        else:
            print(key + ': ' + '{:^4.4f}'.format(parameter[key]), end='\t\t')
            i = i+1

def warm_up_type2(NSE, Qsim_out, Q_obs_pre, initial_state):
    print('*' * 88)
    print('NSE without warm-up:')
    print(NSE(Qsim_out, Q_obs_pre)[0])
    print('*' * 88)
    print('initial state value after warm-up: ')
    print(initial_state)

def warm_up_type3():
    print('-' * 88)
    print('-' * 88)