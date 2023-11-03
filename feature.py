import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

def momentum(df, var, days):
    tmp = df[var].shift(days)
    return (df[var] - tmp) / tmp

def get_window_info(direction):
    if '(' in direction and '*' in direction and ')' in direction:
        direction, window = direction.split('(')
        slide, count = window.replace(')', '').split('*')
        slide = int(slide); count = int(count)
    else:
        slide = 0; count = 0
        
    return direction, slide, count

def sliding_window(df, var, slide, count):
    for i in range(1, count):
        df[var + '_-' + str(i)] = df[var].shift(slide * i)
    return df

def _slope(data, scope, current_criterion = 'max', past_criterion = 'max'):
    if scope[3]:
        current_scope = data[(-scope[2]):(-scope[3])]
    else:
        current_scope = data[(-scope[2]):]
    
    if current_criterion == 'max': 
        curr_idx = np.argmax(current_scope)
        current = current_scope[curr_idx]
    elif current_criterion == 'min': 
        curr_idx = np.argmin(current_scope)
        current = current_scope[curr_idx]
    elif current_criterion == 'median': 
        current = np.median(current_scope)
        curr_idx = int((scope[2]-scope[3])/2) #min(enumerate(current_scope), key = lambda x: abs(x[1] - current))[0]
    elif current_criterion == 'mean': 
        current = current_scope.mean()
        curr_idx = int((scope[2]-scope[3])/2)
    elif current_criterion == 'last':
        current = current_scope[-1]
        curr_idx = scope[2]-scope[3]-1
    else:
        raise()
    curr_idx -= scope[2]
        
    if scope[1]:
        past_scope = data[(-scope[0]):(-scope[1])]
    else:
        current_scope = data[(-scope[0]):]
        
    if past_criterion == 'max': 
        past_idx = np.argmax(past_scope)
        past = past_scope[past_idx]
    elif past_criterion == 'min': 
        past_idx = np.argmin(past_scope)
        past = past_scope[past_idx]
    elif past_criterion == 'median': 
        past = np.median(past_scope)
        past_idx = int((scope[0]-scope[1])/2) #min(enumerate(past_scope), key = lambda x: abs(x[1] - past))[0]
    elif past_criterion == 'mean': 
        past = past_scope.mean()
        past_idx = int((scope[0]-scope[1])/2)
    elif past_criterion == 'last':
        past = past_scope[-1]
        past_idx = scope[0]-scope[1]-1
    else:
        raise()
    past_idx -= scope[0]
        
    return past_idx, curr_idx, past, current, (current - past) / (curr_idx - past_idx), np.log(current / past) / (curr_idx - past_idx) if current > 0 and past > 0 else None

def slope(df, var, scope, current_criterion = 'max', past_criterion = 'max', return_rate = True):
    scope = [int(x) for x in scope.split(',')]
    slopes = [None] * len(df)
    data = df[var].to_numpy()
    
    if return_rate:
        for i in range(scope[0], len(df)+1):
            p_idx, c_idx, p, c, sl, rr = _slope(data[i-scope[0]:i], scope)
            slopes[i-1] = rr
    else:
        for i in range(scope[0], len(df)+1):
            p_idx, c_idx, p, c, sl, rr = _slope(data[i-scope[0]:i], scope)
            slopes[i-1] = sl
        
    return slopes

def regression(df, var, days):
    X = np.arange(days)
    y = df[var].to_numpy()
    betas = [None] * len(df)
    
    for i in range(days, len(df)+1):
        beta, _ = np.polyfit(X, y[(i-days):i], 1)
        betas[i-1] = beta
    
    return betas

def MACD(df, var, num_fast = 12, num_slow = 26):
    EMAFast = df[var].ewm(span = num_fast, min_periods = num_fast - 1).mean()
    EMASlow = df[var].ewm(span = num_slow, min_periods = num_slow - 1).mean()
    return EMAFast - EMASlow

def PPO(df, var, num_fast = 12, num_slow = 26):
    EMAFast = df[var].ewm(span = num_fast, min_periods = num_fast - 1).mean()
    EMASlow = df[var].ewm(span = num_slow, min_periods = num_slow - 1).mean()
    return (EMAFast - EMASlow) / EMASlow * 100

def RSI(df, var, k = 14):
    diff = df[var].diff(1)
    U = pd.Series(np.where(diff > 0, diff, 0), index = df.index)
    D = pd.Series(np.where(diff < 0, -diff, 0), index = df.index)
    AU = U.rolling(window = k, min_periods = k).mean()
    AD = D.rolling(window = k, min_periods = k).mean()
    RSI = AU.div(AD+AU) * 100
    return RSI

def stochRSI(df, var, k = 14):
    rsi = RSI(df, var, k = k)
    rsi_min = rsi.rolling(window = k).min()
    rsi_max = rsi.rolling(window = k).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min) * 100

def bollinger_band(df, var, w = 20, k = 2):
    mbb = df[var].rolling(w).mean()
    std = df[var].rolling(w).std()
    ubb = mbb + k * std
    lbb = mbb - k * std
    return ((df[var] - lbb) / (ubb - lbb) - 0.5) * 2

def pibonacci(df, var, k):
    min_ = df[var].rolling(k).min()
    max_ = df[var].rolling(k).max()
    return (df[var] - min_) / (max_ - min_)

def standard_deviation(df, var, k):
    return df[var].rolling(window = k).std()

def minmax_gap(df, var, k, s):
    if s > 0:
        stds = df[var].rolling(window = s).std()
    else:
        stds = 1
    return (df[var].rolling(window = k).max() - df[var].rolling(window = k).min()) / stds

def spike(df, var, k):
    max_ = df[var].rolling(k).max().shift(1)
    return df[var] / max_

def ADX(df, var, k):
    low = var + '_low'; high = var + '_high'
    minusDM = df[low].shift(1) - df[low]
    plusDM = df[high] - df[high].shift(1)
    plusDM_ = pd.Series(np.where((plusDM > minusDM) & (plusDM > 0), plusDM, 0.0), index = df.index)
    minusDM_ = pd.Series(np.where((minusDM > plusDM) & (minusDM > 0), minusDM, 0.0), index = df.index)
    TR_TMP1 = df[high] - df[low]
    TR_TMP2 = np.abs(df[high] - df[var].shift(1))
    TR_TMP3 = np.abs(df[low] - df[var].shift(1))
    TR = np.maximum(np.maximum(TR_TMP1, TR_TMP2), TR_TMP3)
    
    TRI = TR.rolling(k).sum()
    plusDMI = plusDM_.rolling(k).sum()
    minusDMI = minusDM_.rolling(k).sum()
    plusDII = plusDMI / TRI * 100
    minusDII = minusDMI / TRI * 100
    
    numerator = np.abs(plusDII - minusDII)
    denominator = plusDII + minusDII
    DX = numerator / denominator * 100 
    ADX = DX.rolling(k).mean()
    
    return ADX

def ma(df, var, k):
    return df[var].rolling(k).mean()

def generation(df, param):
    param = param.fillna('')
                   
    for i in tqdm(list(range(len(param)))):
        request = param.iloc[i]
        var_name = request['var_name']
    
        for j, direction in enumerate(request['ma'].split(',')):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_ma_'+direction
            direction = int(direction)
            df[col_name] = ma(df, var_name, direction)
            df = sliding_window(df, col_name, slide, count)
    
        momentum_map = {}
        for j, direction in enumerate(request['momentum'].split(',')):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_momentum_'+direction
            direction = int(direction)
            df[col_name] = momentum(df, var_name, direction)
            df = sliding_window(df, col_name, slide, count)
            momentum_map[str(j)] = (col_name, direction)

        for direction in request['accel_mmt'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            a, b = direction.split(',')
            col_name = var_name+'_accel_mmt_'+direction
            df[col_name] = df[momentum_map[a][0]] / momentum_map[a][1] - df[momentum_map[b][0]] / momentum_map[b][1]
            df = sliding_window(df, col_name, slide, count)

        slope_map = {}
        for j, direction in enumerate(request['slope'].split('/')):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_slope_'+direction
            df[col_name] = slope(df, var_name, direction[:-2], return_rate = (direction[-1] == 'T'))
            df = sliding_window(df, col_name, slide, count)
            slope_map[str(j)] = col_name

        for direction in request['accel_slp'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            a, b = direction.split(',')
            col_name = var_name+'_accel_slp_'+direction
            df[col_name] = df[slope_map[a]] - df[slope_map[b]]
            df = sliding_window(df, col_name, slide, count)

        for direction in request['regression'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_regression_'+direction
            df[col_name] = regression(df, var_name, int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['macd'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_macd_'+direction
            a, b = direction.split(',')
            df[col_name] = MACD(df, var_name, num_fast = int(a), num_slow = int(b))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['ppo'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_ppo_'+direction
            a, b = direction.split(',')
            df[col_name] = PPO(df, var_name, num_fast = int(a), num_slow = int(b))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['rsi'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_rsi_'+direction
            df[col_name] = RSI(df, var_name, k = int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['stoch_rsi'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_stoch_rsi_'+direction
            df[col_name] = stochRSI(df, var_name, k = int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['bol_band'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_bol_band_'+direction
            w, k = direction.split(',')
            df[col_name] = bollinger_band(df, var_name, w = int(w), k = int(k))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['pibonacci'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_pibonacci_'+direction
            df[col_name] = pibonacci(df, var_name, int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['std'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_std_'+direction
            df[col_name] = standard_deviation(df, var_name, int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['minmax_gap'].split('/'):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_minmax_gap_'+direction
            k, s = direction.split(',')
            df[col_name] = minmax_gap(df, var_name, int(k), int(s))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['spike'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_spike_'+direction
            df[col_name] = spike(df, var_name, int(direction))
            df = sliding_window(df, col_name, slide, count)

        for direction in request['adx'].split(','):
            if direction == '': continue
            direction, slide, count = get_window_info(direction)
            col_name = var_name+'_adx_'+direction
            df[col_name] = ADX(df, var_name, int(direction))
            df = sliding_window(df, col_name, slide, count)
            
    return df

def aggregation(df, agg):
    agg = agg.fillna('')

    for i in tqdm(list(range(len(agg)))):
        request = agg.iloc[i]
        var_name = request['var_name']
        days = int(request['days'])

        if request['condition1']:
            col_name = var_name + '_' + request['function'] + '_' + request['condition1'] + request['condition2'] + '_' + request['days']
        else:
            col_name = var_name + '_' + request['function'] + '_' + request['days']


        if request['function'] == 'min':
            df[col_name] = df[var_name].rolling(window = days).min()
        elif request['function'] == 'max':
            df[col_name] = df[var_name].rolling(window = days).max()
        elif request['function'] == 'mean':
            df[col_name] = df[var_name].rolling(window = days).mean()
        elif request['function'] == 'median':
            df[col_name] = df[var_name].rolling(window = days).median()
        elif request['function'] == 'sum':
            df[col_name] = df[var_name].rolling(window = days).sum()
        elif request['function'] == 'std':
            df[col_name] = df[var_name].rolling(window = days).std()
        elif request['function'] == 'count':
            amount = float(request['condition2'])
            if request['condition1'] == '>':
                df[col_name] = df[var_name] > amount
            elif request['condition1'] == '>=':
                df[col_name] = df[var_name] >= amount
            elif request['condition1'] == '<':
                df[col_name] = df[var_name] < amount
            elif request['condition1'] == '<=':
                df[col_name] = df[var_name] <= amount
            elif request['condition1'] == '==':
                df[col_name] = df[var_name] == amount
            df[col_name] = df[col_name].rolling(window = days).sum()
            
    return df
