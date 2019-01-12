import random
import numpy as np


# State transition table
TRANSITIONS = [
    [('T', 1), ('P', 2)],  # 0=B
    [('X', 3), ('S', 1)],  # 1=BT
    [('V', 4), ('T', 2)],  # 2=BP
    [('X', 2), ('S', 5)],  # 3=BTX
    [('P', 3), ('V', 5)],  # 4=BPV
    [('E', -1)],  # 5=BTXS
]

TRANSITIONS_EMB = [
    [('T', 1), ('P', 11)],  
    [('B', 1), ('P', 11)],  
    [('X', 3), ('S', 1)],  
    [('V', 4), ('T', 2)],  
    [('X', 2), ('S', 5)],  
    [('P', 3), ('V', 5)],  
    [('E', -1)],  
]

# Symbol encoding
SYMS = {'T': 0, 'P': 1, 'X': 2, 'S': 3, 'V': 4, 'B': 5, 'E': 6}


def make_reber():
    """Generate one string from the Reber grammar."""
    idx = 0
    out = 'B'
    while idx != -1:
        ts = TRANSITIONS[idx]
        symbol, idx = random.choice(ts)
        out += symbol
    return out


def make_embedded_reber():
    """Generate one string from the embedded Reber grammar."""
    c = random.choice(['T', 'P'])
    return 'B%s%s%sE' % (c, make_reber(), c)


def str_to_vec(s):
    """Convert a (embedded) Reber string to a sequence of unit vectors."""
    a = np.zeros((len(s), len(SYMS)))
    for i, c in enumerate(s):
        a[i][SYMS[c]] = 1
    return a


def str_to_next(s):
    """Given a Reber string, return a vectorized sequence of next chars.
    This is the target output of the Neural Net."""
    out = np.zeros((len(s), len(SYMS)))
    idx = 0
    for i, c in enumerate(s[1:]):
        ts = TRANSITIONS[idx]
        for next_c, _ in ts:
            out[i, SYMS[next_c]] = 1

        next_idx = [j for next_c, j in ts if next_c == c]
        assert len(next_idx) == 1
        idx = next_idx[0]

    return out


def str_to_next_embed(s):
    """Given an embedded Reber string, return a vectorized sequence of next chars.
       This is the target output of the Neural Net for embedded Reber."""
   
    out_reb = str_to_next(s[2:-2])
    ns = out_reb.shape[1]
    out = np.zeros((out_reb.shape[0]+4,ns))
    out[0,SYMS['T']]=1
    out[0,SYMS['P']]=1
    out[1,SYMS['E']]=1
    for i, st in enumerate(out_reb):
        out[i+2,:] = st
    idx = i+2
    out[idx,SYMS[s[1]]]=1
    out[idx+1,SYMS['E']]=1
    return out


def vec_to_str(xs):
    """Given a matrix, return a Reber string (with choices)."""
    #idx_to_sym = dict((v,k) for k,v in SYMS.iteritems())
    idx_to_sym = dict((v,k) for k,v in SYMS.items())
    out = ''
    for i in range(0, xs.shape[0]):
        vs = np.nonzero(xs[i,:])[0]
        chars = [idx_to_sym[v] for v in vs]
        if len(chars) == 1:
            out += chars[0]
        else:
            out += '{%s}' % ','.join(chars)
    return out


