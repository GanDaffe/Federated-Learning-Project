import math 
import numpy as np 

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)

    mask = (p != 0) & (q != 0)
    p = p[mask]
    q = q[mask]

    return np.sum(p * np.log(p / q))

def jensen_shannon_divergence_distance(p, q):
    p = np.array(p)
    q = np.array(q)

    m = (p + q) / 2

    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)

    return (kl_p_m + kl_q_m) / 2

def hellinger(p, q):
    """
    Hellinger distance between two discrete distributions.
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
        list_of_squares.append(s)

    sosq = sum(list_of_squares)    

    return sosq / math.sqrt(2)