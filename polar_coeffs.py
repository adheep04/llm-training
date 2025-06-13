from math import inf , sqrt
import numpy as np

# https://arxiv.org/abs/2505.16932

def optimal_quintic(low, high):
    assert 0 <= low <= high
    if 1 - 5e-6 <= low / high:
        return (15/8)/high, (-10/8)/(high**3), (3/8)/(high**5)
    mid_low = (3*low + high) / 4
    mid_high = (low + 3*high) / 4
    error, prev_error = inf, None
    while not prev_error or abs(prev_error - error) > 1e-16:
        prev_error = error
        LHS = np.array([
            [low, low**3, low**5, 1],
            [mid_low, mid_low**3, mid_low**5, -1],
            [mid_high, mid_high**3, mid_high**5, 1],
            [high, high**3, high**5, -1],
        ])
        a, b, c, error = np.linalg.solve(LHS, np.ones(4))
        mid_low, mid_high = np.sqrt((-3*b + np.array([-1, 1]) * 
            sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)

def optimal_composition(low, num_steps, cushion=0.02407327424182761):
    high = 1
    coefficients = []
    for _ in range(num_steps):
        a, b, c = optimal_quintic(max(low, cushion*high), high)
        pl = a*low + b*low**3 + c*low**5
        pu = a*high + b*high**3 + c*high**5
        rescalar = 2/(pl + pu)
        a *= rescalar; b *= rescalar; c *= rescalar
        coefficients.append((a, b, c))
        low = a*low + b*low**3 + c*low**5
        high = 2 - low

    # safety factor for numerical stability ( but exclude last polynomial )
    coefficients = [( a / 1.01 , b / 1.01**3 , c / 1.01**5)
        for (a , b , c) in coefficients[:-1]] + [coefficients[-1]]
    return coefficients

def get_coeffs():
    return optimal_composition(1e-3, 9)

