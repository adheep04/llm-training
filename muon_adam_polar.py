import torch

from math import inf , sqrt
import numpy as np

# https://arxiv.org/pdf/2505.02222
# https://arxiv.org/abs/2505.16932
class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups, use_polar=True, steps=5):
        self.coeffs = get_coeffs()[:steps] if use_polar else [(3.4445, -4.7750, 2.0315)] * steps
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = self.muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = self.adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


    def muon_update(self, grad, momentum, beta=0.95, nesterov=True):
        momentum.lerp_(grad, 1 - beta)
        update = grad.lerp_(momentum, beta) if nesterov else momentum
        if update.ndim == 4: # for the case of conv filters
            update = update.view(len(update), -1)
        update = zeropower_via_polar_express(update, self.coeffs)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update

    def adam_update(self, grad, buf1, buf2, step, betas, eps):
        buf1.lerp_(grad, 1 - betas[0])
        buf2.lerp_(grad.square(), 1 - betas[1])
        buf1c = buf1 / (1 - betas[0]**step)
        buf2c = buf2 / (1 - betas[1]**step)
        return buf1c / (buf2c.sqrt() + eps)

@torch.compile
def zeropower_via_polar_express(G, coeffs):
    assert G.ndim >= 2
    X = G.to(torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01)
    # Perform the NS iterations
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A 
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

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

