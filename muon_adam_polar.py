import torch
import polar_coeffs

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups, use_polar=True, steps=5):
        self.coeffs = polar_coeffs.get_coeffs()[:steps] if use_polar else [(3.4445, -4.7750, 2.0315)] * steps
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
        update = self.zeropower_via_polar_express(update)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update

    def adam_update(self, grad, buf1, buf2, step, betas, eps):
        buf1.lerp_(grad, 1 - betas[0])
        buf2.lerp_(grad.square(), 1 - betas[1])
        buf1c = buf1 / (1 - betas[0]**step)
        buf2c = buf2 / (1 - betas[1]**step)
        return buf1c / (buf2c.sqrt() + eps)

    @torch.compile
    def zeropower_via_polar_express(self, G):
        assert G.ndim >= 2
        X = G.to(torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01)
        # Perform the NS iterations
        for a, b, c in self.coeffs:
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

