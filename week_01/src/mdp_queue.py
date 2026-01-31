import numpy as np

class QueuePricingMDP:
    """
    Finite-state queue pricing MDP:
      state n = number in system (0..N)
      action a = price level
      arrivals depend on price via lambda(a)
      service rate mu
      discrete time step dt
    """

    def __init__(self, N, prices, lambda0, beta, mu, cw, dt, gamma):
        self.N = N
        self.prices = np.array(prices, dtype=float)
        self.lambda0 = float(lambda0)
        self.beta = float(beta)
        self.mu = float(mu)
        self.cw = float(cw)
        self.dt = float(dt)
        self.gamma = float(gamma)

        self.S = np.arange(N + 1)
        self.A = np.arange(len(self.prices))

        # Precompute transition matrices and rewards
        self.P = self._build_transition_tensor()   # shape: (A, S, S)
        self.R = self._build_reward_matrix()       # shape: (S, A)

    def lam(self, a_idx):
        p = self.prices[a_idx]
        return self.lambda0 * np.exp(-self.beta * p)

    def _build_transition_tensor(self):
        A = len(self.A)
        S = len(self.S)
        P = np.zeros((A, S, S), dtype=float)

        for a in self.A:
            lam = self.lam(a)
            for n in self.S:
                if n == 0:
                    p_up = lam * self.dt
                    p_same = 1.0 - p_up
                    P[a, n, 0] = max(0.0, p_same)
                    P[a, n, 1] = max(0.0, p_up)

                elif n == self.N:
                    p_down = self.mu * self.dt
                    p_same = 1.0 - p_down
                    P[a, n, self.N] = max(0.0, p_same)
                    P[a, n, self.N - 1] = max(0.0, p_down)

                else:
                    p_up = lam * self.dt
                    p_down = self.mu * self.dt
                    p_same = 1.0 - (p_up + p_down)

                    P[a, n, n + 1] = max(0.0, p_up)
                    P[a, n, n - 1] = max(0.0, p_down)
                    P[a, n, n] = max(0.0, p_same)

                # Normalize
                row_sum = P[a, n].sum()
                if row_sum <= 0:
                    P[a, n, n] = 1.0
                else:
                    P[a, n] /= row_sum

        return P

    def _build_reward_matrix(self):
        # R(n,a) = price * lambda(a) * dt - cw * n * dt
        S = len(self.S)
        A = len(self.A)
        R = np.zeros((S, A), dtype=float)

        for a in self.A:
            lam = self.lam(a)
            price = self.prices[a]
            for n in self.S:
                revenue = price * lam * self.dt
                wait_cost = self.cw * n * self.dt
                R[n, a] = revenue - wait_cost
        return R
