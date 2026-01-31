import numpy as np

def policy_evaluation(P, R, policy, gamma, tol=1e-10, max_iter=100000):
    """
    Iterative policy evaluation:
      V(s) = R(s, pi(s)) + gamma * sum_s' P_pi(s,s') V(s')
    P: (A,S,S), R: (S,A), policy: (S,)
    """
    S = R.shape[0]
    V = np.zeros(S, dtype=float)

    for _ in range(max_iter):
        V_new = np.empty_like(V)
        for s in range(S):
            a = policy[s]
            V_new[s] = R[s, a] + gamma * np.dot(P[a, s], V)

        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new

    return V  # if max_iter hit


def policy_improvement(P, R, V, gamma):
    """
    Greedy improvement:
      pi_new(s) = argmax_a [ R(s,a) + gamma * sum_s' P(a,s,s') V(s') ]
    """
    S, A = R.shape
    pi_new = np.zeros(S, dtype=int)

    for s in range(S):
        q_sa = np.zeros(A, dtype=float)
        for a in range(A):
            q_sa[a] = R[s, a] + gamma * np.dot(P[a, s], V)
        pi_new[s] = int(np.argmax(q_sa))
    return pi_new


def policy_iteration(P, R, gamma, tol=1e-10, max_outer=1000):
    """
    Classic policy iteration:
      initialize policy
      repeat eval -> improve until stable
    """
    S, A = R.shape
    policy = np.zeros(S, dtype=int)  # start with lowest price action index 0
    history = []

    for it in range(max_outer):
        V = policy_evaluation(P, R, policy, gamma, tol=tol)
        new_policy = policy_improvement(P, R, V, gamma)

        changed = np.sum(new_policy != policy)
        history.append({"iter": it + 1, "changed_states": int(changed)})

        if changed == 0:
            return policy, V, history

        policy = new_policy

    return policy, V, history
