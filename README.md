# neu-self-improve-ai

# Week 01 — Low68 (Queue Pricing) using Policy Iteration

## Real-world toy application
We model a service system (e.g., call center / cloud service) with a queue. At each decision epoch, we choose a posted price. Higher price reduces arrival rate (demand) but increases revenue per arrival. Waiting in the system incurs a cost. Goal: compute an optimal stationary pricing policy via policy iteration.

## MDP Formulation (formal)
We define the MDP as \( (\mathcal{S}, \mathcal{A}, P, R, \gamma) \).

**States**
\[
\mathcal{S} = \{0,1,\dots,N\}
\]
where state \(n\) is the number of customers/jobs in the system.

**Actions**
\[
\mathcal{A} = \{p_1, p_2, \dots, p_K\}
\]
(discrete price levels)

**Arrival rate (price-dependent)**
\[
\lambda(a)=\lambda_0 e^{-\beta a}
\]

**Service rate**
\[
\mu
\]

**Transitions (discrete time step \(\Delta t\))**
For \(0 < n < N\):
\[
P(n+1\mid n,a)=\lambda(a)\Delta t,\quad
P(n-1\mid n,a)=\mu\Delta t,\quad
P(n\mid n,a)=1-(\lambda(a)+\mu)\Delta t
\]
Boundary cases:
- \(n=0\): no departures
- \(n=N\): arrivals blocked, only departures

**Reward**
\[
R(n,a)=a\lambda(a)\Delta t - c_w n \Delta t
\]
(revenue minus waiting cost)

**Objective**
\[
\pi^*=\arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R(s_t,\pi(s_t))\right]
\]

## Method: Policy Iteration
We use classic policy iteration:
1) Policy evaluation (iterative Bellman evaluation)
2) Policy improvement (greedy w.r.t. Q)
Repeat until the policy is stable.

## Code
- `src/mdp_queue.py` — builds transition tensor \(P\) and reward matrix \(R\)
- `src/policy_iteration.py` — policy evaluation + policy improvement + policy iteration loop
- `src/run_experiment.py` — runs experiment and writes CSV outputs

## Parameters used
- \(N=30\)
- Prices \( \{1,2,3,4,5,6\}\)
- \(\lambda_0=8.0,\ \beta=0.25\)
- \(\mu=6.0\)
- \(c_w=0.5\)
- \(\Delta t=0.05\)
- \(\gamma=0.99\)

## Results (experimental)
Policy iteration converged in **3 iterations**.

Outputs:
- `results/policy_table.csv`
- `results/value_function.csv`
- `results/policy_iteration_history.csv`

Qualitative behavior: the learned policy sets lower prices at low congestion and increases the price as queue length grows (throttling demand to reduce waiting cost).
