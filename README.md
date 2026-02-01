## 1. Problem Description and Motivation

We consider a pricing and congestion control problem in a service system such as a call center, cloud computing service, or shared processing facility. Customers arrive over time requesting service. If the server is busy, arriving customers wait in a queue until service becomes available.

At each decision epoch, the system operator selects a price for the service. The selected price directly influences the arrival rate of customers: lower prices encourage demand, while higher prices discourage arrivals. At the same time, congestion in the system leads to increased waiting times, which incur operational and customer dissatisfaction costs.

The operator faces a trade-off between:
- generating revenue by attracting customers, and
- controlling congestion by limiting arrivals when the system is heavily loaded.

The goal is to determine an optimal pricing strategy that adapts to the current congestion level of the system.

---

### Modeling as a Markov Decision Process

This decision-making problem is naturally modeled as a **Markov Decision Process (MDP)** because:
- the system state (queue length) evolves stochastically over time,
- the controller makes sequential decisions (price selection),
- the future evolution of the system depends only on the current state and action,
- and the objective is to maximize long-term expected reward.

To keep the analysis tractable and aligned with the assignment requirements, we study this problem using a **real-world inspired but synthetic queueing model**, allowing full control over system dynamics while preserving realistic behavior.

---

### Scope and Assumptions

The following assumptions are made:
- Arrivals follow a price-dependent stochastic process.
- Service times are exponentially distributed with a fixed service rate.
- The system has a finite capacity to prevent unbounded growth of the queue.
- Decisions are made at discrete time intervals.
- The policy is restricted to be stationary and deterministic.

These assumptions result in a controlled birth–death process suitable for analysis via policy iteration.

---

### Objective of the Assignment

The objectives of this assignment are to:
1. Formulate the queue pricing problem formally as an MDP.
2. Apply policy iteration to compute an optimal pricing policy.
3. Analyze the structure and behavior of the learned policy.
4. Demonstrate how congestion-aware pricing emerges from optimal control.

This formulation and solution approach directly correspond to the **Low68 queueing example** 

## 2. Markov Decision Process (MDP) Formulation

We model the queue pricing problem as a Markov Decision Process (MDP), defined by the 5-tuple:

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

where each component is formally defined below.

---

### 2.1 State Space

The state represents the level of congestion in the system.

$$
\mathcal{S} = \{0,1,2,\dots,N\}
$$

A state \( s_t = n \) denotes that there are \( n \) customers (waiting or in service) in the system at decision epoch \( t \).

---

### 2.2 Action Space

The action corresponds to the price charged for the service.

$$
\mathcal{A} = \{p_1, p_2, \dots, p_K\}
$$

Each action selects a discrete price level. Higher prices reduce demand but increase revenue per arrival.

---

### 2.3 Arrival and Service Dynamics

The arrival rate depends on the selected price through a decreasing demand function:

$$
\lambda(a) = \lambda_0 e^{-\beta a}
$$

where \( \lambda_0 \) is the baseline arrival rate and \( \beta \) captures price sensitivity.

Service times are modeled with a constant service rate:

$$
\mu > 0
$$

---

### 2.4 Transition Probabilities

We use a discrete time step \( \Delta t \).  
For interior states \( 0 < n < N \), the system follows a birth–death process:

$$
P(n+1 \mid n, a) = \lambda(a)\Delta t
$$

$$
P(n-1 \mid n, a) = \mu \Delta t
$$

$$
P(n \mid n, a) = 1 - (\lambda(a) + \mu)\Delta t
$$

Boundary conditions ensure feasibility:

For an empty system \( n = 0 \):

$$
P(1 \mid 0, a) = \lambda(a)\Delta t
$$

$$
P(0 \mid 0, a) = 1 - \lambda(a)\Delta t
$$

For a full system \( n = N \), arrivals are blocked:

$$
P(N-1 \mid N, a) = \mu \Delta t
$$

$$
P(N \mid N, a) = 1 - \mu \Delta t
$$

---

### 2.5 Reward Function

The reward captures revenue earned from arrivals minus congestion cost:

$$
R(n,a) = a \cdot \lambda(a)\Delta t - c_w \cdot n \Delta t
$$

where:
- the first term is expected revenue,
- the second term penalizes waiting and congestion.

---

### 2.6 Objective

The objective is to find an optimal stationary policy \( \pi^* \) that maximizes expected discounted reward:

$$
\pi^* = \arg\max_{\pi}
\mathbb{E}
\left[
\sum_{t=0}^{\infty}
\gamma^t
R\bigl(s_t,\pi(s_t)\bigr)
\right]
$$

where \( \gamma \in (0,1) \) is the discount factor.

---

## 3. Solution Method: Policy Iteration

We solve the MDP using **policy iteration**, which alternates between policy evaluation and policy improvement.

---

### 3.1 Policy Evaluation

For a fixed policy \( \pi \), the value function satisfies the Bellman expectation equation:

$$
V^\pi(s) =
R\bigl(s,\pi(s)\bigr)
+
\gamma
\sum_{s' \in \mathcal{S}}
P\bigl(s' \mid s,\pi(s)\bigr)
V^\pi(s')
$$

This equation is solved iteratively until convergence.

---

### 3.2 Policy Improvement

The policy is updated greedily using the current value function:

$$
\pi_{\text{new}}(s)
=
\arg\max_{a \in \mathcal{A}}
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)
V^\pi(s')
\right]
$$

Policy evaluation and improvement are repeated until the policy stabilizes.

---

## 4. Results and Findings

Policy iteration converged after **3 iterations**.

The learned optimal pricing policy exhibits a threshold structure:

$$
\pi^*(n) =
\begin{cases}
4, & n = 0,1,2 \\
5, & 3 \le n < n_1 \\
6, & n \ge n_1
\end{cases}
$$

This indicates that the controller increases prices as congestion grows, throttling demand to reduce waiting costs.

---

### Value Function Property

The value function decreases as queue length increases:

$$
V(n+1) \le V(n)
$$

This reflects increasing congestion penalties at higher system loads.

---

### Discount Factor

$$
\gamma = 0.99
$$

### Example Execution Output

The following output is produced when running the experiment:

```text
Policy Iteration finished.
Iterations: 3
First 10 states policy (n -> price):
 state_n  action_price
       0           4.0
       1           4.0
       2           4.0
       3           5.0
       4           5.0
       5           5.0
       6           5.0
       7           5.0
       8           5.0
       9           5.0

Unique prices used: [4.0, 5.0, 6.0]

Saved files:
 - week_01/results/policy_table.csv
 - week_01/results/value_function.csv
 - week_01/results/policy_iteration_history.csv

