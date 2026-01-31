# Week 01 — Low68 Queue Pricing MDP (Policy Iteration)

## 1. Problem Overview (Real-World Toy Application)

We study a queue pricing problem inspired by service systems such as call centers, cloud services, or shared compute resources. Customers arrive to a system and wait in a queue if the server is busy. At each decision epoch, the controller chooses a price level for the service.

Higher prices reduce demand (arrival rate) but increase revenue per arrival, while longer queues incur waiting costs. The goal is to compute an optimal stationary pricing policy that balances revenue generation and congestion costs.

This problem is modeled as a **Markov Decision Process (MDP)** and solved using **policy iteration**, as required.

---

## 2. Markov Decision Process (MDP) Formulation

We define the MDP as a 5-tuple:

\[
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
\]

### 2.1 State Space

\[
\mathcal{S} = \{0,1,2,\dots,N\}
\]

Each state \( n \in \mathcal{S} \) represents the **number of customers/jobs currently in the system** (waiting + in service).

---

### 2.2 Action Space

\[
\mathcal{A} = \{p_1, p_2, \dots, p_K\}
\]

Each action corresponds to selecting a **discrete price level** for the service at the current decision epoch.

---

### 2.3 Arrival and Service Dynamics

The arrival rate depends on the chosen price through a decreasing demand function:

\[
\lambda(a) = \lambda_0 e^{-\beta a}
\]

where:
- \( \lambda_0 > 0 \) is the baseline arrival rate
- \( \beta > 0 \) is the price sensitivity parameter

The service process is modeled with a constant service rate:

\[
\mu > 0
\]

---

### 2.4 Transition Probabilities

We use a discrete time step \( \Delta t \). For states \( 0 < n < N \):

\[
P(n+1 \mid n, a) = \lambda(a)\Delta t
\]

\[
P(n-1 \mid n, a) = \mu \Delta t
\]

\[
P(n \mid n, a) = 1 - (\lambda(a) + \mu)\Delta t
\]

Boundary conditions:
- For \( n = 0 \), departures are not possible
- For \( n = N \), arrivals are blocked and only departures occur

This results in a **birth–death queueing process** with controlled arrival rates.

---

### 2.5 Reward Function

At each time step, the reward is defined as **revenue minus waiting cost**:

\[
R(n,a) = a \cdot \lambda(a)\Delta t - c_w \cdot n \Delta t
\]

where:
- \( a \cdot \lambda(a)\Delta t \) is expected revenue
- \( c_w \cdot n \Delta t \) is the congestion (waiting) cost

---

### 2.6 Objective

The goal is to find an optimal stationary policy \( \pi^* \) that maximizes the expected discounted return:

\[
\pi^* = \arg\max_\pi \mathbb{E}
\left[
\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t))
\right]
\]

where \( \gamma \in (0,1) \) is the discount factor.

---

## 3. Solution Method: Policy Iteration

We solve the MDP using **policy iteration**, which alternates between:

### 3.1 Policy Evaluation

For a fixed policy \( \pi \), the value function satisfies the Bellman expectation equation:

\[
V^\pi(s) = R(s,\pi(s)) + \gamma \sum_{s'} P(s' \mid s, \pi(s)) V^\pi(s')
\]

This is solved iteratively until convergence.

---

### 3.2 Policy Improvement

The policy is updated greedily using:

\[
\pi_{new}(s) = \arg\max_{a \in \mathcal{A}}
\left[
R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^\pi(s')
\right]
\]

Policy evaluation and improvement are repeated until the policy stabilizes.

---

## 4. Experimental Setup

Parameters used in the experiment:

- Maximum queue length: \( N = 30 \)
- Price levels: \( \{1,2,3,4,5,6\} \)
- Baseline arrival rate: \( \lambda_0 = 8.0 \)
- Price sensitivity: \( \beta = 0.25 \)
- Service rate: \( \mu = 6.0 \)
- Waiting cost: \( c_w = 0.5 \)
- Time step: \( \Delta t = 0.05 \)
- Discount factor: \( \gamma = 0.99 \)

---

## 5. Results and Findings

Policy iteration converged in **3 iterations**, indicating fast convergence for this MDP.

### 5.1 Learned Optimal Policy

From `results/policy_table.csv`, the optimal pricing policy exhibits a **threshold structure**:

- For low congestion (\( n = 0,1,2 \)):  
  → Optimal price \( = 4 \)

- For moderate congestion (\( n = 3 \) to approximately mid-range states):  
  → Optimal price \( = 5 \)

- For high congestion (large \( n \)):  
  → Optimal price \( = 6 \)

This demonstrates that the optimal controller **increases price as queue length grows**, throttling arrivals to control congestion.

---

### 5.2 Value Function Behavior

The value function \( V(n) \), stored in `results/value_function.csv`, is monotonically decreasing with respect to queue length. Larger queues lead to lower expected long-term reward due to increased waiting costs.

---

### 5.3 Interpretation

The learned policy aligns with economic intuition:
- Low congestion → encourage demand with lower prices
- High congestion → discourage demand with higher prices

This validates both the MDP formulation and the correctness of policy iteration.

---

## 6. Code Structure

- `src/mdp_queue.py`  
  Defines the MDP, transition probabilities, and reward function.

- `src/policy_iteration.py`  
  Implements policy evaluation, policy improvement, and the policy iteration algorithm.

- `src/run_experiment.py`  
  Runs the experiment and exports results to CSV files.

---

## 7. Summary

This assignment demonstrates how a real-world inspired queue pricing problem can be modeled as an MDP and solved using policy iteration. The resulting optimal policy exhibits interpretable, congestion-aware behavior and converges efficiently.
