### 3.1 Policy Evaluation

For a fixed policy $\pi$, the value function satisfies the Bellman expectation equation:

$$
V^\pi(s) = R\bigl(s,\pi(s)\bigr) + \gamma \sum_{s' \in \mathcal{S}} P\bigl(s' \mid s,\pi(s)\bigr)\, V^\pi(s')
$$

This is solved iteratively until convergence.
