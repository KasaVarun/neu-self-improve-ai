import numpy as np
import pandas as pd

from mdp_queue import QueuePricingMDP
from policy_iteration import policy_iteration

def main():
    # ---- Toy-realistic params ----
    N = 30
    prices = [1, 2, 3, 4, 5, 6]
    lambda0 = 8.0
    beta = 0.25
    mu = 6.0
    cw = 0.5
    dt = 0.05
    gamma = 0.99

    mdp = QueuePricingMDP(N, prices, lambda0, beta, mu, cw, dt, gamma)

    policy, V, hist = policy_iteration(mdp.P, mdp.R, mdp.gamma)

    # Save results
    states = np.arange(N + 1)
    chosen_prices = [mdp.prices[a] for a in policy]

    policy_df = pd.DataFrame({"state_n": states, "action_price": chosen_prices})
    value_df = pd.DataFrame({"state_n": states, "V": V})
    hist_df = pd.DataFrame(hist)

    policy_df.to_csv("../results/policy_table.csv", index=False)
    value_df.to_csv("../results/value_function.csv", index=False)
    hist_df.to_csv("../results/policy_iteration_history.csv", index=False)

    # Console summary
    print("Policy Iteration finished.")
    print(f"Iterations: {len(hist)}")
    print("First 10 states policy (n -> price):")
    print(policy_df.head(10).to_string(index=False))
    print("\nUnique prices used:", sorted(set(chosen_prices)))

if __name__ == "__main__":
    main()
