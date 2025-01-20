# Install gymnasium
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create a Cliff Walk environment using Gymnasium
env = gym.make("CliffWalking-v0", is_slippery=True, render_mode="rgb_array")#'ansi')#

# Reset the environment and display it
env.reset()

#print (env.render())  # textual output if mode == 'ansi' or 'human'
env.render() # gui/rgb output
nS = env.observation_space.n    # number of states -- 48
nA = env.action_space.n         # number of actions -- four directions; 0:left, 1:down, 2:right, 3:up
print ("{}, {}".format(nS, nA))
# Probatilies from State 0 (top-left corner).
#env.P[0]

# Access the underlying environment using env.unwrapped
env_unwrapped = env.unwrapped

# Now you can access the transition probabilities
env_unwrapped.P[0]
# Probatilies from the start state (36)
env_unwrapped.P[36]
# Transition probability for trying to go left (action 3) from the start state (36)
env_unwrapped.P[36][3]
def generate_random_policy(num_actions, num_states, seed=None):
    """
    A policy is a 1D array of length # of states, where each element is a
    number between 0 (inclusive) and # of actions (exclusive) randomly chosen.
    If a specific seed is passed, the same numbers are genereated, while
    if the seed is None, the numbers are unpredictable every time.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=num_actions, size=num_states)
def run(env, pi, printinfo = False):
    """
    Run the policy on the environment and returns the cumulative reward.
    :param: env: The environment
    :param: pi: A given policy, represented as a 1D array of length # of states.
    :return: Cumulative reward
    """
    s = env.reset()
    if printinfo == True:
      print (f'\n* Episode starting from state {s[0]}') # ensure starting from state 36

    s = s[0]      # extract the state value/index from the tuple
    done = False  # this becomes true when agent reaches the goal state (47)
    sum_r = 0
    while not done:
        a = pi[s]   # action for the state s, according to the policy
        s, r, done, info, p = env.step(a)  # take the action
        sum_r += r  # accumulate reward

        ### uncomment below to see the information for each step
        #print (f'next_state={s}, reward={r}, done={done}, info={info}, p={p}')

        # prints info in text if render_mode is 'ansi' or no output if 'human',
        # or graphical output if 'rgb_array' AND if the code is run from command line.
        env.render()
    return sum_r
policy = generate_random_policy(nA, nS, 17) # third parameter is the random seed
print ("*** Policy ***\n{}".format(policy.reshape((4, 12))))

# Do just one run
result = run(env, policy)
# Print the total rewards/return
print (f' ==> Total return: {result}')

def generate_random(num_actions, num_states, seed=None):
  rng = np.random.default_rng(seed)
  return rng.integers(low=0, high=num_actions, size=num_states)
def run(env, pi):
    s = env.reset()
    s = s[0]  # Extract the state index
    done = False
    total_reward = 0
    total_steps = 0
    near_falls = 0
    
    while not done:
        a = pi[s]
        s, r, done, info, _ = env.step(a)
        total_reward += r
        total_steps += 1
        if s in range(37, 47):  # Near-cliff states
            near_falls += 1
    
    return total_reward, total_steps, near_falls
def evaluate_policy(env, seed, num_runs=100):
    policy = generate_random_policy(env.action_space.n, env.observation_space.n, seed)
    rewards = []
    steps = []
    near_falls = []
    for _ in range(num_runs):
      total_reward, total_steps, near_fall_count = run(env, policy)
      rewards.append(total_reward)
      steps.append(total_steps)
      near_falls.append(near_fall_count)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps)
    std_steps = np.std(steps)
    
    return policy, mean_reward, std_reward, mean_steps, std_steps, rewards, steps, near_falls
def generate_histograms(steps, near_falls, rewards, seed):
    # Calculate ratio of near-falls
    near_fall_ratios = [nf / step if step > 0 else 0 for nf, step in zip(near_falls, steps)]
    
    plt.figure(figsize=(10, 7))

    # Histogram for steps
    plt.subplot(1, 3, 1)
    plt.hist(steps, bins=10, alpha=0.7, color="orange", edgecolor="black")
    plt.title("Steps")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")

    # Histogram for ratio of near-falls
    plt.subplot(1, 3, 2)
    plt.hist(near_fall_ratios, bins=10, alpha=0.7, color="cyan", edgecolor="black")
    plt.title("Ratio of Near-falls")
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")

    # Histogram for rewards
    plt.subplot(1, 3, 3)
    plt.hist(rewards, bins=10, alpha=0.7, color="green", edgecolor="black")
    plt.title("Rewards")
    plt.xlabel("Rewards")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.suptitle(f"Histogram Analysis (Seed: {seed})", y=1.05, fontsize=16)
    plt.show()

# Example usage
# Simulate some data
num_runs = 100
seed = 42  # Example seed
steps = np.random.randint(500, 10000, size=num_runs)  # Simulated step data
near_falls = np.random.randint(0, 50, size=num_runs)  # Simulated near-falls
rewards = np.random.randint(-50000, 0, size=num_runs)  # Simulated rewards

# Generate histograms
generate_histograms(steps, near_falls, rewards, seed)