import numpy as np


def return_state_utility(v, T, u, reward, gamma):
    """
    Return the utility of a state
    @param: v the state vector, (1, 12)
    @param: T transition matrix, (12, 12, 4) = 12 current state, 12 next states, 4 possible actions. In this matrix most of elements will be zero, because from 1 state we can only move to the neighboring state
    @param: u: utility vector (1, 12)
    @param: reward: reward for that state?
    @param: gamma: discount factor
    @return: the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(4):
        """
        np.dot(v, T[:, :, action]) is the probabilities of being in the next states
        np.multiply(u, dot) is the utilities of each next states
        np.sum(.) is the expected utilities of all next states
        """
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return reward + gamma * np.max(action_array)


def main():
    # Starting state vector
    # The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0]])

    # Transition matrix loaded from file
    # (It is too big to write here)
    T = np.load("T.npy")

    # Utility vector
    u = np.array([[0.812, 0.868, 0.918, 1.0,
                   0.762, 0.0, 0.660, -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    # Defining the reward for state (1,1)
    reward = -0.04

    gamma = 1.0
    # Use the Bellman equation to find the utility of state (1,1)
    utility_11 = return_state_utility(v, T, u, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))


if __name__ == "__main__":
    main()
