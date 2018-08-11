import numpy as np


def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.

    @param v the value vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return reward + gamma * np.max(action_array)


def main():
    tot_states = 12
    gamma = 0.999
    iteration = 0
    epsilon = 0.01  # stops when utility change < epsilon * (1-gamma) / (gamma)
    T = np.load("T.npy")

    # Reward vector
    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])
    # utility vector
    u1 = np.zeros(12)

    while True:
        delta = 0
        u = u1.copy()  # starts from the previous state
        iteration += 1
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1, tot_states))
            v[0, s] = 1.0  # probability of being in current state =1, rest =0 because we are already in that state
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s]))  # stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            print(u[0:4])
            print(u[4:8])
            print(u[8:12])
            print("===================================================")
            break


if __name__ == "__main__":
    main()
