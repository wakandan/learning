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
    for action in range(0, 4):  # for each action
        """
        T[:, :, action] = probabilities from all current states to move to all next states after taking the action
        np.dot(v, T[:, :, action]) = probabilities of all next state of one current state
        np.multiply(u, np.dot) = probabilities * utilities of next states
        np.sum() = expected utilities of next states
        """
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return reward + gamma * np.max(action_array)


def return_policy_evaluation(p, u, r, T, gamma):
    """
    Return policy evaluation
    :param p: policy of state -> action (1, 12)
    :param u: utility matrix -> (1, 12)
    :param r: reward vector (1, 12)
    :param T: transition model (12, 12, 4)
    :param gamma: discount 1
    :return:
    """
    for s in range(12):
        if not np.isnan(p[s]):  # if the action for the current state is defined
            v = np.zeros((1, 12))
            v[0, s] = 1.0
            """
            T[:, :, p[s]] = possible next states from all states if taken action dictated by the policy
            np.dot(v, T[:, :, p[s]]) = return a row of T, which is probablities of moving from this state to next state
            np.multiply(u, np.dot) = utility vector of next states by their probabilities
            np.sum() = expected utility of all next states
            """
            action = int(p[s])
            # u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
            u[s] = np.linalg.solve(np.identity(12) - gamma * T[:, :, int(p[s])], r)[s]
            # u[s] = np.dot(np.linalg.inv(np.identity(12) - gamma * T[:, :, int(p[s])]), r)[s]

    return u


def return_expected_action(u, T, v):
    """
    Returns an action that maximizes the expected utility of next states from a given state
    Why there's no reward here?
    :param u: current utility vector
    :param T: state transition
    :param v: current state vector
    :return:
    """
    u1 = np.zeros(4)
    for a in range(4):
        u1[a] = np.sum(np.multiply(u, np.dot(v, T[:, :, a])))
    return np.argmax(u1)


def print_policy(p, shape):
    """Printing utility.

    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    # obstacles
    """
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if (p[counter] == -1):
                policy_string += " *  "
            elif (p[counter] == 0):
                policy_string += " ^  "
            elif (p[counter] == 1):
                policy_string += " <  "
            elif (p[counter] == 2):
                policy_string += " v  "
            elif (p[counter] == 3):
                policy_string += " >  "
            elif (np.isnan(p[counter])):
                policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def main():
    gamma = 0.999
    epsilon = 0.0001
    iteration = 0
    T = np.load('T.npy')
    # generate the first policy randomly
    # NAN = Nothing, -1 = Terminal, 0 = Up, 1 = Left, 2 = Down, 3 = Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NAN  # wall
    p[3] = p[7] = -1
    # initialization of utility vector
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])
    # reward vector
    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        u1 = u.copy()
        # update policy evaluation from the current u
        u = return_policy_evaluation(p, u, r, T, gamma)
        # check if there's any update from the last u
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma: break
        # for each state find the action that maximize its expect utility
        # and update the policy to return the action when being in that state
        for i in range(12):
            if not np.isnan(p[i]) and not p[i] == -1:
                v = np.zeros((1, 12))
                v[0, i] = 1.0
                a = return_expected_action(u, T, v)
                if a != p[i]: p[i] = a
        print_policy(p, shape=(3, 4))

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
    print_policy(p, shape=(3, 4))
    print("===================================================")


if __name__ == "__main__":
    main()
