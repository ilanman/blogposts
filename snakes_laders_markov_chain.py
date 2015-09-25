import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rcParams
import copy
rcParams.update({'figure.autolayout': True})


def initialize_matrix(M, n):
    """
    This function takes in an empty matrix and returns an (n+1)x(n+1) matrix that is 1/6 along the diagonals, 6 wide
    incrementing by 1 index, for every row
    At the end, to gather up the remaining probabilities, it adds them cumulatively, i.e. in square 98, you can reach square
    99 or 100 with more than 1/6 probability because you can overshoot it
    """

    for i in range(n):
        M[i, (i+1):(i+7)] = float(1)/6

    M[:, 100] = np.cumsum(M[:, 100])
    M = M[:, :101]

    return M


def create_matrix(obstacles, M):
    """
    Depending on if the obstacle is a snake or ladder, this will apply the correct logic by zeroing out the square after the ladder or snake (since
    according to the rules, we won't land on them in this turn) and instead populating the squares that we end up in
    """

    for i in range(len(obstacles['enter'])):
        new_squares = copy.copy(M[:, obstacles['enter'][i]])
        ind = np.where(new_squares > 0)
        M[ind, obstacles['enter'][i]] = 0
        M[ind, obstacles['exit'][i]] = M[ind, obstacles['exit'][i]] + new_squares[ind]
    return M


def multiplyTransitionMatrix(M, h):
    """
    Multiplies the transition matrix by itself h times
    """

    M_h = M
    if h > 1:
        for k in range(1, h):
            M_h = np.dot(M_h, M)
    return M_h


def pmf(initial, x, M):
    """ Returns the probability mass function at each roll """

    return np.dot(initial, multiplyTransitionMatrix(M, x))


def animate(x):
    """
    Create a new heatmap for every x, which is the number of frames I want to glued together for the gif
    """

    test = prob[x, 1:].reshape(10, 10)
    plt.pcolor(test,cmap=plt.cm.Blues)
    plt.title("Snakes and Ladders Board\nSnakes and Ladders")

if __name__ == '__main__':

    n = 100
    M = np.zeros(101*107).reshape(101, 107)
    M = initialize_matrix(M, n)

    snakes = {
        "enter": np.array([98,95,93,87,62,64,56,49,48,16]),
        "exit": np.array([78,75,73,24,19,60,53,11,26,6])
    }

    ladders = {
        "enter": np.array([1,4,9,21,28,36,51,71,80]),
        "exit": np.array([38,14,31,42,84,44,67,91,99])
    }

    initial = np.append(1, np.repeat(0, n))
    M = create_matrix(ladders, M)
    M = create_matrix(snakes, M)

    prob = np.zeros(100*101).reshape(100, 101)

    for i in range(100):
        prob[i,:] = pmf(initial, i, M)

    fig = plt.figure(figsize=(5, 4))

    anim = animation.FuncAnimation(fig, animate, frames=35, interval=5, blit=False)
    anim.save('heatmap_both.gif', writer='imagemagick', fps=8)
    plt.show()
