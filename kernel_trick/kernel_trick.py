import numpy as np
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from matplotlib import animation
import matplotlib.pyplot as plt

def generate_data():

    def kernel(x1, x2):
        return np.array([x1, x2, 2*x1**2 + 2*x2**2])

    X, Y = make_circles(500, noise=0.12, factor=0.01)

    A = X[np.where(Y == 0)]
    B = X[np.where(Y == 1)]

    X0_orig = A[:, 0]
    Y0_orig = A[:, 1]

    X1_orig = B[:, 0]
    Y1_orig = B[:, 1]

    A = np.array([kernel(x,y) for x,y in zip(np.ravel(X0_orig), np.ravel(Y0_orig))])

    X0 = A[:, 0]
    Y0 = A[:, 1]
    Z0 = A[:, 2]

    A = np.array([kernel(x,y) for x,y in zip(np.ravel(X1_orig), np.ravel(Y1_orig))])
    X1 = A[:, 0]
    Y1 = A[:, 1]
    Z1 = A[:, 2]

    return X0, X1, Y0, Y1, Z0, Z1


def run_SVM(X0, X1, x_vals):

    np.random.seed([12345])
    y = np.concatenate([np.repeat(0, len(X0)), np.repeat(1, len(X1))])
    X = np.vstack([X0, X1])
    X = np.vstack([x_vals,X.flatten()]).T

    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X,y)

    return clf, X, y

def make_multiple_lines(xx,yy, x_vals, y, X):

    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    plt.plot(1*xx, yy+2,'r')
    plt.plot(2*xx, yy+1,'b')
    plt.plot(xx, yy-1,'g')
    plt.plot(3*xx, yy-1,'y')
    plt.plot(4*xx, yy-2,'purple')
    plt.xlim(-10,40)
    plt.ylim(-5,15)
    plt.scatter(x_vals, X[:,1], c=[i for i in y])
    plt.grid(b=True, which='major', linestyle='-',alpha=0.1,color='black')
    plt.title("Which one to select?",size=16)
    plt.show()


def make_SVM_boundary(clf, xx, yy, y, X):

    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    plt.plot(xx, yy,'black',linewidth=1.5)
    plt.plot(xx, yy_down,'b--',1.5)
    plt.plot(xx, yy_up,'b--',1.5)
    plt.xlim(-10,40)
    plt.ylim(-5,15)
    plt.scatter(x_vals, X[:,1], c=[i for i in y])
    plt.legend(('Boundary','Max Margin'))
    plt.title("SVM with linear kernel",size=16)
    plt.grid(b=True, which='major', linestyle='-',alpha=0.1,color='black')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
    plt.show()


def generate_circles():
 
    X1, Y1 = make_circles(n_samples=500, noise=0.07, factor=0.4)
    plt.figure(figsize=(5, 5))
    plt.scatter(X1[:, 0], X1[:, 1], c=Y1)
    plt.grid(b=True, which='major', linestyle='-',alpha=0.1,color='black')
    plt.title("Can this be solved linearly?", size=16)
    plt.show()


def plot_2D_boundary(X0,X1,Y0,Y1):

    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    ax.scatter(X0, Y0, c='r')
    ax.scatter(X1, Y1, c='b')
    plt.xlim(-1.75,1.75)
    plt.ylim(-1.75,1.75)
    ax.add_patch(plt.Circle((0,0), radius=sqrt(0.3),fill=True, linewidth=3, facecolor='#808000', alpha=0.2,edgecolor='black'))
    plt.title("2D view of boundary", size=16)
    ax.grid(b=True, which='major', linestyle='-',alpha=0.1,color='black')
    plt.show()


def make_bowl(X0,X1,Y0,Y1):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=5, azim=45)
    ax.scatter(X0, Y0, Z0, c='r')
    ax.scatter(X1, Y1, Z1, c='b')

    x = np.arange(-1.2, 1.2, 0.1)
    y = np.arange(-1.2, 1.2, 0.1)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    Z[:,:] = 0.6

    ax.plot_surface(X, Y, Z, color='yellow', alpha=0.5)
    plt.title("3D view of boundary plane", size=16)


def animate(i):
	ax.view_init(elev=i, azim=30)

if __name__ == '__main__':
    
    # 3d visuals
    # generate_data() uses kernel
    X0, X1, Y0, Y1, Z0, Z1 = generate_data()
    plot_2D_boundary(X0,X1,Y0,Y1)
    make_bowl(X0,X1,Y0,Y1)
    generate_circles()
    
    # data for SVM
    A = np.random.normal(0, 1, 50)
    B = np.random.normal(10, 1, 50)
    x_vals = np.linspace(-5, 20, 100)

    # create classifier
    clf, X, y = run_SVM(A, B, x_vals)
    w = clf.coef_[0]
    a = -w[0] / w[1]

    # SVM boundary
    xx = np.linspace(-10,40)
    yy = a * xx - clf.intercept_[0] / w[1]

    # plots that rely on SVM
    make_multiple_lines(xx,yy, x_vals, y, X)
    make_SVM_boundary(clf, xx, yy, y, X)

#	anim = animation.FuncAnimation(fig, animate, init_func=make_bowl, frames=90, interval=20, blit=True)
#	anim.save('boundary.gif', fps=30)
