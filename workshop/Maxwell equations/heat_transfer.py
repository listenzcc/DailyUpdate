# %%
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
size = (100, 100)
init_num = 10
transfer_ratio = 0.1
cmap = plt.get_cmap('gist_heat')


def legal(j, k, size=size):
    """ Check if ([j], [k]) is legal node. """
    return all([j > -1, k > -1, j < size[0], k < size[1]])


def mk_neighbors():
    """ Make [neighbors] as dict,
    keys are nodes, values are neighbor nodes. """
    neighbors = dict()
    for j in range(size[0]):
        for k in range(size[1]):
            neighbors[(j, k)] = [(j + a, k + b) for a in [-1, 0, 1]
                                 for b in [-1, 0, 1] if legal(j + a, k + b)
                                 if not all([a == 0, b == 0])]
    return neighbors


def scatter(board, num=init_num):
    """ Scatter [num] heat points in [board] with random temperature. """
    for p in [(random.randint(0, size[0] - 1), random.randint(0, size[1] - 1))
              for _ in range(init_num)]:
        board[p] = random.random() * 100
    return board


def transfer(board, neighbors_table, ratio=transfer_ratio):
    """ Transfer heat based on [ratio],
    [neighbors_table] is used to quickly access neighbors for each node. """
    for node in neighbors_table:
        # Get temperature of node
        T = board[node]
        # Get neighbors of node
        neighbors = neighbors_table[node]
        H_sum = 0
        for n in neighbors:
            if T > board[n]:
                D = board[node] - board[n]
                H = D * ratio
                H_sum += H
                board[n] += H
        board[node] -= H_sum
    return board


# %%
# Init board
board = np.zeros(size)
# Make neighbors dict
neighbors = mk_neighbors()
# Set up heat points
board = scatter(board)
# Transfer
boards = [board]
for _ in tqdm.trange(100):
    boards.append(transfer(boards[-1].copy(), neighbors))

# %%

# def plot(board):
#     print('.')
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))

#     # Draw Scalar Field
#     ax = axes[0][0]
#     im = ax.imshow(board, cmap=cmap, origin='lower')
#     fig.colorbar(im, ax=ax, extend='max', spacing='proportional', shrink=1.0)

#     # Draw Contourf
#     ax = axes[0][1]
#     im = ax.contourf(board, cmap=cmap)
#     ax.set_aspect(1)
#     fig.colorbar(im, ax=ax, extend='both', spacing='proportional', shrink=1.0)

#     # Draw Gradient
#     Y, X = np.mgrid[0:board.shape[0]:1, 0:board.shape[1]:1]
#     U, V = np.gradient(board)
#     M = np.hypot(U, V)
#     strong = np.sqrt(U**2 + V**2) * 50
#     ax = axes[1][0]

#     def down(x, d=3):
#         return x[::d, ::d]

#     ax.quiver(down(X), down(Y), down(U), down(V), down(M))
#     fig.tight_layout()

# plt.style.use('default')
# # plot(board)
# # for j in range(0, len(boards), 10):
# #     plot(boards[j])
# plot(boards[-1])
# plt.show()

# %%

fig, axes = plt.subplots(1, 2, figsize=(8, 4))


def update(board):
    print('.', end='')
    # Draw Scalar Field
    ax = axes[0]
    ax.clear()
    im = ax.imshow(board, cmap=cmap, origin='lower')
    # fig.colorbar(im,
    #              ax=ax,
    #              extend='max',
    #              spacing='proportional',
    #              shrink=1.0)

    # Draw Gradient
    Y, X = np.mgrid[0:board.shape[0]:1, 0:board.shape[1]:1]
    U, V = np.gradient(board)
    M = np.hypot(U, V)

    ax = axes[1]
    ax.clear()

    def down(x, d=2):
        return x[::d, ::d]

    ax.quiver(down(X), down(Y), down(U), down(V), down(M), cmap=cmap, width=0.005)
    ax.set_facecolor((0, 0, 0))
    fig.tight_layout()


anim = FuncAnimation(fig, update, frames=boards, interval=200)
anim.save('a.gif')

# %%
