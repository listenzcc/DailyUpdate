# %%
import random
import numpy as np
import matplotlib.pyplot as plt

size = (100, 100)
init_num = 10
# %%
board = np.zeros(size)

for p in [(random.randint(0, size[0]), random.randint(0, size[1])) for _ in range(init_num)]:
    board[p] = random.random() * 100

def legal(j, k, size=size):
    return all([j > -1, k > -1, j < size[0], k < size[1]])

def mk_neighbors():
    neighbors = dict()
    for j in range(size[0]):
        for k in range(size[1]):
            neighbors[(j, k)] = [(j+a, k+b)
            for a in [-1, 0, 1]
            for b in [-1, 0, 1]
            if legal(j+a, k+b)
            if not all([a == 0, b == 0])]
    return neighbors

neighbors = mk_neighbors()

# %%
plt.style.use('default')
plt.imshow(board, cmap=plt.get_cmap('hot'))
plt.show()

plt.contourf(board)
plt.gca().set_aspect(1)
plt.show()
# %%
