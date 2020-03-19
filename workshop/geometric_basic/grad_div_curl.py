""" It will draw a vector field cycling the sources,
The curl and divergence of the vector field will be drawn either. """
# %%
import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# %%
# Basic settings
BoardRange = dict(x=(-100, 100), y=(-100, 100))
BoardSize = dict(x=150, y=100)

axis_values = dict()
for k in ['x', 'y']:
    axis_values[k] = np.linspace(
        BoardRange[k][0], BoardRange[k][1], BoardSize[k], endpoint=False)

X, Y = np.meshgrid(axis_values['x'], axis_values['y'])


SourceNum = 5


# %%

def vector_field(source):
    """ Compute vector field based on [source] """
    px, py = source['pos']
    s = (0, 0, source['strong'])
    U = np.zeros((BoardSize['y'], BoardSize['x']))
    V = U.copy()

    for j, x in enumerate(axis_values['x']):
        for k, y in enumerate(axis_values['y']):
            r = (x-px, y-py, 0)
            r = r / np.linalg.norm(r)
            c = np.cross(s, r)
            d = np.linalg.norm(r)
            if d < 0.5:
                U[k, j], V[k, j] = np.nan, np.nan
            else:
                U[k, j], V[k, j], _ = c / d / d

    source['U'], source['V'] = U, V


def random_source(BoardRange=BoardRange):
    """ New source on random position. """
    source = dict(pos=[np.random.uniform(BoardRange[k][0], BoardRange[k][1]) / 2
                       for k in ['x', 'y']],
                  strong=np.random.randn())
    return source


# Makeup sources
sources = [random_source() for _ in range(SourceNum)]

# Compute cycling flows
for j in range(len(sources)):
    vector_field(sources[j])

pprint(sources)

# %%
# Get [U] x-gradient and [V] y-gradient
U = sum([s['U'] for s in sources])
V = sum([s['V'] for s in sources])

# Prepare figure
R = BoardSize['y'] / BoardSize['x']
cmap = plt.get_cmap('gist_heat')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[1][0].remove()

def down(x, d=5):
    """ Down sample [x]: 2-D matrix, as [d]: down sampling rate. """
    return x[::d, ::d]


# Draw cycling flows
ax = axes[0][0]
for s in sources:
    ax.plot(s['pos'][0], s['pos'][1], 'o', label=s['strong'])
    # ax.annotate(s['strong'], (s['pos'][0], s['pos'][1]))
ax.legend(loc='best', bbox_to_anchor=(0.5, -0.7, 0.5, 0.5))
ax.quiver(down(X), down(Y), 1 * down(U), 1 * down(V), cmap=cmap, width=0.005)
ax.set_aspect(R)

# Draw curl
dyU, dxU = np.gradient(U)
dyV, dxV = np.gradient(V)
cc = dxV - dyU
ax = axes[0][1]
im = ax.imshow(cc, origin='lower')
ax.set_title('Curl')
fig.colorbar(im, ax=ax, extend='max', spacing='proportional', shrink=1.0)

# Draw div
dd = dxU + dyV
ax = axes[1][1]
im = ax.imshow(dd, origin='lower')
ax.set_title('Divergence')
fig.colorbar(im, ax=ax, extend='max', spacing='proportional', shrink=1.0)

# %%

fig.savefig('grad_div_curl.png')

# %%
