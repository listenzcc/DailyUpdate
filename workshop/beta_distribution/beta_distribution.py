# %%
# Imports
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# %%
# Profiles
x_values = np.linspace(0, 1, 1000)


def beta_pdf(a, b, x=x_values):
    """ Calculate the PDF of Beta distribution with parameters [a] and [b] on [x_values] """
    return stats.beta.pdf(x, a, b)


def linspace(start, stop):
    """ Make linspace between start and stop with intervel as 1 """
    assert (stop > start)
    return np.linspace(start, stop, num=stop - start, endpoint=False)


# %%
# Setup figure
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
print()

# %%
# Draw Beta distribution with different parameters alpha and beta.
# alpha as a, beta as b.
para_beta = [(2, 8), (20, 80), (200, 800), (5, 5), (50, 50), (80, 20)]

ax = axes[0][0]

for a, b in para_beta:
    label = f'Alpha={a}, Beta={b}'
    pdf = beta_pdf(a, b)
    curve = ax.plot(x_values, pdf, label=label)[0]

ax.set_title('Beta distribution')
ax.legend(loc='auto')

# %%
# Draw mean and variance of Beta distribution with different parameters.
start, stop = 3, 30
a_values = linspace(start, stop)
b_values = linspace(start, stop)

# mean, var, skew, kurt
mean = np.zeros((stop, stop))
var = np.zeros((stop, stop))

for a in a_values:
    for b in b_values:
        m, v, s, k = stats.beta.stats(a, b, moments='mvsk')
        mean[int(a)][int(b)] = m
        var[int(a)][int(b)] = v


def fill_ax(mat, ax, title='title'):
    im = ax.imshow(mat)
    ax.set_xlim([start, stop-1])
    ax.invert_yaxis()
    ax.set_ylim([start, stop-1])
    ax.set_title(title)
    fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax)
    return im, ax


im_mean, ax_mean = fill_ax(mean, axes[0][1], title='mean')
im_var, ax_var = fill_ax(var, axes[1][1], title='var')

# %%
# Show figure
fig.tight_layout()
fig.savefig('beta_distribution.png')
fig

# %%
