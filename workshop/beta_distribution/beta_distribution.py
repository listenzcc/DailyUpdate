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
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
gs = axes[0][0].get_gridspec()
for ax in axes[0]:
    ax.remove()
axbig = fig.add_subplot(gs[0, :])
print()

# %%
# Draw Beta distribution with different parameters alpha and beta.
# alpha as a, beta as b.
para_beta = [(2, 8), (20, 80), (200, 800), (5, 5), (50, 50), (80, 20)]

ax = axbig

for a, b in para_beta:
    label = f'Alpha={a}, Beta={b}'
    pdf = beta_pdf(a, b)
    curve = ax.plot(x_values, pdf, label=label)[0]

ax.set_title('Beta distribution')
ax.legend(loc='best')

# %%
# Draw mean and variance of Beta distribution with different parameters.
start, stop = 3, 30
a_values = linspace(start, stop)
b_values = linspace(start, stop)
x_grid, y_grid = np.meshgrid(a_values, b_values)

# mean, var, skew, kurt
mean = np.zeros((stop, stop))
var = np.zeros((stop, stop))

for a in a_values:
    for b in b_values:
        m, v, s, k = stats.beta.stats(a, b, moments='mvsk')
        mean[int(a)][int(b)] = m
        var[int(a)][int(b)] = v


def imshow_ax(ax, mat, title='undefined'):
    """ Fill [ax] with [mat] as [title] using imshow, [start] and [stop] are pre-defined parameters to control display range. """
    # Imshow
    im = ax.imshow(mat)
    # Fine x-axis
    ax.set_xlim([start, stop - 1])
    ax.set_xlabel('Alpha')
    # Fine y-axis
    ax.invert_yaxis()
    ax.set_ylim([start, stop - 1])
    ax.set_ylabel('Beta')
    # Add title
    ax.set_title(title)
    # Add colorbar
    fig.colorbar(im, ax=ax, extend='both', spacing='proportional', shrink=1.0)
    return im, ax


def contourf_ax(ax, mat, title='undefined'):
    """ Fill [ax] with [mat] as [title] using contourf, [x_grid] and [y_grid] are pre-defined parameters to control display range. """
    # Contourf
    im = ax.contourf(
        x_grid,
        y_grid,
        mat[x_grid.astype(int), y_grid.astype(int)].transpose(),
        10,
        alpha=0.6)
    ax.set_aspect(1)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    # Add colorbar
    fig.colorbar(im, ax=ax, extend='both', spacing='proportional', shrink=1.0)
    # Add title
    ax.set_title(title)


im_mean, ax_mean = imshow_ax(axes[1][0], mean, title='Mean')
im_var, ax_var = imshow_ax(axes[2][0], var, title='Var')

contourf_ax(axes[1][1], mean, title='Mean')
contourf_ax(axes[2][1], var, title='Var')

# %%
# Show figure
fig.tight_layout()
fig.savefig('beta_distribution.png')
plt.show()
# %%
