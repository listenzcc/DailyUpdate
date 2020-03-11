# %%
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

# %%
# Tools


def legal_check(lst):
    """ Check if all element in [lst] is 1-dim array. """
    if not isinstance(lst, list):
        lst = [lst]
    for e in lst:
        assert(len(e.shape) == 1)


def entropy(a, b=None):
    """ Compute entropy of [a] if [b] is None,
    compute joint entropy of [a] and [b],
    histogram is used to estimate the distribution. """
    if b is None:
        legal_check(a)
        dist = np.histogram(a)[0]
    else:
        legal_check([a, b])
        dist = np.histogram2d(a, b)[0].ravel()
    return scipy.stats.entropy(dist)


def conditional_entropy(a, b):
    """ Compute conditional entropy of P([a]|[b]). """
    legal_check([a, b])
    return entropy(a, b) - entropy(b)


def mutual_entropy(a, b):
    """ Compute mutual entropy of [a] and [b]. """
    legal_check([a, b])
    return entropy(a) - conditional_entropy(a, b)


def triangle_function(fun, x, freq):
    """ Compute triangle [fun] of [x] as [freq]. """
    # Correct fun, like sin, cos, ...
    assert(isinstance(fun, type(np.sin)))
    # Time axis, unit is Second
    assert(len(x.shape) == 1)
    # Frequency, unit is Hz
    assert(isinstance(freq, type(5.0)))

    return fun(x * freq * 2 * np.pi), x


# %%
x = np.linspace(start=0, stop=1, num=100, endpoint=False)
cos, x_cos = triangle_function(fun=np.cos, x=x, freq=5.0,)
sin, x_sin = triangle_function(fun=np.sin, x=x, freq=1.0)
plt.style.use('ggplot')
plt.plot(x_sin, sin, label='sin')
plt.plot(x_cos, cos, label='cos')
plt.legend(loc='best')

# %%
H = dict(
    H_sin=entropy(sin),
    H_cos=entropy(cos),
    H_joint=entropy(sin, cos),
    H_joint2=entropy(cos, sin),
    H_conditional=conditional_entropy(sin, cos),
    H_conditional2=conditional_entropy(cos, sin),
    H_mutual=mutual_entropy(sin, cos),
    H_mutual2=mutual_entropy(cos, sin)
)

pprint(H)

# %%
plt.show()