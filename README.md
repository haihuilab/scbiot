scimorph: theme_publication
=============

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI version</td>
            <td style="text-align: center;">
                <a href="https://badge.fury.io/py/scimorph">
                    <img src="https://badge.fury.io/py/scimorph.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">conda-forge version</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/conda-forge/scimorph">
                    <img src="https://anaconda.org/conda-forge/scimorph/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

> **Warning**
> : The `scimorph` library requires LaTeX for math formatting. Install it using:
>   `sudo apt install texlive texlive-latex-extra texlive-fonts-recommended cm-super dvipng`.
> : Starting with version 1.0.0, you must add `import scimorph` **before** setting the style with `theme_publication('publication')`.


*Matplotlib styles for scientific figures*

This repo has Matplotlib styles to format your figures for scientific papers, presentations and theses.

<p align="center">
<img src="https://github.com/haihuilab/scimorph/blob/main/examples/plots/fig01a.jpg" width="500">
</p>

You can find [the full tutorials of scimorph here](https://github.com/haihuilab/scimorph/wiki/Gallery).

Getting Started
---------------

The easiest way to install scimorph is by using `pip`:

```bash
# to install the latest release (from PyPI)
pip install scimorph

# to install the latest commit (from GitHub)
pip install git+https://github.com/haihuilab/scimorph

# to clone and install from a local copy
git clone https://github.com/haihuilab/scimorph.git
cd scimorph
pip install -e .
```

From version `v1.0.0` on, `import scimorph` is needed on top of your scripts so Matplotlib can make use of the styles.

**Notes:** 
- scimorph-theme_publication requires matplotlib or seaborn

Using the Styles
----------------

``"publication"`` is the primary style in this repo. Whenever you want to use it, simply add the following to the top of your python script:

```python
import matplotlib.pyplot as plt
import scimorph

theme_publication('publication')
```


Examples
--------
```python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scimorph.theme_publication import theme_publication
dir = Path.cwd()
print('parent dir: ',dir)

x = np.linspace(0, 2 * np.pi, 500)
y = np.sin(x)
# df = pd.DataFrame({'x': x, 'y': y})

theme_publication('publication', 
                    figsize='medium', 
                    fontsize=None, 
                    grid=True,
                    border=True)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'{dir}/examples/plots/fig01a.jpg')
plt.show()

```
The basic ``publication`` style is shown below:

<img src="https://github.com/haihuilab/scimorph/blob/main/examples/plots/fig01a.jpg" width="500">



If you use ``scimorph`` in your paper/thesis, feel free to add it to the list!

Citing scimorph:
-------------------

    @article{scimorph,
      author       = {Haihui Zhang et al},
      title        = {haihuilab/scimorph},
      month        = Jan,
      year         = 2025,
      publisher    = {github},
      version      = {1.0.0},      
      url          = {https://github.com/haihuilab/scimorph}
    }
