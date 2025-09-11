import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scimorph import theme_publication

dir = Path.cwd()
print('Current dir:',dir)
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
plt.savefig(f'{dir}/examples/plots/fig01a.png')
plt.show()