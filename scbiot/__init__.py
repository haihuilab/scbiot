from pathlib import Path
import matplotlib.pyplot as plt
import scimorph

# Register bundled stylesheets in the matplotlib style library
styles_path = Path(scimorph.__path__[0]) / 'styles'
stylesheets = plt.style.core.read_style_directory(styles_path)

# Update matplotlib style library
plt.style.library.update(stylesheets)
# Available styles
plt.style.available[:] = sorted(plt.style.library.keys())
