import matplotlib.pyplot as plt
from pathlib import Path
from theme_publication import theme_publication
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to apply matplotlib styles and save figures.")
    parser.add_argument('--style', type=str, help="Path to a matplotlib style file.")
    parser.add_argument('--output', type=str, help="Output path for the saved figure.")
    parser.add_argument('--size', type=str, help="Size of the figure (e.g., 'medium', 'large').", default='small')
    parser.add_argument('--dpi', type=int, help="DPI for the saved figure.", default=300)
    args = parser.parse_args()   

    if args.style:
        theme_publication(args.style, size=args.size, grid=True)
    if args.output:
        plt.savefig(args.output,  dpi=args.dpi)

    import numpy as np
    from pathlib import Path
    dir = Path.cwd()
    print('parent dir: ',dir)
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    # df = pd.DataFrame({'x': x, 'y': y})

    theme_publication('publication', 
                      figsize='medium', 
                      fontsize=20, 
                      grid=True,
                      border=True)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{dir}/examples/plots/fig01a.jpg')
    plt.show()
