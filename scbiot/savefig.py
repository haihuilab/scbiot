import matplotlib.pyplot as plt



def savefig(output_path, size=(3, 3), dpi=300):
    """
    Save the current matplotlib figure to a file with a specified figure size.

    Parameters:
    - output_path: Path to save the figure
    - size: Tuple specifying the figure size or a key in the size_mapping
    - dpi: Resolution of the saved figure (default: 300)
    """
    size_mapping = {
        'small': (2.5, 2.5),
        'medium': (5, 5),
        'large': (10, 10),
        'small_wide': (4, 2.5),
        'medium_wide': (8, 5),
        'large_wide': (16, 10),
        'small_long': (2.5, 4),
        'medium_long': (5, 8),
        'large_long': (10, 16),
        'super': (20, 20)
    }

    if isinstance(size, str):
        size = size_mapping.get(size, (3, 3))  # Default to (3, 3) if key not found
    elif not (isinstance(size, tuple) and len(size) == 2):
        raise ValueError("Size must be a tuple or a key in size_mapping.")

    plt.gcf().set_size_inches(*size)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

