import matplotlib.pyplot as plt
from pathlib import Path



def theme_publication(style='publication', figsize=(3, 3), fontsize=None, grid=False, border=True):
    """
    Apply a custom matplotlib style for publication-quality plots.
    """
    # Determine the directory paths
    # current_dir = Path.cwd()
    # parent_dir = current_dir.parent
    # print(f"Parent Directory: {parent_dir}")

    # # Check if a custom style file exists and apply it
    # mplstyles = list(parent_dir.rglob('styles/*.mplstyle'))
    # if mplstyles:
    #     print(f"mplstyles: {mplstyles}")
    #     for style in mplstyles:            
    #         print(f"Applying style: {style}")
    #         plt.style.use(str(style))
    # else:
    #     print('No .mplstyle files were found!')

    # # Get a list of colors from the style's color cycle
    # try:
    #     colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    #     print(f'Available Colors: {colors}')
    #     # print('\n'.join(color for color in colors))
    # except KeyError:
    #     print('No color cycle found in the current style.')
    if style in plt.style.available:
        try:
            plt.style.use([style])
        except OSError as e:
            print(f"Error: Could not apply style '{style}'. {e}")
    

    #------------------------------------------------------------------
    # fontsize
    size_mapping = {
        # Predefined figure sizes and fontsize: [(width, height), fontsize]
        'small':       [(2.5, 2.5),  8],
        'medium':      [(5, 5),     16],
        'large':       [(10, 10),   32],
        'small_wide':  [(4, 2.5),    8],
        'medium_wide': [(8, 5),     16],
        'large_wide':  [(16, 10),   32],
        'small_long':  [(2.5, 4),    8],
        'medium_long': [(5, 8),     16],
        'large_long':  [(10, 16),   32],
        'super':       [(20, 20),   64]
    }

    # 1) If `figsize` is a string, get the default from `size_mapping`.
    if isinstance(figsize, str):
        default_figsize, default_fontsize = size_mapping.get(figsize, ((3, 3), 8))
        figsize = default_figsize
        # Use custom fontsize if provided; otherwise fall back to default_fontsize     
        fontsize = default_fontsize if fontsize is None else fontsize
        

    # 2) If `figsize` is a 2-tuple, let it be, but use default fontsize=8 if none is provided.
    elif isinstance(figsize, tuple) and len(figsize) == 2:
        figsize = figsize 
        fontsize = 8 if fontsize is None else fontsize

    else:
        raise ValueError("`figsize` must be a 2-tuple or a valid key in size_mapping.")

    # Now we have a valid figsize and a determined fontsize (either user-provided or default).
    plt.gcf().set_size_inches(*figsize)
    plt.rcParams.update({
    'axes.labelsize':   fontsize,  # Axis labels
    'xtick.labelsize':  fontsize,  # X tick labels
    'ytick.labelsize':  fontsize,  # Y tick labels
    'legend.fontsize':  fontsize,  # Legend
    'figure.titlesize': fontsize,  # Figure title
    })


    #-------------------------------------------
    # grid
    if grid:
        # Turn on
        plt.gca().grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    
    #-------------------------------------------
    # border
    if border:
        plt.gca().spines['top'].set_linewidth(0.5)
        plt.gca().spines['right'].set_linewidth(0.5)
        plt.gca().spines['bottom'].set_linewidth(0.5)
        plt.gca().spines['left'].set_linewidth(0.5)
    else:
        plt.gca().spines['top'].set_linewidth(0)
        plt.gca().spines['right'].set_linewidth(0)
        plt.gca().spines['bottom'].set_linewidth(0.5)
        plt.gca().spines['left'].set_linewidth(0.5)


    #-------------------------------------------
    # for subplots
    # Update subplot font sizes
    for ax in plt.gcf().get_axes():
        ax.title.set_size(fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontsize)

    # Enable grid if requested
    if grid:
        for ax in plt.gcf().get_axes():
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Configure border visibility and linewidth
    for ax in plt.gcf().get_axes():     
        if border:
            ax.spines['top'].set_linewidth(0.5)
            ax.spines['right'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
        else:
            ax.spines['top'].set_linewidth(0)
            ax.spines['right'].set_linewidth(0)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)



        
# def savefig(output_path, figsize='small', dpi=300):
#     """
#     Save the current matplotlib figure to a file with a specified figure size
#     and (optionally) a custom font size.

#     Parameters:
#     - output_path: Path to save the figure
#     - figsize: Either a 2-tuple (width, height) or one of the predefined keys in `size_mapping`.
#     - fontsize: Font size to use. If None, falls back to the default from `size_mapping`
#                 (for string keys) or 8 (for numeric tuples).
#     - dpi: Resolution of the saved figure (default: 300)
#     """

    
#     plt.tight_layout()

#     plt.savefig(output_path, dpi=dpi)
