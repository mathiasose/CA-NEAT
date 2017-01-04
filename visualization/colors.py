from matplotlib.colors import BoundaryNorm, ListedColormap
from seaborn.palettes import color_palette

n_colors = 4
colormap = ListedColormap(
    color_palette(['#EEEEEE', '#e74c3c', '#95a5a6', '#34495e'], n_colors=n_colors),
    N=n_colors
)
norm = BoundaryNorm(list(range(0, n_colors + 1)), n_colors)
