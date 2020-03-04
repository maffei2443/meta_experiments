import matplotlib as mpl
pgf_with_latex = {
    # # LaTeX default is 10pt font.
    "text.usetex": True,
    # # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",
    # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": "Times",
    "font.size": 14,
    # # blank entries should cause plots to inherit fonts from the document
    #"font.sans-serif": [],
    #"font.monospace": [],
    'path.simplify': True,
    'path.simplify_threshold': 0.1,
    'legend.markerscale': .9,
    'legend.numpoints': 1,
    'legend.handlelength': 2,
    'legend.scatterpoints': 1,
    'legend.labelspacing': 0.5,
    'legend.facecolor': '#eff0f1',
    'legend.edgecolor': 'none',
    'legend.handletextpad': 0.5,  # pad between handle and text
    'legend.borderaxespad': 0.5,  # pad between legend and axes
    'legend.borderpad': 0.5,  # pad between legend and legend content
    'legend.columnspacing': 1,  # pad between each legend column
    'axes.spines.left': True,
    'axes.spines.top': True,
    'axes.titlesize': 'large',
    'axes.spines.bottom': True,
    'axes.spines.right': True,
    'axes.axisbelow': True,
    'axes.grid': True,
    'image.cmap': 'RdYlBu',
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    'grid.alpha': .5,
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'lines.markeredgewidth': 1,
    'pgf.preamble': [
        r'\usepackage[utf8x]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{{{typeface}}}'
    ]
}
def set_config():
    mpl.rcParams.update(pgf_with_latex)