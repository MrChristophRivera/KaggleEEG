# This is my default parameters for graphs that I am making to customize my graphs using the RC settings. I am also including the tableu colors. 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Helvetica', 'Arial']

# create a dictionary to hold the stles of interst
style = {'axes.axisbelow': True,
         'axes.edgecolor': '.8',
         'axes.facecolor': 'white',
         'axes.grid': True,
         'axes.labelcolor': '.15',
         'axes.linewidth': 1.0,
         'axes.labelsize': 24,
         'axes.labelcolor': 'darkgrey',
         'axes.spines.right': 0,
         'axes.spines.top': 0,
         'axes.titlesize': 24,
         'figure.facecolor': (1, 1, 1, 0),
         'figure.figsize': (10, 7.5),
         'font.family': 'sans-serif',
         'font.sans-serif': ['Helvetica', 'Arial'],
         'grid.color': '.8',
         'grid.linestyle': u':',
         'image.cmap': u'Greys',
         'legend.frameon': False,
         'legend.numpoints': 1,
         'legend.scatterpoints': 1,
         'lines.solid_capstyle': u'round',
         'lines.linewidth': 2.5,
         'lines.markeredgewidth': 1,
         'lines.markersize': 10,
         'text.color': 'darkgrey',
         'xtick.color': 'darkgrey',
         'xtick.direction': u'out',
         'xtick.major.size': 0.0,
         'xtick.labelsize': 16,
         'xtick.minor.size': 0.0,
         'ytick.color': 'darkgrey',
         'ytick.direction': u'out',
         'ytick.major.size': 0.0,
         'ytick.minor.size': 0.0,
         'ytick.labelsize': 16}


def create_tableau_colors():
    '''Create as color list of the 20 tableue colors'''
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    return tableau20


def configure_plots():
    '''sets the matplotlib RC parameters and color map using searborn '''

    sns.set_context(style)
    sns.set_context(style)
    tableau20 = create_tableau_colors()
    sns.set_palette(tableau20)
