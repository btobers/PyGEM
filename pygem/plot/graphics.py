import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def plot_modeloutput_section(model=None, ax=None, title='', **kwargs):
    """Plots the result of the model output along the flowline.
    A paired down version of OGGMs graphics.plot_modeloutput_section()

    Parameters
    ----------
    model: obj
        either a FlowlineModel or a list of model flowlines.
    fig
    title
    """

    try:
        fls = model.fls
    except AttributeError:
        fls = model

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    else:
        fig = plt.gcf()

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in fls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # plot Centerlines
    cls = fls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Plot glacier
    t1 = cls.thick[:-2]
    t2 = cls.thick[1:-1]
    t3 = cls.thick[2:]
    pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
    cls.surface_h[np.where(pnan)[0] + 1] = np.nan

    if 'srfcolor' in kwargs.keys():
        srfcolor = kwargs['srfcolor']
    else: 
        srfcolor = '#003399'

    if 'srfls' in kwargs.keys():
        srfls = kwargs['srfls']
    else: 
        srfls = '-'

    ax.plot(x, cls.surface_h, color=srfcolor, linewidth=2, ls=srfls, label='Glacier')

    # Plot tributaries
    for i, l in zip(cls.inflow_indices, cls.inflows):
        if l.thick[-1] > 0:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='#993399',
                    markeredgecolor='k',
                    label='Tributary (active)')
        else:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='w',
                    markeredgecolor='k',
                    label='Tributary (inactive)')
    if getattr(model, 'do_calving', False):
        ax.hlines(model.water_level, x[0], x[-1], linestyles=':', color='C0')

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Title
    ax.set_title(title, loc='left')
