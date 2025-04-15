"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os,sys
import logging

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic

from oggm.utils import entity_task
import matplotlib.pyplot as plt
import pygem.setup.config as config
# Read the config
pygem_prms = config.read_config()  # This reads the configuration file

# Module logger
log = logging.getLogger(__name__)

@entity_task(log, writes=['inversion_flowlines'])
def dhst_binned(gdir, ignore_debris=False, fl_str='inversion_flowlines', filesuffix=''):
    """Bin Hugonnet et al. 20201 dhdt.
        
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    
    fl_str : str
        The name of the flowline file to read. Default is 'inversion_flowlines'.

    filesuffix : str
        The filesuffix to use when reading the flowline file. Default is ''.

    """
    # Nominal glaciers will throw error, so make sure inversion_flowlines exist
    try:
        flowlines = gdir.read_pickle(fl_str, filesuffix=filesuffix)
        fl = flowlines[0]
        
        assert len(flowlines) == 1, 'Error: binning dhdt only works for single flowlines at present'
        
    except:
        flowlines = None        

    if flowlines is not None:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            glacier_mask = ds['glacier_mask'].values
            topo = ds['topo_smoothed'].values
            assert 'hugonnet_dhdt' in ds.data_vars, 'Error: Hugonnet dhdt data not found in gdir'
            dhdt = ds['hugonnet_dhdt'].values

            # Only bin on-glacier values
            idx_glac = np.where(glacier_mask == 1)
            topo_onglac = topo[idx_glac]
            dhdt_onglac = dhdt[idx_glac]

            # Bin edges        
            nbins = len(fl.dis_on_line)
            z_center = (fl.surface_h[0:-1] + fl.surface_h[1:]) / 2
            z_bin_edges = np.concatenate((np.array([topo[idx_glac].max() + 1]), 
                                          z_center, 
                                          np.array([topo[idx_glac].min() - 1])))
            # # Loop over bins and calculate the mean debris thickness and enhancement factor for each bin
            dhdt_binned = np.full(nbins, np.nan)

            for nbin in np.arange(0,len(z_bin_edges)-1):
                bin_max = z_bin_edges[nbin]
                bin_min = z_bin_edges[nbin+1]
                in_bin = (topo_onglac < bin_max) & (topo_onglac >= bin_min)

                values = dhdt_onglac[in_bin]
                # Compute nanmean if there are values
                if np.any(~np.isnan(values)):
                    dhdt_binned[nbin] = np.nanmean(values)

            fl.dhdt = dhdt_binned

        # Overwrite pickle
        gdir.write_pickle(flowlines, fl_str, filesuffix=filesuffix)        