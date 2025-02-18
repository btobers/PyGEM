"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

NASA Operation IceBridge data and processing class
"""
import re, os, glob, json, pickle, datetime, warnings, sys
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt
# Local libraries
import pygem.setup.config as config
# Read the config
pygem_prms = config.read_config()  # This reads the configuration file

class oib:
    def __init__(self, rgi6id='', rgi7id=''):
        self.oib_datpath = f"{pygem_prms['root']}/{pygem_prms['calib']['data']['oib']['oib_relpath']}"
        self.rgi7_6_df = pd.read_csv(f"{self.oib_datpath}/../oibak_rgi6_rgi7_ids.csv")
        self.rgi7_6_df['rgi7id'] = self.rgi7_6_df['rgi7id'].str.split('RGI2000-v7.0-G-').str[1]
        self.rgi7_6_df['rgi6id'] = self.rgi7_6_df['rgi6id'].str.split('RGI60-').str[1]
        self.rgi6id = rgi6id
        self.rgi7id = rgi7id
        self.name = None
        # instatntiate dictionary to hold all data - store the data by survey date, with each key containing a tuple with the binned differences and uncertainties (diffs, sigma)
        self.oib_diffs = {}
        self.dbl_diffs = {}
        self.bin_edges = None
        self.bin_centers = None
        self.bin_area = None
    
    def _get_diffs(self):
        return self.oib_diffs
    def _set_diffs(self, diffs_dict):
        self.oib_diffs = diffs_dict
    def _get_dbldiffs(self):
        return self.dbl_diffs
    def _get_centers(self):
        return self.bin_centers
    def _get_edges(self):
        return self.bin_edges
    def _get_area(self):
        return self.bin_area
    def _get_name(self):
        return self.name

    def _rgi6torgi7id(self, debug=False):
        """
        return RGI version 7 glacier id for a given RGI version 6 id

        """
        self.rgi6id = self.rgi6id.split('.')[0].zfill(2) + '.' + self.rgi6id.split('.')[1]
        # rgi7id = self.rgi7_6_df.loc[lambda self.rgi7_6_df: self.rgi7_6_df['rgi6id'] == rgi6id,'rgi7id'].tolist()
        rgi7id = self.rgi7_6_df.loc[self.rgi7_6_df['rgi6id'] == self.rgi6id, 'rgi7id'].tolist()
        if len(rgi7id)==1:
            self.rgi7id =  rgi7id[0]
            if debug:
                print(f'RGI6:{self.rgi6id} -> RGI7:{self.rgi7id}')
        elif len(rgi7id)==0:
            raise IndexError(f'No matching RGI7Id for {self.rgi6id}')
        elif len(rgi7id)>1:
            raise IndexError(f'More than one matching RGI7Id for {self.rgi6id}')
        

    def _rgi7torgi6id(self, debug=False):
        """
        return RGI version 6 glacier id for a given RGI version 7 id

        """
        self.rgi7id = self.rgi7id.split('-')[0].zfill(2) + '-' + self.rgi7id.split('-')[1]
        # rgi6id = self.rgi7_6_df.loc[lambda self.rgi7_6_df: self.rgi7_6_df['rgi7id'] == rgi7id,'rgi6id'].tolist()
        rgi6id = self.rgi7_6_df.loc[self.rgi7_6_df['rgi7id'] == self.rgi7id, 'rgi6id'].tolist()
        if len(rgi6id)==1:
            self.rgi6id = rgi6id[0]
            if debug:
                print(f'RGI7:{self.rgi7id} -> RGI6:{self.rgi6id}')
        elif len(rgi6id)==0:
            raise IndexError(f'No matching RGI6Id for {self.rgi7id}')
        elif len(rgi6id)>1:
            raise IndexError(f'More than one matching RGI6Id for {self.rgi7id}')


    def _load(self):
        """
        load Operation IceBridge data
        """
        oib_fpath = glob.glob(f"{self.oib_datpath}/diffstats5_*{self.rgi7id}*.json")
        if len(oib_fpath)==0:
            return
        else:
            oib_fpath = oib_fpath[0]
        # load diffstats file
        with open(oib_fpath, 'rb') as f:
            self.oib_dict = json.load(f)
            self.name = split_by_uppercase(self.oib_dict['glacier_shortname'])


    def _parsediffs(self, filter_count_pctl=10, debug=False):
        """
        loop through OIB dataset, get differences
        diffs_stacked: np.ndarray (#bins, #surveys)
        """
        # get seasons stored in oib diffs
        seasons = list(set(self.oib_dict.keys()).intersection(['march','may','august']))
        for ssn in seasons:
            for yr in list(self.oib_dict[ssn].keys()):
                # get survey date
                doy_int = int(np.ceil(self.oib_dict[ssn][yr]['mean_doy']))
                dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
                # get survey data and filter by pixel count
                diffs = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_median_diffs_vec'])
                counts = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_count_vec'])
                mask = _filter_on_pixel_count(counts, filter_count_pctl)
                diffs[mask] = np.nan
                # uncertainty represented by IQR
                sigmas = np.asarray(self.oib_dict[ssn][yr]['bin_vals']['bin_interquartile_range_diffs_vec'])
                sigmas[mask] = np.nan
                # add masked diffs to master dictionary
                self.oib_diffs[round_to_nearest_month(dt_obj)] = (diffs,sigmas)
        # Sort the dictionary by date keys
        self.oib_diffs = dict(sorted(self.oib_diffs.items()))

        if debug:
            print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in list(self.oib_diffs.keys())])}')
        # get bin centers
        self.bin_centers = (np.asarray(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']) + 
                    np.asarray(self.oib_dict[ssn][list(self.oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2
        self.bin_area = self.oib_dict['aad_dict']['hist_bin_areas_m2']
        # bin_edges = oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']
        # bin_edges.append(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'][-1])
        # bin_edges = np.asarray(bin_edges)


    def _terminus_mask(self, debug=False):
        """
        create mask of missing terminus ice using last oib survey
        """
        survey_dates = list(self.oib_diffs.keys())
        inds = range(len(survey_dates))[::-1]
        diffs = [tup[0] for tup in self.oib_diffs.values()]
        # only look above bins with area>0
        lowest_bin = np.where(np.asarray(self.bin_area) != 0)[0][0]
        idx = None
        mask = []
        try:
            for i in inds:
                tmp = diffs[i][lowest_bin:lowest_bin+50]
                if np.isnan(tmp).all():
                    continue
                else:
                    # find peak we'll bake in the assumption that terminus thickness has decreased over time - we'll thus look for a trough if yr>=2013 (cop30 date)
                    if survey_dates[i].year>2013:
                        idx = np.nanargmin(tmp) + lowest_bin
                    else:
                        tmp = -1*tmp
                        idx = np.nanargmax(tmp) + lowest_bin
                    mask = np.arange(0,idx+1,1)
                    break
            if debug:
                plt.figure()
                cmap=plt.cm.rainbow(np.linspace(0, 1, len(inds)))
                for i in inds[::-1]:
                    plt.plot(diffs[i],label=f'{survey_dates[i].year}:{survey_dates[i].month}:{survey_dates[i].day}',c=cmap[i])
                if idx:
                    plt.axvline(idx,c='k',ls=':')
                plt.legend(loc='upper right')
                plt.show()

        except Exception as err:
            if debug:
                print(f'_filter_terminus_missing_ice error: {err}')
            mask = []

        # apply mask
        for tup in self.oib_diffs.values():
            tup[0][mask] = np.nan
            tup[1][mask] = np.nan


    def _rebin(self, agg=100):
        if agg:
            # aggregate both model and obs to specified size m bins
            nbins = int(np.ceil((self.bin_centers[-1] - self.bin_centers[0]) // agg))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                for i,(k, tup) in enumerate(self.oib_diffs.items()):
                    if i==0:
                        y, self.bin_edges, _ = stats.binned_statistic(x=self.bin_centers, values=tup[0], statistic=np.nanmean, bins=nbins)
                    else:
                        y = stats.binned_statistic(x=self.bin_centers, values=tup[0], statistic=np.nanmean, bins=self.bin_edges)[0]
                    s = stats.binned_statistic(x=self.bin_centers, values=tup[1], statistic=np.nanmean, bins=self.bin_edges)[0]
                    self.oib_diffs[k] = (y,s)
                self.bin_area  = stats.binned_statistic(x=self.bin_centers, values=self.bin_area, statistic=np.nanmean, bins=self.bin_edges)[0]
            self.bin_centers = ((self.bin_edges[:-1] + self.bin_edges[1:]) / 2)


    # double difference all oib diffs from the same season 1+ year apart
    def _dbl_diff(self, tolerance_months=1):
        # prepopulate dbl_diffs dictionary object will structure with dates, dh, sigma
        # where dates is a tuple for each double differenced array in the format of (date1,date2),
        # where date1's cop30 differences were subtracted from date2's to get the dh values for that time span,
        # and the sigma was taken as the mean sigma from each date
        self.dbl_diffs['dates'] = []
        self.dbl_diffs['dh'] = []
        self.dbl_diffs['sigma'] = []

        # convert keys to a sorted list
        sorted_dates = list(self.oib_diffs.keys())
        # iterate through sorted dates
        for i, date1 in enumerate(sorted_dates[:-1]):
            for j in range(i + 1, len(sorted_dates)):
                date2 = sorted_dates[j]
                delta_mon = date2.month - date1.month
                delta_mon = ((date2.year - date1.year) * 12) + delta_mon
                # calculate the modulus to find how far the difference is from the closest multiple of 12
                rem = abs(delta_mon % 12)
                # check if the difference is approximately an integer multiple of years ± n month
                if rem <= tolerance_months or rem >= 12 - tolerance_months:
                    self.dbl_diffs['dates'].append((date1,date2))
                    self.dbl_diffs['dh'].append(self.oib_diffs[date2][0] - self.oib_diffs[date1][0])
                    # self.dbl_diffs['sigma'].append((self.oib_diffs[date2][1] + self.oib_diffs[date1][1]) / 2)
                    self.dbl_diffs['sigma'].append(self.oib_diffs[date2][1] + self.oib_diffs[date1][1])
                    break  # Stop looking for further matches for date1

        # column stack dh and sigmas into single 2d array
        if len(self.dbl_diffs['dh'])>0:
            self.dbl_diffs['dh'] = np.column_stack(self.dbl_diffs['dh'])
            self.dbl_diffs['sigma'] = np.column_stack(self.dbl_diffs['sigma'])
        else:
            self.dbl_diffs['dh'] = np.nan
        # check if deltah is all nan
        if np.isnan(self.dbl_diffs['dh']).all():
            self.dbl_diffs['dh'] = None
            self.dbl_diffs['sigma'] = None


    def _elevchange_to_masschange(self, ela, density_ablation=pygem_prms['constants']['density_ice'], density_accumulation=700):
        # convert elevation changes to mass change using piecewise density conversion
        if self.dbl_diffs['dh'] is not None:
            # populate density conversion column corresponding to bin center elevation
            conversion_factor = np.ones(len(self.bin_centers))
            conversion_factor[np.where(self.bin_centers<ela)] = density_ablation
            conversion_factor[np.where(self.bin_centers>=ela)] = density_accumulation
            # get change in mass per unit area as (dz  * rho) [dmass / dm2]
            self.dbl_diffs['dmda'] = self.dbl_diffs['dh'] * conversion_factor[:,np.newaxis]
            self.dbl_diffs['dmda_err'] = self.dbl_diffs['sigma'] * conversion_factor[:,np.newaxis]
        else:
            self.dbl_diffs['dmda'] = None
            self.dbl_diffs['dmda_err'] = None


def _filter_on_pixel_count(arr, pctl = 15):
    """
    filter oib diffs by perntile pixel count
    """
    arr=arr.astype(float)
    arr[arr==0] = np.nan
    mask = arr < np.nanpercentile(arr,pctl)
    return mask


def split_by_uppercase(text):
    """Add space before each uppercase letter (except at the start of the string."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text)


def round_to_nearest_month(dt):
    """Round a datetime object to the nearest month."""
    if dt.day >= 15:
        # Round up to the first day of next month
        next_month = dt.replace(day=1) + timedelta(days=32)
        return next_month.replace(day=1)
    else:
        # Round down to the first day of the current month
        return dt.replace(day=1)