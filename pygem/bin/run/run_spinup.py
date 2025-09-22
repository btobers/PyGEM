import sys, shutil, json, warnings
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
# pygem imports
import pygem.setup.config as config
# check for config
config.ensure_config()
# read the config
pygem_prms = config.read_config()
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance_wrapper
#from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving, update_cfg
import pygem.pygem_modelsetup as modelsetup
from pygem.shop import debris, oib
from pygem.utils._funcs import interp1d_fill_gaps
from oggm import tasks, workflow
from oggm.core import flowline
from oggm import cfg

# load dhdt data to compare model dhdt against
def get_dhdt(dates_table, ela=0, rgi6id='', rgi7id=''):
    try:
        icebridge = oib.oib(rgi6id=rgi6id, rgi7id=rgi7id)
        if icebridge.rgi7id is None:
            raise ValueError(f"No RGI7id found for {icebridge.rgi6id}")

        icebridge._load()
        icebridge._parsediffs()
        icebridge._filter_on_pixel_count(
            pctl=pygem_prms['calib']['data']['oib']['oib_filter_pctl'], 
            inplace=True
        )
        icebridge._terminus_mask(inplace=True)
        icebridge._remove_outliers_zscore(zscore=3, inplace=True)
        icebridge._rebin(
            agg=pygem_prms['calib']['data']['oib']['oib_rebin'], 
            inplace=True
        )

        # retain only diffs within model timespan
        _, oib_inds, _ = np.intersect1d(
            list(icebridge.oib_diffs.keys()), 
            dates_table.date.to_numpy(), 
            return_indices=True
        )
        icebridge.oib_diffs = {
            key: icebridge.oib_diffs[key] 
            for i, key in enumerate(icebridge.oib_diffs) 
            if i in oib_inds
        }

        if len(icebridge.oib_diffs) < 2:
            raise ValueError("Must be at least two individual OIB surveys to difference.")

        icebridge._dbl_diff()
        icebridge._surge_mask(ela=ela, threshold=2, inplace=True)
        icebridge.set_diff_inds_map(dates_table)

        return icebridge

    except Exception as e:
        print(f"get_dhdt failed: {e}")
        return None


# get model monthly deltah
def get_dhdt_hat(gdir, diff_inds_map, bin_edges, nyears):
    # load flowline_diagnostics from spinup
    f = gdir.get_filepath('fl_diagnostics', filesuffix='_dynamic_spinup_pygem_mb')
    with xr.open_dataset(f, group='fl_0') as ds_spn:
        ds_spn = ds_spn.load()

    thickness_m = ds_spn.thickness_m.values.T # glacier thickness [m ice], (nbins, nyears)

    # set any < 0 thickness to nan
    thickness_m[thickness_m<=0] = np.nan

    # climatic mass balance
    dotb_monthly = np.repeat(ds_spn.climatic_mb_myr.values.T[:,1:] / 12, 12, axis=-1)

    # convert to m ice
    dotb_monthly = dotb_monthly * (pygem_prms['constants']['density_water'] / pygem_prms['constants']['density_ice'])
    ### to get monthly thickness and mass we require monthly flux divergence ###
    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy repeat to repeat values across 12 months
    flux_div_monthly_mmo = np.repeat(-ds_spn.flux_divergence_myr.values.T[:,1:] / 12, 12, axis=-1)
    # get monthly binned change in thickness
    delta_h_monthly = dotb_monthly - flux_div_monthly_mmo # [m ice per month]

    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_h_monthly = np.cumsum(delta_h_monthly, axis=-1)
    h_monthly =  running_delta_h_monthly + thickness_m[:,0][:,np.newaxis]

    # get surface height at the specified reference year
    ref_surface_h = ds_spn.bed_h.values + ds_spn.thickness_m.sel(time=pygem_prms['calib']['data']['oib']['oib_surface_reference_yr']).values

    # aggregate model bin thicknesses as desired
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        h_monthly = np.column_stack([stats.binned_statistic(x=ref_surface_h, values=x, statistic=np.nanmean, bins=bin_edges)[0] for x in h_monthly.T])

    # interpolate over any empty bins
    h_monthly_ = np.column_stack([
        interp1d_fill_gaps(x.copy()) for x in h_monthly.T
    ])

    # difference each set of inds in diff_inds_map
    dh = np.column_stack([h_monthly_[:,tup[1]] - h_monthly_[:,tup[0]] for tup in diff_inds_map])
    # divide by nyears
    dhda = dh / np.array(nyears)
    return dhda

def loss_with_penalty(x, obs, mod, threshold=100, weight=1.0):
    # MAE where observations exist
    mismatch = np.nanmean(np.abs(mod - obs))

    # Penalty: positive modeled values below threshold
    mask = x < threshold
    mod_sub = mod[mask]

    # keep only positives
    positives = np.clip(mod_sub, a_min=0, a_max=None)

    # add to loss (scales with mean positive magnitude)
    penalty = weight * np.nanmean(positives)

    return mismatch + penalty

# run spinup function
def run_spinup(gd, ye, **kwargs):
    out = workflow.execute_entity_task(
        tasks.run_dynamic_spinup,
        gd,
        minimise_for='area',
        ye=ye,
        output_filesuffix="_dynamic_spinup_pygem_mb",
        store_fl_diagnostics=True,
        store_model_geometry=True,
        mb_model_historical=PyGEMMassBalance_wrapper(gd, fl_str="model_flowlines"),
        ignore_errors=False,
        **kwargs
    )
    return out


def run(glacno_list, optimize=False, outdir=None, debug=False, ncores=1, **kwargs):
    # pull target_yr if provided
    target_yr = kwargs.get("target_yr")
    ye = target_yr if (target_yr is not None and not optimize) else 2022

    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
    # model dates
    dt = modelsetup.datesmodelrun(startyear=1940, endyear=ye)     # broad bounds based on current elevation change data available and max spinup period
    # load climate data
    ref_clim = class_climate.GCM(name=pygem_prms['climate']['ref_gcm_name'])

    # Air temperature [degC]
    temp, _ = ref_clim.importGCMvarnearestneighbor_xarray(ref_clim.temp_fn, ref_clim.temp_vn, main_glac_rgi, dt)
    # Precipitation [m]
    prec, _ = ref_clim.importGCMvarnearestneighbor_xarray(ref_clim.prec_fn, ref_clim.prec_vn, main_glac_rgi, dt)
    # Elevation [m asl]
    elev = ref_clim.importGCMfxnearestneighbor_xarray(ref_clim.elev_fn, ref_clim.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    lr, _ = ref_clim.importGCMvarnearestneighbor_xarray(ref_clim.lr_fn, ref_clim.lr_vn, main_glac_rgi, dt)

    # load prior regionally averaged modelprms (from Rounce et al. 2023)
    priors_df = pd.read_csv(pygem_prms["root"] + "/Output/calibration/" + pygem_prms["calib"]["priors_reg_fn"])

    # loop through gdirs and add `glacier_rgi_table`, `historical_climate`, `dates_table` and `modelprms` attributes to each glacier directory
    for i, glac_no in enumerate(glacno_list):
        try:
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[i], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            # instantiate glacier directory
            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_frontalablation']:
                gd = single_flowline_glacier_directory(glacier_str, reset=False)
                gd.is_tidewater = False
            else:
                gd = single_flowline_glacier_directory_with_calving(glacier_str, reset=False)
                gd.is_tidewater = True
            if debug:
                print(f"Running {glac_no}{' (tidewater)' if gd.is_tidewater else ''}")

            # Select subsets of data
            gd.glacier_rgi_table = glacier_rgi_table
            # Add climate data to glacier directory (first inversion data)
            gd.historical_climate = {"elev": elev[i],
                                    "temp": temp[i,:],
                                    "tempstd": np.zeros(temp[i,:].shape),
                                    "prec": prec[i,:],
                                    "lr": lr[i,:]}
            gd.dates_table = dt

            # model ela
            yrs = list(range(pygem_prms['climate']['ref_startyear'], min(pygem_prms['climate']['ref_endyear'], 2019) + 1))
            ela = tasks.compute_ela(gd, years=yrs)

            # get model params from emulator calibration
            modelprms_fn = glacier_str + '-modelprms_dict.json'
            modelprms_fp = (pygem_prms['root'] + '/Output/calibration/' + glacier_str.split('.')[0].zfill(2) 
                            + '/') + modelprms_fn
            with open(modelprms_fp, 'r') as f:
                modelprms_dict = json.load(f)
            
            modelprms_all = modelprms_dict['emulator']
            gd.modelprms = {'kp': modelprms_all['kp'][0],
                        'tbias': modelprms_all['tbias'][0],
                        'ddfsnow': modelprms_all['ddfsnow'][0],
                        'ddfice': modelprms_all['ddfice'][0],
                        'tsnow_threshold': modelprms_all['tsnow_threshold'][0],
                        'precgrad': modelprms_all['precgrad'][0]}

            # # get modelprms from regional priors
            # priors_idx = np.where((priors_df.O1Region == gd.glacier_rgi_table["O1Region"]) & 
            #                                             (priors_df.O2Region == gd.glacier_rgi_table["O2Region"]))[0][0]
            # tbias_mu = float(priors_df.loc[priors_idx, "tbias_mean"])
            # kp_mu = float(priors_df.loc[priors_idx, "kp_mean"])
            # gd.modelprms = {"kp": kp_mu,
            #                     "tbias": tbias_mu,
            #                     "ddfsnow": pygem_prms["calib"]["MCMC_params"]["ddfsnow_mu"],
            #                     "ddfice": pygem_prms["calib"]["MCMC_params"]["ddfsnow_mu"] / pygem_prms["sim"]["params"]["ddfsnow_iceratio"],
            #                     "precgrad": pygem_prms["sim"]["params"]["precgrad"],
            #                     "tsnow_threshold": pygem_prms["sim"]["params"]["tsnow_threshold"]}

            # update cfg.PARAMS
            update_cfg({"continue_on_error" : True}, "PARAMS")
            update_cfg({"store_model_geometry" : True}, "PARAMS")

            # get dhdt data
            dhdt = get_dhdt(gd.dates_table, ela=ela.values.min(), rgi6id=gd.rgi_id.split('-')[1])
            deltah_dict = dhdt._get_dbldiffs()

            ### get bin index cutoff for lowest Nth percentile ###
            valid_inds = np.where(dhdt._get_area() > 0)[0]
            valid_elevs = dhdt._get_centers()[valid_inds]
            thresh = np.percentile(valid_elevs, 30)
            thresh = min([thresh, ela.values.min()])

            # highest index (in valid_inds) where elevation <= threshold
            uppermost_bin = valid_inds[valid_elevs <= thresh].max()

            if dhdt is not None:
                results = {}

                def objective(spinup_period):
                    kwargs['spinup_period'] = spinup_period
                    fls = run_spinup(gd, ye, **kwargs)

                    # get true spinup period (if initial fails, oggm tries period/2)
                    spinup_period_ = gd.rgi_date+1 - fls[0].y0

                    dhdt.set_diff_inds_map(
                        modelsetup.datesmodelrun(
                            startyear=fls[0].y0, endyear=ye
                        )
                    )

                    model = get_dhdt_hat(
                        gd, dhdt._get_diff_inds_map(), dhdt._get_edges(), deltah_dict['nyears']
                    )

                    # penalize positive values below specified elevation threshold
                    loss = loss_with_penalty(dhdt._get_centers(), deltah_dict['dhdt'], model, thresh)
                    # l = np.nanmean(
                    #     np.abs(model[:uppermost_bin, :] - deltah_dict['dhdt'][:uppermost_bin, :])
                    # )
                    return spinup_period_, loss, model

                # evaluate candidates once
                candidate_periods = np.arange(20,61,5)
                for p in candidate_periods:
                    p_, mismatch, model = objective(p)
                    results[p_] = (mismatch, model)

                # find best
                best_period = min(results, key=lambda k: results[k][0])
                best_value, best_model = results[best_period]

                if debug:
                    print("All results:", {k: v[0] for k, v in results.items()})
                    print(f"Best spinup_period = {best_period}, mismatch = {best_value}")

                    best_period = min(results, key=lambda k: results[k][0])
                    best_value, best_model = results[best_period]

                    worst_period = max(results, key=lambda k: results[k][0])
                    worst_value, worst_model = results[worst_period]

                    labels = [f'{t[0].year}{str(t[0].month).zfill(2)}-{t[1].year}{str(t[1].month).zfill(2)}' for t in deltah_dict['dates']]
                    fig, ax = plt.subplots(figsize=(8, 5))

                    for t in range(deltah_dict['dhdt'].shape[1]):
                        # plot Obs first, grab the color
                        line, = ax.plot(
                            dhdt._get_centers(),
                            deltah_dict['dhdt'][:, t],
                            linestyle='-',
                            marker='.',
                            label=labels[t]
                        )
                        color = line.get_color()

                        # plot Best model with same color
                        ax.plot(
                            dhdt._get_centers(),
                            best_model[:, t],
                            linestyle='--',
                            marker='.',
                            color=color,
                        )

                        # plot Worst model with same color
                        ax.plot(
                            dhdt._get_centers(),
                            worst_model[:, t],
                            linestyle=':',
                            marker='.',
                            color=color,
                        )
                    ax.axvline(dhdt._get_centers()[uppermost_bin], c='grey', ls=':')
                    ax.axhline(0, c='grey', ls='-')
                    ax.plot([],[],'k--',label=r'$\hat{best}$')
                    ax.plot([],[],'k:', label=r'$\hat{worst}$')
                    ax.set_xlabel("elevation (m)")
                    ax.set_ylabel(r"elevation change (m yr$^{-1}$)")
                    ax.set_title(
                        f"{glac_no}\nBest={best_period} (mismatch={best_value:.3f}), "
                        f"Worst={worst_period} (mismatch={worst_value:.3f})"
                    )
                    ax.legend(handlelength=1, borderaxespad=0, fancybox=False)
                    # plot area
                    area = dhdt._get_area()
                    area_mask = area>0
                    ax2 = ax.twinx()  # shares x-axis
                    ax2.fill_between(dhdt._get_centers()[area_mask], 0, area[area_mask], color='gray', alpha=0.1)
                    ax2.set_ylim([0,ax2.get_ylim()[1]])
                    ax2.set_ylabel(r"area (m $^{2}$)", color='gray')
                    ax2.tick_params(axis='y', colors='gray')
                    ax2.spines['right'].set_color('gray')
                    ax2.yaxis.label.set_color('gray')
                    fig.tight_layout()
                    if ncores==1:
                        plt.show()
                    if outdir:
                        fig.savefig(f'{outdir}/{glac_no}-spinup_optimization.png',dpi=300)
                    plt.close()
            else:
                best_period = None    # just use OGGM default

            # rerun spinup explicitly for the best candidate - or default if not minimizing against dhdt obs
            kwargs['spinup_period'] = best_period
            run_spinup(gd, ye, **kwargs)

        except Exception as e:
            print(f"Error processing glacier {glac_no}: {e}")
            # continue to next glacier
            continue


def main():
    # define ArgumentParser
    parser = argparse.ArgumentParser(description="perform dynamical spinup")
    # add arguments
    parser.add_argument('-rgi_region01', type=int, default=pygem_prms['setup']['rgi_region01'],
                        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)', nargs='+')
    parser.add_argument('-rgi_region02', type=str, default=pygem_prms['setup']['rgi_region02'], nargs='+',
                        help='Randoph Glacier Inventory subregion (either `all` or multiple spaced integers,  e.g. `-run_region02 1 2 3`)')
    parser.add_argument('-rgi_glac_number', action='store', type=float, default=pygem_prms['setup']['glac_no'], nargs='+',
                        help='Randoph Glacier Inventory glacier number (can take multiple)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='filepath containing list of rgi_glac_number, helpful for running batches on spc'),
    parser.add_argument('-target_yr', type=int, default=None)
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-outdir', type=str, default=None, help='directory to store any ouputs (diagnostic figures, etc.)')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-spinup_period',
        type=int,
        default=None,
        help="Fixed spinup period (years). If not provided, OGGM default is used."
    )
    group.add_argument(
        '-optimize',
        action='store_true',
        help="Optimize the spinup_period by minimizing against elevation change data."
    )
    args = parser.parse_args()
    
    # RGI glacier number
    glac_no = None
    if args.rgi_glac_number:
        glac_no = args.rgi_glac_number
        # format appropriately
        glac_no = [float(g) for g in glac_no]
        glac_no = [f"{g:.5f}" if g >= 10 else f"0{g:.5f}" for g in glac_no]
    elif args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'r') as f:
            glac_no = json.load(f)
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=args.rgi_region01, rgi_regionsO2=args.rgi_region02,
                include_landterm=pygem_prms['setup']['include_landterm'], include_laketerm=pygem_prms['setup']['include_laketerm'],
                include_tidewater=pygem_prms['setup']['include_tidewater'], min_glac_area_km2=pygem_prms['setup']['min_glac_area_km2'])
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    if glac_no is None:
        raise ValueError('Need to specify either -rgi_glac_number or -rgi_glac_number_fn')
    
    # number of cores for parallel processing
    if args.ncores > 1:
        ncores = int(np.min([len(glac_no), args.ncores]))
    else:
        ncores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=ncores)

    # set up partial function with debug argument
    run_partial = partial(run, optimize=args.optimize, outdir=args.outdir, debug=args.debug, ncores=ncores, target_yr=args.target_yr, spinup_period=args.spinup_period)
    # parallel processing
    print(f'Processing with {ncores} cores... \n{glac_no_lsts}')
    with multiprocessing.Pool(ncores) as p:
        p.map(run_partial, glac_no_lsts)

if __name__ == "__main__":
    main()    