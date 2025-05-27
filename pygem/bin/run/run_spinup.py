import sys, shutil, json
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
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
from pygem.shop import debris
from oggm import tasks, workflow
from oggm.core import flowline
from oggm import cfg


def l3_proc(gdir, spinup_opt, **kwargs):
    """
    OGGGM L3 preprocessing steps
    """
    # process climate_hisotrical data to gdir
    workflow.execute_entity_task(tasks.process_climate_data, gdir)

    # process mb_calib data from geodetic mass balance
    workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                gdir, informed_threestep=True, overwrite_gdir=True,
                                )

    # glacier bed inversion
    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdir, **kwargs)

    # add debris back to inversion_flowlines after inversion
    debris.debris_binned(gdir, fl_str='inversion_flowlines')
    
    workflow.calibrate_inversion_from_consensus(
        gdir,
        apply_fs_on_mismatch=True,
        error_on_mismatch=True,  # if you running many glaciers some might not work
        filter_inversion_output=True,  # this partly filters the overdeepening due to
        # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
        volume_m3_reference=None,  # here you could provide your own total volume estimate in m3
    )

    # after inversion, merge data from preprocessing tasks form model_flowlines
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdir)

    # add debris to model_flowlines
    debris.debris_binned(gdir, fl_str="model_flowlines")

    # copy model_flowlines to model_flowlines_{spinup_opt}
    shutil.copy(gdir.get_filepath('model_flowlines'), gdir.get_filepath('model_flowlines', filesuffix=f"_0_{spinup_opt}_mb"))


def oggm_spinup(gdir ,spinup_opt, **kwargs):
    # perform OGGM dynamic spinup
    workflow.execute_entity_task(tasks.run_dynamic_spinup,
                            gdir,
                            # spinup_start_yr=,  # When to start the spinup
                            minimise_for='area',  # what target to match at the RGI date
                            # target_yr=target_yr, # The year at which we want to match area or volume. If None, gdir.rgi_date + 1 is used (the default)
                            # ye=,  # When the simulation should stop
                            model_flowline_filesuffix=f"_0_{spinup_opt}_mb",  # The suffix of the model file to start from
                            output_filesuffix=f"_dynamic_spinup_{spinup_opt}_mb",
                            store_fl_diagnostics=True,
                            store_model_geometry=True,
                            # first_guess_t_spinup = , could be passed as input argument for each step in the sampler based on prior tbias, current default first guess is -2
                            **kwargs);


def run(glacno_list, mb_model='oggm', reset_gdir=False, do_spinup=True, **kwargs):
    # remove any None-valued kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
    if mb_model == 'oggm':

        for glac in range(main_glac_rgi.shape[0]):

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_frontalablation']:
                gdir = single_flowline_glacier_directory(glacier_str, reset=reset_gdir)
                gdir.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, reset=reset_gdir)
                gdir.is_tidewater = True

            # update cfg.PARAMS
            update_cfg({"continue_on_error" : True}, "PARAMS")

            # do bed inversion
            l3_proc(gdir, mb_model)
            if do_spinup:
                # do spinup
                oggm_spinup(gdir, mb_model, **kwargs)
    
    elif mb_model == 'pygem':
        dt = modelsetup.datesmodelrun(startyear=kwargs['spinup_start_yr'], endyear=kwargs['ye']-1)
        gcm_name = 'ERA5'
        gcm = class_climate.GCM(name=gcm_name)

        # Air temperature [degC]
        gcm_temp, _ = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dt)
        if pygem_prms['mb']['option_ablation'] == 2 and gcm_name in ['ERA5']:
            gcm_tempstd, _ = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                            main_glac_rgi, dt)
        else:
            gcm_tempstd = np.zeros(gcm_temp.shape)
        # Precipitation [m]
        gcm_prec, _ = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dt)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
        # Lapse rate [degC m-1]
        gcm_lr, _ = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dt)

        for glac in range(main_glac_rgi.shape[0]):

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_frontalablation']:
                gdir = single_flowline_glacier_directory(glacier_str, reset=reset_gdir)
                gdir.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, reset=reset_gdir)
                gdir.is_tidewater = True

            # Add climate data to glacier directory (first inversion data)
            gdir.historical_climate = {'elev': gcm_elev[glac],
                                    'temp': gcm_temp[glac,:],
                                    'tempstd': gcm_tempstd[glac,:],
                                    'prec': gcm_prec[glac,:],
                                    'lr': gcm_lr[glac,:]}
            gdir.dates_table = dt

            # get modelprms from regional priors
            priors_df = pd.read_csv(pygem_prms['root'] + '/Output/calibration/' + pygem_prms['calib']['priors_reg_fn'])
            priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                                        (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
            tbias_mu = float(priors_df.loc[priors_idx, 'tbias_mean'])
            kp_mu = float(priors_df.loc[priors_idx, 'kp_mean'])
            modelprms = {'kp': kp_mu,
                                'tbias': tbias_mu,
                                'ddfsnow': pygem_prms['calib']['MCMC_params']['ddfsnow_mu'],
                                'ddfice': pygem_prms['calib']['MCMC_params']['ddfsnow_mu'] / pygem_prms['sim']['params']['ddfsnow_iceratio'],
                                'precgrad': pygem_prms['sim']['params']['precgrad'],
                                'tsnow_threshold': pygem_prms['sim']['params']['tsnow_threshold']}
                
            # update cfg.PARAMS
            update_cfg({"continue_on_error" : True}, "PARAMS")
            update_cfg({"store_model_geometry" : True}, "PARAMS")
            # add debris to inversion_flowlines
            debris.debris_binned(gdir, fl_str='inversion_flowlines')

            # do bed inversion
            l3_proc(gdir, mb_model,
                    **{
                        'mb_model': PyGEMMassBalance_wrapper(gdir=gdir, 
                                        modelprms=modelprms, 
                                        glacier_rgi_table=glacier_rgi_table, 
                                        fls=gdir.read_pickle('inversion_flowlines')),})

            if do_spinup:
                # do spinup
                oggm_spinup(gdir, mb_model,
                                **{**{'mb_model_historical' : PyGEMMassBalance_wrapper(gdir=gdir, 
                                            modelprms=modelprms, 
                                            glacier_rgi_table=glacier_rgi_table, 
                                            fls=gdir.read_pickle("model_flowlines", filesuffix=f"_0_{mb_model}_mb"))},
                                **kwargs})


def main():
    # define ArgumentParser
    parser = argparse.ArgumentParser(description="perform dynamical spinup")
    # add arguments
    parser.add_argument('-rgi_glac_number', action='store', type=float, default=pygem_prms['setup']['glac_no'], nargs='+',
                        help='Randoph Glacier Inventory glacier number (can take multiple)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='filepath containing list of rgi_glac_number, helpful for running batches on spc'),
    parser.add_argument('-mb_model', type=str, choices=['oggm', 'pygem'], default='oggm',
                        help='mass balance model to use during inversion and spinup ["oggm" or "pygem"]')
    parser.add_argument('-spinup_start_yr', type=int, default=1979)
    parser.add_argument('-target_yr', type=int, default=None)
    parser.add_argument('-ye', type=int, default=2020)
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-no_spinup', action='store_true', default=False,
                        help='Skip dynamical spinup?')
    parser.add_argument('-reset_gdir', action='store_true', default=False,
                        help='Reset oggm galcier directory?')
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
    run_partial = partial(run, mb_model=args.mb_model, spinup_start_yr=args.spinup_start_yr, target_yr=args.target_yr, ye=args.ye, reset_gdir=args.reset_gdir, do_spinup=not args.no_spinup)
    # parallel processing
    print(f'Processing with {ncores} cores... \n{glac_no_lsts}')
    with multiprocessing.Pool(ncores) as p:
        p.map(run_partial, glac_no_lsts)

if __name__ == "__main__":
    main()    