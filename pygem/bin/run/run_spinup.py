import sys, shutil, json
import argparse
import numpy as np
import pandas as pd
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
    shutil.copy(gdir.get_filepath('model_flowlines'), gdir.get_filepath('model_flowlines', filesuffix=f"_w{spinup_opt}"))


def oggm_spinup(gdir ,spinup_opt, **kwargs):
    # perform OGGM dynamic spinup
    workflow.execute_entity_task(tasks.run_dynamic_spinup,
                            gdir,
                            spinup_start_yr=1979,  # When to start the spinup
                            minimise_for='area',  # what target to match at the RGI date
                            target_yr=2000, # The year at which we want to match area or volume. If None, gdir.rgi_date + 1 is used (the default)
                            ye=2020,  # When the simulation should stop
                            model_flowline_filesuffix=f"_w{spinup_opt}",  # The suffix of the model file to start from
                            output_filesuffix=f"_dynamic_spinup_w{spinup_opt}",
                            store_fl_diagnostics=True,
                            store_model_geometry=True,
                            # first_guess_t_spinup = , could be passed as input argument for each step in the sampler based on prior tbias, current default first guess is -2
                            **kwargs);

    # store model flowlines at year 2000 - add debris back to flowlines
    fmd_dynamic = flowline.FileModel(gdir.get_filepath("model_geometry", filesuffix=f"_dynamic_spinup_w{spinup_opt}"));
    fmd_dynamic.run_until(2000);
    gdir.write_pickle(fmd_dynamic.fls, "model_flowlines", filesuffix=f"_dynamic_spinup_w{spinup_opt}_yr2000");
    debris.debris_binned(gdir, fl_str="model_flowlines", filesuffix=f"_dynamic_spinup_w{spinup_opt}_yr2000");


def run(glacno_list, mb_model='oggm'):
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)

    if mb_model == 'oggm':

        for glac in range(main_glac_rgi.shape[0]):

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_frontalablation']:
                gdir_spinup = single_flowline_glacier_directory(glacier_str, reset=True)
                gdir_spinup.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir_spinup = single_flowline_glacier_directory_with_calving(glacier_str, reset=True)
                gdir_spinup.is_tidewater = True

            # update cfg.PARAMS
            update_cfg({"continue_on_error" : True}, "PARAMS")

            # do bed inversion
            l3_proc(gdir_spinup, mb_model)
            
            # do spinup
            oggm_spinup(gdir_spinup, mb_model)
    
    elif mb_model == 'pygem':

        gcm_name = pygem_prms['climate']['gcm_name']
        dt_spinup = modelsetup.datesmodelrun(startyear=1979, endyear=2019)
        gcm_spinup = class_climate.GCM(name=gcm_name)

        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        # Air temperature [degC]
        gcm_temp_spinup, gcm_dates_spinup = gcm_spinup.importGCMvarnearestneighbor_xarray(gcm_spinup.temp_fn, gcm_spinup.temp_vn, main_glac_rgi, dt_spinup)
        if pygem_prms['mb']['option_ablation'] == 2 and gcm_name in ['ERA5']:
            gcm_tempstd_spinup, gcm_dates_spinup = gcm_spinup.importGCMvarnearestneighbor_xarray(gcm_spinup.tempstd_fn, gcm_spinup.tempstd_vn,
                                                                            main_glac_rgi, dt_spinup)
        else:
            gcm_tempstd_spinup = np.zeros(gcm_temp_spinup.shape)
        # Precipitation [m]
        gcm_prec_spinup, gcm_dates_spinup = gcm_spinup.importGCMvarnearestneighbor_xarray(gcm_spinup.prec_fn, gcm_spinup.prec_vn, main_glac_rgi, dt_spinup)
        # Elevation [m asl]
        gcm_elev_spinup = gcm_spinup.importGCMfxnearestneighbor_xarray(gcm_spinup.elev_fn, gcm_spinup.elev_vn, main_glac_rgi)
        # Lapse rate [degC m-1]
        gcm_lr_spinup, gcm_dates_spinup = gcm_spinup.importGCMvarnearestneighbor_xarray(gcm_spinup.lr_fn, gcm_spinup.lr_vn, main_glac_rgi, dt_spinup)

        for glac in range(main_glac_rgi.shape[0]):

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms['setup']['include_frontalablation']:
                gdir_spinup = single_flowline_glacier_directory(glacier_str, reset=True)
                gdir_spinup.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir_spinup = single_flowline_glacier_directory_with_calving(glacier_str, reset=True)
                gdir_spinup.is_tidewater = True
            
            # Add climate data to glacier directory
            gdir_spinup.historical_climate = {'elev': gcm_elev_spinup[glac],
                                    'temp': gcm_temp_spinup[glac,:],
                                    'tempstd': gcm_tempstd_spinup[glac,:],
                                    'prec': gcm_prec_spinup[glac,:],
                                    'lr': gcm_lr_spinup[glac,:]}
            gdir_spinup.dates_table = dt_spinup

            # get modelprms from regional priors

            priors_df = pd.read_csv(pygem_prms['root'] + '/Output/calibration/' + pygem_prms['calib']['priors_reg_fn'])
            priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                                        (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
            tbias_mu = float(priors_df.loc[priors_idx, 'tbias_mean'])
            kp_mu = float(priors_df.loc[priors_idx, 'kp_mean'])
            modelprms_spinup = {'kp': kp_mu,
                                'tbias': tbias_mu,
                                'ddfsnow': pygem_prms['calib']['MCMC_params']['ddfsnow_mu'],
                                'ddfice': pygem_prms['calib']['MCMC_params']['ddfsnow_mu'] / pygem_prms['sim']['params']['ddfsnow_iceratio'],
                                'precgrad': pygem_prms['sim']['params']['precgrad'],
                                'tsnow_threshold': pygem_prms['sim']['params']['tsnow_threshold']}
                
            # update cfg.PARAMS
            update_cfg({"continue_on_error" : False}, "PARAMS")

            # add debris to inversion_flowlines
            debris.debris_binned(gdir_spinup, fl_str='inversion_flowlines')

            # do bed inversion
            l3_proc(gdir_spinup, mb_model,
                    **{'mb_model': PyGEMMassBalance_wrapper(gdir=gdir_spinup, 
                                            modelprms=modelprms_spinup, 
                                            glacier_rgi_table=glacier_rgi_table, 
                                            fls=gdir_spinup.read_pickle('inversion_flowlines'))})

            # do spinup
            oggm_spinup(gdir_spinup, mb_model,
                            **{'mb_model_historical' : PyGEMMassBalance_wrapper(gdir=gdir_spinup, 
                                        modelprms=modelprms_spinup, 
                                        glacier_rgi_table=glacier_rgi_table, 
                                        fls=gdir_spinup.read_pickle("model_flowlines", filesuffix=f"_w{mb_model}"))})


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

    # call main
    run(glac_no, mb_model=args.mb_model)

if __name__ == "__main__":
    main()    