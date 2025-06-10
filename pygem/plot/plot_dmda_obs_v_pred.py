# btobers, 20250211
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cftime
import warnings


def main(jsonfp, outfp='', outdir=None):
    print(jsonfp)
    with open(jsonfp,'r') as dat:
        data = json.load(dat)['MCMC']
    glacno = jsonfp.split('/')[-1][:7]
    print('mb obs:\t\t',round(data['mb_obs_mwea'][0],3))
    print('mb post:\t',round(np.nanmedian(np.asarray(data['mb_mwea']['chain_0'])),3))

    # get density vector from median rho of MCMC chain
    rhoabl = np.nanmedian(np.asarray(data['rhoabl']['chain_0']))
    rhoacc = np.nanmedian(np.asarray(data['rhoacc']['chain_0']))
    ela = np.min(data['ela']['z'])
    bin_z = np.asarray(data['dmda']['x'])
    abl_mask = (bin_z<ela)
    rho = np.ones_like(bin_z)
    rho[abl_mask] = rhoabl
    rho[~abl_mask] = rhoacc

    data=data['dmda']
    cum_area = np.cumsum(np.asarray(data['area'])*1e-6)

    # get date time spans
    dates = data['dates']
    labels = [f'{l[0][:-2].replace('-','')}:{l[1][:-3].replace('-','')}' for l in dates]
    dates = [tuple([cftime.DatetimeNoLeap(*map(int, date.split('-'))) for date in sublist]) for sublist in dates]

    # bin_centers = np.asarray(bin_z)
    # Compute bin width (assuming evenly spaced bins)
    # bin_width = np.diff(bin_z).mean()

    # # Compute bin edges
    # bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2], 
    #                             bin_centers + bin_width / 2))
    
    # reshape obs
    obs = np.asarray(data['obs'][0]).reshape(len(bin_z), len(data['obs'][0]) // len(bin_z))/1e3
    sigma_obs = np.asarray(data['obs'][1]).reshape(len(bin_z), len(data['obs'][1]) // len(bin_z))/1e3

    # instantiate subplots
    fig, ax = plt.subplots(nrows=len(dates), ncols=1, figsize=(5, len(dates)*2), 
                            gridspec_kw={'hspace': 0.075}, sharex=True)
    
    # Transform functions
    def cum_area_to_elev(x): return np.interp(x, cum_area, bin_z)
    def elev_to_cum_area(x): return np.interp(x, bin_z, cum_area)

    if not isinstance(ax,np.ndarray):
        ax=[ax]
    for t in range(len(data['chain_0'][0]) // len(bin_z)):
        # axb = ax[t].twinx()
        ax[t].xaxis.set_label_position('top')
        ax[t].xaxis.tick_top()  # move ticks to top
        ax[t].tick_params(axis='x', which='both', top=False)

        ax[t].axhline(y=0, c='grey', lw=0.5)
        # axb.yaxis.set_label_position('left')
        # axb.yaxis.set_ticks_position('left')
        stack = []

        # i are the chain steps
        for i in range(len(data['chain_0'])):
            preds = np.asarray(data['chain_0'][i]).reshape(len(bin_z), len(data['chain_0'][0]) // len(bin_z))[:,t] / 1e3
            stack.append(preds)

        stack = np.stack(stack)
        stack[:,np.where(np.asarray(data['area'])==0)[0]] = np.nan  # mask out where area <= 0

        ax[t].fill_between(cum_area,
                         obs[:,t]*rho-sigma_obs[:,t]*rho,
                         obs[:,t]*rho+sigma_obs[:,t]*rho,
                         color='k',alpha=.125)
        ax[t].plot(cum_area, obs[:,t]*rho, 'k', label='Obs.')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ax[t].fill_between(cum_area,
                            np.nanpercentile(stack, 5, axis=0),
                            np.nanpercentile(stack, 95, axis=0),
                            color='r', alpha=.25)
            ax[t].plot(cum_area, np.nanmedian(stack,axis=0), 'r', label='Pred.')

        # for r in stack:
        #     axb.plot(bin_z, r, 'r', alpha=.0125)
        # axb.plot(bin_z, np.nanmin(stack,axis=0), 'r', label='Pred.')

        # dummy label for timespan
        ax[t].text(0.99175, 0.980, labels[t], transform=ax[t].transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(facecolor='white', edgecolor='black', alpha=1, boxstyle='square,pad=0.25'), zorder=10)

        secaxx = ax[t].secondary_xaxis('bottom', functions=(cum_area_to_elev, elev_to_cum_area))

        if t != len(dates)-1:
            secaxx.tick_params(axis='x', labelbottom=False)
        else:
            secaxx.set_xlabel("Elevation (m)")

        if t==0:
            leg = ax[t].legend(handlelength=1, borderaxespad=0, fancybox=False, loc='lower right', edgecolor='k', framealpha=1)
            for legobj in leg.legend_handles:
                legobj.set_linewidth(2.0)
        # else:
            # Turn off cumulative area ticks and labels
        ax[t].tick_params(axis='x', which='both', top=False, labeltop=False)

    ax[0].set_xlim([elev_to_cum_area(np.min(bin_z)), elev_to_cum_area(np.max(bin_z))])


    for a in ax:
        # plot ela
        a.axvline(x=elev_to_cum_area(ela), c='k', ls=':', lw=1)
        # plot area
        # a.fill_between(bin_z, 0, np.asarray(data['area'])*1e-6, color='steelblue', alpha=.125)
        # a.set_ylim([0,a.get_ylim()[-1]*3])
        # a.yaxis.set_label_position('right')
        # a.yaxis.set_ticks_position('right')
        # a.yaxis.set_label_position("right")
        # a.spines['right'].set_color('steelblue')
        # a.yaxis.label.set_color('steelblue')
        # a.tick_params(axis='y', colors='steelblue')
        # a.set_ylim([0, np.ceil(2*a.get_ylim()[1])])      
        # a.set_yticks([0,np.rint(max(np.asarray(data['area'])*1e-6))])
    
    # axb.set_xlim(np.asarray(bin_z)[np.where(np.asarray(data['area'])>0)[0][[0,-1]]].tolist())
    # ax[-1].sec('Elevation (m)')
    ax[-1].text(0.0125, .5, r'Mass change (kg m$^{-2}$)', horizontalalignment='left', rotation=90,
                    verticalalignment='center', transform=fig.transFigure)

    # ax[-1].text(0.95, .5, 'Glacier Area (km$^2$)', c='steelblue', horizontalalignment='left', rotation=90,
    #                 verticalalignment='center', transform=fig.transFigure)
    ax[0].text(0.5, 1.1, f'{glacno}', horizontalalignment='center',
                    verticalalignment='center', transform=ax[0].transAxes)

    # Remove overlapping tick labels from secaxx
    fig.canvas.draw()  # Force rendering to get accurate bounding boxes
    labels = secaxx.get_xticklabels()
    renderer = fig.canvas.get_renderer()
    bboxes = [label.get_window_extent(renderer) for label in labels]
    # Only show labels spaced apart by at least `min_spacing` pixels
    min_spacing = 15  # adjust as needed
    last_right = -float('inf')
    for label, bbox in zip(labels, bboxes):
        if bbox.x0 > last_right + min_spacing:
            last_right = bbox.x1
        else:
            label.set_visible(False)

    if outfp:
        plt.savefig(outfp, dpi=300, bbox_inches='tight')
        plt.close()
    elif outdir:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfp = os.path.join(outdir, f'{glacno}_dmda_obs_v_pred.png')
        plt.savefig(outfp, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot specific mass change, observed v. predicted")
    # add arguments
    parser.add_argument('jsonfp', type=str, nargs='+')
    parser.add_argument('-outfp', type=str, default='')
    parser.add_argument('-outdir', type=str, default='')

    args = parser.parse_args()
    path = args.jsonfp

    if path:
        for p in path:
            print(p)
            main(p, outfp=args.outfp, outdir=args.outdir)