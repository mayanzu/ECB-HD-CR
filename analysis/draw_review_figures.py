"""Vector manuscript figures from the revision's recorded evidence.

Contract: quantitative deployment and finite-window diagnostic panels.
Python/matplotlib; PDF converted from editable SVG text,
183 mm width, white background, per-seed CSV sources, mean +/- sample SD.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cairosvg

ROOT=Path(__file__).resolve().parents[1]
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
WIDTH=183/25.4
COLORS={'ECB-HD-CR':'#7a0019','D3QN-EH':'#111111'}
SETTINGS=[('centered_bs',80,'Center\n80'),('edge_bs_high_load',80,'Edge\n80'),
          ('critical_scarcity',80,'Critical\n80'),('edge_bs_high_load',120,'Edge\n120'),
          ('scarce_high_load',120,'Scarce\n120')]


def save(fig,prefix):
    fig.savefig(str(prefix)+'.svg',bbox_inches='tight')
    # The editable SVG is the source used for the manuscript PDF.
    cairosvg.svg2pdf(url=str(prefix)+'.svg',write_to=str(prefix)+'.pdf')
    fig.savefig(str(prefix)+'.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def draw_quantitative(directory,results):
    df=pd.read_csv(results/'deployment_per_seed.csv')
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,2.75),layout='constrained')
    metrics=[('completion','Completion ratio'),('FND','First-node death (round)'),
             ('energy_per_task_uJ',r'Scheduling energy / task ($\mu$J)')]
    x=np.arange(len(SETTINGS)); offset=.13
    for panel,(ax,(metric,label)) in enumerate(zip(axes,metrics)):
        for method,shift in [('ECB-HD-CR',-offset),('D3QN-EH',offset)]:
            means=[]; stds=[]
            for scenario,n,_ in SETTINGS:
                values=df[(df.scenario==scenario)&(df.N==n)&(df.method==method)][metric]
                if len(values)!=10: raise ValueError('Figure requires all 10 seeds')
                means.append(values.mean()); stds.append(values.std(ddof=1))
            ax.errorbar(x+shift,means,yerr=stds,fmt='o',markersize=3,capsize=2,
                        color=COLORS[method],label=method,linewidth=.8)
        ax.set_xticks(x,[s[2] for s in SETTINGS]); ax.set_ylabel(label)
        ax.text(-.15,1.06,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.grid(axis='y',color='#E8E8E8',linewidth=.5)
    axes[0].legend(fontsize=6,loc='lower left')
    save(fig,directory/'deployment_audit')

    diag=pd.read_csv(results/'deployment_diagnostics.csv')
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,2.75),layout='constrained')
    for scenario,color,label in [('edge_bs_high_load','#7a0019','Edge, N=80'),
                                 ('critical_scarcity','#111111','Critical, N=80')]:
        d=diag[(diag.scenario==scenario)&(diag.N==80)].copy()
        for i,(metric,ylabel) in enumerate([('reference','Bill reference (J)'),
                                           ('alive_after_round','Alive nodes'),('loss','Replay MSE')]):
            g=d.groupby('round')[metric].agg(['mean','std'])
            if metric=='loss':
                # Smoothing is for display only; CSV retains every original loss.
                g=g.rolling(20,min_periods=1).mean()
            axes[i].plot(g.index,g['mean'],color=color,label=label,linewidth=1)
            axes[i].fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=color,alpha=.12)
            axes[i].set_xlabel('Round'); axes[i].set_ylabel(ylabel)
    for i,ax in enumerate(axes):
        ax.text(-.15,1.06,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.grid(axis='y',color='#E8E8E8',linewidth=.5)
    axes[0].legend(fontsize=6)
    save(fig,directory/'controller_diagnostics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/deployment')
    parser.add_argument('--results', type=Path, default=ROOT/'results/review_revision_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    draw_quantitative(args.output, args.results)


if __name__ == '__main__':
    main()
