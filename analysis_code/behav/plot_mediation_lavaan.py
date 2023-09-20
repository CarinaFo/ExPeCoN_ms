
import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import pingouin as pg
from pathlib import Path

# load df
df_lavaan = pd.read_csv(Path("D:/expecon_ms/analysis_code/behav/R\mediation/lavaan_mediation_multiple_alldata_expecon1.csv"))

df_lavaan2 = df_lavaan.pivot_table(values='est', index='subj_idx', columns='label').reset_index()
        
df = df_lavaan2

# significant mediation effect?

ttest_against_zero = pg.ttest(df['indirect'], 0)
ttest_against_zero              

# one plot for the main figure 3b
plt.close('all')
df['x'] = 0
ylims = [[0.005, 0.005, 0.005, 0.25], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]]

for vsidx, varsets in enumerate(['a', 'ab', 'b', 'c', 'total']):
    fig, ax = plt.subplots(nrows=1, ncols=len(varsets), figsize=(len(varsets)*1.3, 2))
    for vidx, v in enumerate(varsets):

        # now the summary
        ttest_against_zero = pg.ttest(df[v] * 10000, 0)
        if ttest_against_zero['p-val'].item() < 0.05:
            plotvars = {'mec': 'w', 'mfc': 'k', 'ms': 8}
        else:
            plotvars = {'mec': 'k', 'mfc': 'w', 'ms': 6}

        yerr = np.array(df[v].mean() - ttest_against_zero['CI95%'].item()[0] / 10000,
                        ttest_against_zero['CI95%'].item()[1] / 10000 - df[v].mean()).T
        ax[vidx].errorbar(y=df[v].mean(), x=0, yerr=yerr,
                            xerr=None, marker=group_markers[gridx], color='k', **plotvars)
        ax[vidx].axhline(0, linestyle=':', color='.15', zorder=-100)
        ax[vidx].set(ylim=[-ylims[vsidx][vidx], ylims[vsidx][vidx]],
                        xlim=[-0.1, 0.1],
        #    ylim=[-np.max(np.abs(df[v])) * 1.1, np.max(np.abs(df[v])) * 1.1],
                        xlabel='', xticklabels=[],
                        ylabel=v)
        #ax[vidx].legend_.remove()

        # do stats on the posterior distribution
        pval = ttest_against_zero['p-val'].item()
        txt = "p = {:.4f}".format(pval)
        # if pval < 0.001:
        #     txt = "***"
        # elif pval < 0.01:
        #     txt = "**"
        # elif pval < 0.05:
        #     txt = "*"
        # else:
        #     txt = ''
        
        if df[v].mean() > 0:
            star_y = ttest_against_zero['CI95%'].item()[1] /  10000 * 1.2
        else:
            star_y = ttest_against_zero['CI95%'].item()[0] / 10000 * 1.6

        ax[vidx].text(0, star_y, txt, fontsize='small', ha='center',
                        fontweight='bold')
        ax[vidx].tick_params(axis='x', colors='w')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(datapath,
                                'figures', 'lavaanMulti_3mediators_tr%s_v%s_gr%s.pdf'%(trials, vsidx, groups)),
                facecolor='white')