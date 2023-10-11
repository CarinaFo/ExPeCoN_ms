
# not cleaned up yet

def prev_choice_bias():
    """plot previous choice bias
    Parameters  ---------- None.
    Returns
    -------
    None
    """
    # load data
    # Correlate criterion with dprime

    os.chdir(savepath)

    diff_c = np.array(criterion_high) - np.array(criterion_low)
    diff_d = np.array(d_prime_high) - np.array(d_prime_low)

    sns.regplot(x=diff_c, y=diff_d, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in dprime', fontname="Arial", fontsize=14)
    plt.ylabel('change in c', fontname="Arial", fontsize=14)
    plt.savefig('c_d_corr.png')
    plt.savefig('c_d_corr.svg')

    plt.show()
    stats.pearsonr(diff_c, diff_d) # correlation of 0.45, p < .01

    # Previous choice bias

    # load choice bias
    choice = pd.read_csv("D:\\expecon_ms\\data\\behav\\previous_choice.csv")
    choice = choice.drop(9)
    choice = choice.reset_index(drop=True)

    # plot overall choice bias


    # Create strip plot with 'tip' column
    sns.stripplot(x=list(range(1,40)), y=choice.prev)

    # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("Participant ID")
    plt.ylabel("Previous choice bias")
    plt.xticks([])
    plt.savefig('prev_choice_groups.png')
    plt.savefig('prev_choice_groups.svg')

    # Show the plot
    plt.show()

    # only for the high exp trials

    sns.stripplot(x=list(range(1,40)), y=choice.prev_high)

    # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("Participant ID")
    plt.ylabel("Previous choice bias")
    plt.xticks([])
    # Show the plot
    plt.show()

    sns.stripplot(x=list(range(1,40)), y=choice.prev_low)
    """
    """ # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("participant")
    plt.ylabel("previous choice bias")
    plt.xticks([])

    # Show the plot
    plt.show()

    # Define repeater and alternator as boolean mask

    rep = choice.prev>0 # 19 repeaters
    alt = choice.prev<0 # 20 alternator

    # check correlation between criterion and choice bias for both groups

    # repeaters, sign. neg correlation (-0.73)

    sns.regplot(x=diff_c[rep], y=choice.prev[rep], color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in c', fontname="Arial", fontsize=14)
    plt.ylabel('choice bias', fontname="Arial", fontsize=14)
    plt.savefig('diff_c_prev_rep.png')
    plt.savefig('diff_c_prev_rep.svg')
    plt.show()

    stats.pearsonr(diff_c[rep], choice.prev[rep]) # overall correlation of 0.5

    # alternators, no sign. correlation (r=0.16, p=0.5)

    sns.regplot(x=diff_c[alt], y=choice.prev[alt], color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in c', fontname="Arial", fontsize=14)
    plt.ylabel('choice bias', fontname="Arial", fontsize=14)
    plt.savefig('diff_c_prev_alt.png')
    plt.savefig('diff_c_prev_alt.svg')
    plt.show()

    stats.pearsonr(diff_c[alt], choice.prev[alt]) 

    # repeaters
    # load questionnaire data (intolerance of uncertainty)

    q_df = pd.read_csv(r"D:\expecon\data\behav_brain\questionnaire_data\q_clean.csv")
    # Get the row index of the original DataFrame

    # Create a Boolean mask to identify the rows containing the values to drop
    mask = q_df['ID'].isin([16,32,42,45])

    # Drop the rows containing the values to drop
    q_df = q_df.drop(q_df[mask].index)

    q_df = q_df.reset_index()

    row_index = q_df.index

    clean_q = q_df.dropna(subset = "iu18_A")

    # Get the row index of the dropped rows
    dropped_row_index = row_index.difference(clean_q.index)

    # drop the 2 NaN from questionnaire data
    choice = choice.drop([dropped_row_index], axis=0)
    diff_c = np.delete(diff_c, [dropped_row_index], axis=0)

    # correlate c diff and prev choice bias with IUQ

    stats.pearsonr(stats.zscore(clean_q.iu_sum), diff_c)

    # Plot regplot 

    sns.regplot(x=stats.zscore(clean_q.iu_sum), y=diff_c, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('Intolerance of Uncertainty zscored', fontname="Arial", fontsize=14)
    plt.ylabel('previous choice bias', fontname="Arial", fontsize=14)
    plt.savefig('prev_choice_IUQ.png')
    plt.savefig('prev_choice_IUQ.svg')

    # Show the plot
    plt.show()

    sns.regplot(x=stats.zscore(clean_q.iu_sum), y=diff_c, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('Intolerance of Uncertainty zscored', fontname="Arial", fontsize=14)
    plt.ylabel('criterion change', fontname="Arial", fontsize=14)
    plt.savefig('c_diff_IUQ.png')
    plt.savefig('c_diff_IUQ.svg')

    # Show the plot
    plt.show()