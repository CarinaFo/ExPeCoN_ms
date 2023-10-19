def permutation_cluster_correlation_test(X, behavior, test='pearson', threshold=0.05, n_permutations=100):
    '''performs cluster based permutation test on time freq correlation data

    each time freq voxel is correlated with behavioral data (across participants),
    based on threshold only sig correlations are kept and clustered,
    cluster mass (= T-value) is calculated as the sum of t-values of neighboring significant
    correlations, observed cluster mass of the biggest T-value and second biggest T-value is
    returned,
    in addition T-value distribution is returned when behavior array is permuted and
    biggest cluster mass is computed each time

    Parameters
    ----------
    X : ndarray
        eeg data, expected shape is (subjects, frequencies, timepoints)
    behavior : ndarray
        behavioural variable, expected shape is (subjects)
    test : str, optional
        'pearson' or 'spearman', by default 'pearson'
    threshold : float, optional
        initial clustering threshold - vertices with data values more 
        extreme than threshold will be used to form clusters, by default 0.05
    n_permutations : int, optional
        how often to permute behavioral data, by default 100

    Returns
    -------
    corr_matrix : ndarray
        correlation values for each time frequency voxel with behavioral data,
        expected shape is (frequencies, timepoints)
    cluster_matrix : ndarray
        each time frequency voxel which does not belong to a cluster is 0,
        cluster voxels are numbered, adjacent voxels have the same number,
        expected shape is (frequencies, timepoints)
    n_cluster : int
        number of clusters found
    observed_T : float
        cluster mass of the biggest observed cluster
    observed_T_2 : float
        cluster mass of the second biggest observed cluster
    T_distribution : ndarray
        all T values from permuted correlations tests,
        expected shape is (n_permutations,)
    '''

    import numpy as np
    import scipy.ndimage as ndimage
    import scipy.stats as stats
    import seaborn as sns
    import matplotlib.pyplot as plt

    # create correlation matrix
    corr_matrix = np.zeros([X.shape[1], X.shape[2]])
    t_matrix = np.zeros([X.shape[1], X.shape[2]])
    p_matrix = np.zeros([X.shape[1], X.shape[2]])
    n_VP = X.shape[0]
    n_freq = X.shape[1]
    n_time = X.shape[2]


    # calculate observed r, p and t values for each voxel
    for f in range(n_freq):
        for time in range(n_time):
            ERD = X[:, f, time]
            if test == 'pearson':
                r = np.corrcoef(ERD, behavior)[0, 1]  # pearson correlation
            else:
                r = stats.spearmanr(ERD, behavior)[0] # spearman correlation
            corr_matrix[f, time] = r

            # calculates t-values for each correlation value
            # is this also correct for spearman??
            t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
            t_matrix[f, time] = t_value

            # calculate p value based on t value
            # is this also correct for spearman??
            p = (1 - stats.t.cdf(x=abs(t_value), df=n_VP - 2)) * 2
            p_matrix[f, time] = p

    #  plot 2D correlation matrix of observed data
    fig = plt.figure()
    ax = sns.heatmap(corr_matrix)
    ax.invert_yaxis()  # to have lowest freq at the bottom
    plt.show()

    # keep only sig cluster
    # set all p-values to 0 where p-value is > .05
    p_matrix = np.where(p_matrix < threshold, p_matrix, 0)

    # label features: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    cluster = ndimage.label(p_matrix)
    cluster_matrix = cluster[0]
    n_cluster = cluster[1]

    # sum t values of individual clusters
    input_ndi = t_matrix
    labels = cluster_matrix
    index = np.arange(1, n_cluster, 1)
    t_sum_all_observed = ndimage.sum_labels(input_ndi, labels, index)

    def sort_abs(arr):
        return sorted([abs(x) for x in arr])

    if t_sum_all_observed is not None:
        # sort t values of individual clusters
        t_sum_all_observed_abs = sort_abs(t_sum_all_observed)

        observed_T = t_sum_all_observed_abs[-1]

        if len(t_sum_all_observed_abs) > 1:
            observed_T_2 = t_sum_all_observed_abs[-2]
        else:    
            observed_T_2 = 0
    else:
        print("no clusters found")

    # now calculate big T for permuted behavioral data (correlation matrix)
    # https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/
    # preallocate array
    T_distribution = np.zeros(n_permutations)
    
    for i, shuffle in enumerate(T_distribution):
        np.random.shuffle(behavior)  # shuffle behavior values randomly

        for f in range(n_freq):
            for time in range(n_time):
                ERD = X[:, f, time]
                r = np.corrcoef(ERD, behavior)[0, 1]  # pearson correlation

                t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
                t_matrix[f, time] = t_value

                p = (1 - stats.t.cdf(x=abs(t_value), df=n_VP - 2)) * 2
                p_matrix[f, time] = p

        # keep only sig cluster
        # set all t-values to 0 where p-value is > .05
        t_matrix = np.where(p_matrix <= threshold, t_matrix, 0)

        # label features: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        cluster = ndimage.label(t_matrix)
        cluster_matrix_perm = cluster[0]
        n_cluster = cluster[1]

        # sum t values of individual clusters
        input_ndi = t_matrix
        labels = cluster_matrix_perm
        index = np.arange(1, n_cluster, 1)
        t_sum_all = ndimage.sum_labels(input_ndi, labels, index)

        t_sum_all_abs = abs(t_sum_all)

        cluster_mass = t_sum_all_abs.max()

        T_distribution[i] = cluster_mass

    return corr_matrix, cluster_matrix, observed_T, T_distribution