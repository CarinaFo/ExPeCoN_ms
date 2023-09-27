# run cluster test on sensor space data and plot contrast in source space using eLoreta

# not cleaned yet

def extract_cluster_and_plot_source_contrast(clusters=None, good_cluster_inds=None,
                                             evoked_dataa=None, evoked_datab=None,
                                             evokeds=None):

    """extract cluster timepoints and channels from cluster test in sensor space and put contrast into source space
    using eLoreta"""

    # extract cluster channels and timepoints from significant clusters in sensor space and store difference per participant
    cluster_zero = clusters[good_clusters_idx[1]] # TPJ
    cluster_one = clusters[good_clusters_idx[2]] # ACC/frontal pole

    # store unique timepoints and channels per cluster

    timepoint_idx = np.unique(cluster_zero[0])
    channel_idx = np.unique(cluster_zero[1])

    timepoint_idx1 = np.unique(cluster_one[0])
    channel_idx1 = np.unique(cluster_one[1])

    # extract only significant timepoints from evoked data
    high_zero = [h.data[:,timepoint_idx] for h in evoked_dataa]
    low_zero = [h.data[:,timepoint_idx] for h in evoked_datab]
    # calculate difference for both conditions
    diff_zero = [h-l for h,l in zip(high_zero, low_zero)]

    high_one = [h.data[:,timepoint_idx1] for h in evoked_dataa]
    low_one = [h.data[:,timepoint_idx1] for h in evoked_datab]

    diff_one = [h-l for h,l in zip(high_one, low_one)]

    # convert back to EvokedArray for source reconstruction for both cluster
    diff_evo_zero = [mne.EvokedArray(z, evokeds.info) for z in diff_zero]
    diff_evo_one = [mne.EvokedArray(z, evokeds.info) for z in diff_one]

    # source reconstruct the contrast

    for idx, subj in enumerate(IDlist):

        noise_cov = mne.read_cov('D:\expecon_ms\data\eeg\source\cluster_source\covariance_prestim.cov')

        inv_op = mne.minimum_norm.make_inverse_operator(diff_evo_zero[0].info, fwd, noise_cov)

        diff_evo_zero[idx].set_eeg_reference(projection=True)  # needed for inverse modeling

        cluster_0 = mne.minimum_norm.apply_inverse(diff_evo_zero[idx], inv_op, lambda2=0.05,
                                                    method='eLORETA', pick_ori='normal')
    
        os.chdir("D:\expecon_ms\data\eeg\source\cluster_source")

        cluster_0.save('cluster0_' + subj)
    
    # load and plot the stc contrast

    stc_all = []

    for idx, subj in enumerate(IDlist):

        os.chdir("D:\expecon_ms\data\eeg\source\cluster_source")

        conditiona_stc = mne.read_source_estimate('cluster1_' + subj)

        stc_all.append(conditiona_stc)

    stc_array = np.array([np.array(s.data) for s in stc_all])

    # average over time and participants
    X_avg = np.mean(stc_array[:,:,:], axis=(0))

    fsave_vertices = [s['vertno'] for s in src]

    # create source estimate

    stc = mne.SourceEstimate(X_avg, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices)

    pos, latency = stc.get_peak(hemi='rh', vert_as_index=True, time_as_index=True)

    peak_vertex_surf = stc.rh_vertno[pos]

    # plot source contrast
    brain = stc.plot(
        hemi='both', views='medial', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=False, background='white')
    # save source contrast image
    brain.save_image()