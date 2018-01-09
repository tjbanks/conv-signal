from sig2learn import Signal

matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"

s = Signal().load_mat(matfile,matvar).replace_with_cutout(1000,1200).append_bandpass_filter(20,25).replace_with_hilbert_transform().signals[1].testprint()
