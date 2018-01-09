from sig2learn import Signal

matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"

s = Signal().load_mat(matfile,matvar).signals[0].testprint()
