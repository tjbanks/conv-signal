import sig
from sig import Signal

matfile = "LFP_QW_long_tuning7_synr1.mat"
matvar = "LFP_array_all_sum_afterHP"

s = Signal(filename=matfile, matvarname=matvar)
