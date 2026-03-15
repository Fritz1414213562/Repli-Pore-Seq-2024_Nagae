import numpy as np
from constant import DataConst as const
import sys

def _read_fast5(fname):
	import h5py
	import hdf5plugin
	retval = dict()
	with h5py.File(fname, 'r', swmr = False) as fin:
		for read in fin:
#			signal = _get_signal_in_fast5(fin[read])
			dset = fin[read]["Raw"]["Signal"]
			signal = np.empty(dset.shape, dset.dtype)
			dset.read_direct(signal)
			read_id = read.lstrip('read_')
			retval[read_id] = signal[(const.signal_outlier_lim[0] <= signal) & (signal <= const.signal_outlier_lim[1])]
	return retval

def _read_pod5(fname):
	import pod5
	retval = dict()
	with pod5.Reader(fname) as reader:
		for read_record in reader.reads():
			key = str(read_record.read_id)
			signal = read_record.signal
			retval[key] = signal[(const.signal_outlier_lim[0] <= signal) & (signal <= const.signal_outlier_lim[1])]
	return retval


def read_signal(fname):
	if fname.endswith(".fast5"):
		return _read_fast5(fname)
	elif fname.endswith(".pod5"):
		return _read_pod5(fname)
	else:
		print("Warning: The format of {:s} is not supported. Please check or change file suffix.", file = sys.stderr)
