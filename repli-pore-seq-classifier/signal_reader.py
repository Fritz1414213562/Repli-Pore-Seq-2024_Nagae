import numpy as np
from constant import DataConst as const
import h5py
import pod5
import sys


def _read_fast5(fname):
	retval = dict()
	with h5py.File(fname, 'r') as fin:
		keys = list(fin.keys())
		for key in keys:
			read_id = key.lstrip("read_")
			signal = np.array(fin[key + "/Raw/Signal"])
			retval[read_id] = signal[(const.signal_outlier_lim[0] <= signal) & (signal <= const.signal_outlier_lim[1])]
	return retval


def _read_pod5(fname):
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
		print("The format of {:s} is not supported. Please check or change file suffix.", file = sys.stderr)
		sys.exit()
