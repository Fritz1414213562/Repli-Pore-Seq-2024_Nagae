import numpy as np
import h5py


def dump_events(events_data, out):
	from constant import DataConst as const

	with h5py.File(out, 'w') as fout:
		data_num = len(events_data)
		grp1 = fout.create_group("DataSize")
		grp1.create_dataset("shape", data = np.array([data_num, const.data_size, 3]))
		grp2 = fout.create_group("DataSet")
		for key in events_data.keys():
			subgrp = grp2.create_group("{:s}".format(key))
			subgrp.create_dataset("signal", data = events_data[key])


def read_events(fname):
	data_shape = ...
	retval = dict()
	keys = ...
	signals = ...

	with h5py.File(fname, 'r') as fin:
		data_shape = fin["DataSize/shape"]
		signals = np.zeros((data_shape[0], data_shape[1], data_shape[2]))
		keys = list(fin["DataSet"].keys())

		for idx, key in enumerate(keys):
			signal = fin["DataSet/{:s}/signal".format(key)]
			signals[idx, :, :] = signal
	
	retval["signal"] = signals
	retval["read_id"] = keys

	return retval

