import h5py
import sys
import numpy as np


def read_all(fname):

	data_shape = ...
	retval = dict()
	labels = ...
	signals = ...

	with h5py.File(fname, 'r') as fin:
		data_shape = fin["DataSize/shape"]
		labels = np.zeros(data_shape[0])
		signals = np.zeros((data_shape[0], data_shape[1], data_shape[2]))

		keys = list(fin["DataSet"].keys())

		for idx, key in enumerate(keys):
			signal = fin["DataSet/{:s}/signal".format(key)]
			signals[idx, :, :] = signal
			label  = fin["DataSet/{:s}/label".format(key)]
			labels[idx] = label[0]
	
	retval["signal"] = signals
	retval["label"]  = labels
	
	return retval


def read_id_and_signal(fname):

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


def split_dataset(fname, test_ratio = 0.2):

	if (test_ratio <= 0.0) | (test_ratio >= 1.0):
		print("The value in 'test_ratio' is invalid. It should be between 0 < r < 1")
		sys.exit()

	data = read_all(fname)
	signals = data["signal"]
	labels  = data["label"]
	data_size = len(labels)
	retval = dict()
	
	shuffle_indices = np.random.choice(data_size, data_size, replace = False)
	train_size = int(data_size * (1 - test_ratio))
	train_indices = shuffle_indices[:train_size]
	test_indices = shuffle_indices[train_size:]
	retval["train"] = dict()
	retval["test"] = dict()
	retval["train"]["signal"] = signals[train_indices]
	retval["train"]["label"]  = labels[train_indices]
	retval["test"]["signal"]  = signals[test_indices]
	retval["test"]["label"]   = labels[test_indices]
	return retval



if __name__ == "__main__":

	from matplotlib import pyplot as plt

	fname = sys.argv[1]
	#data = read_all(fname)
	data = split_dataset(fname)
	idx = 0
	test_signal = data["train"]["signal"][idx]
	test_label  = data["train"]["label"][idx]
	reconst_mean = list()
	reconst_stdv = list()

	for raw in test_signal:
		length = int(raw[0])
		reconst_mean += [raw[1] for _ in range(length)]
		reconst_stdv += [raw[2] for _ in range(length)]
	
	reconst_mean = np.array(reconst_mean)
	reconst_stdv = np.array(reconst_stdv)
	print("label: {:d}".format(int(test_label)))

	fig, ax = plt.subplots()
	xs = np.arange(len(reconst_mean))
	ax.plot(xs, reconst_mean, color = "red")
	ax.fill_between(xs, reconst_mean + reconst_stdv, reconst_mean - reconst_stdv, color = "red", alpha = 0.2)
	#ax.set_xlim(5000, 6000)
	plt.show()
