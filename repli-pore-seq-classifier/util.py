
def signalfile_in(dirpath):
	import os
	files = os.listdir(dirpath)
	retval = [dirpath + "/" + file for file in files if file.endswith(".fast5") or file.endswith(".pod5")]
	return retval


def eventfile_in(dirpath):
	import os
	files = os.listdir(dirpath)
	retval = [dirpath + "/" + file for file in files if file.endswith(".event5")]
	return retval


def normalization(signal):
	q20, q90 = np.quantile(signal, [0.2, 0.9])
	shift = max(10, 0.51 * (q20 + q90))
	scale = max(1.0, 0.53 * (q90 - q20))
	return (signal - shift) / scale
