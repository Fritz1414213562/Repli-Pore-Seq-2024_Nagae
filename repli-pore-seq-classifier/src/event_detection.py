import numpy as np
import h5py
import sys


class EventDetector():

	DEF_PEAK_POS = -1
	DEF_PEAK_VAL = sys.float_info.max

	def __init__(self, tstats, threshold, window_length):
		self.signal = tstats
		self.signal_length = len(tstats)
		self.threshold = threshold
		self.window_length = window_length
		self.masked_to = 0
		self.peak_pos = -1
		self.peak_value = sys.float_info.max
		self.valid_peak = False


class Event():

	def __init__(self, start, length, mean, stdv):
		self.start = start
		self.length = length
		self.mean = mean
		self.stdv = stdv


class Params():

	short_detector_threshold = 3
	short_detector_windowsize = 6
	long_detector_threshold = 1.4
	long_detector_windowsize = 9.0
	peak_height = 0.2


def __read_fast5(fast5_name):

	retval = dict()

	with h5py.File(fast5_name, 'r') as fin:
		seq_ids = list(fin.keys())

		for seq_id in seq_ids:
			key = seq_id.lstrip("read_")
			retval[key] = np.array(fin[seq_id + "/Raw/Signal"], dtype = np.float32)
	
	return retval


def __compute_tstat(cumsum, csumsq, windowsize: int):

	if len(cumsum) != len(csumsq):
		print("The vector length of cumsum is not consistent with that of cumurative sum of squared signal", file = sys.stderr)
		sys.exit()

	data_size = len(cumsum) - 1
	tstats = np.zeros(data_size)

	if (data_size < 2 * windowsize) | (windowsize < 2):
		return tstats
	
	for iwin in range(windowsize, data_size - windowsize + 1):

		pre_sum = cumsum[iwin]
		pre_ssq = csumsq[iwin]
		if iwin > windowsize:
			pre_sum -= cumsum[iwin - windowsize]
			pre_ssq -= csumsq[iwin - windowsize]
		post_sum = cumsum[iwin + windowsize] - cumsum[iwin]
		post_ssq = csumsq[iwin + windowsize] - csumsq[iwin]
		comb_var = pre_ssq / windowsize - (pre_sum / windowsize) ** 2 + post_ssq / windowsize - (post_sum / windowsize) ** 2
		#comb_var = min(sys.float_info.min, comb_var)
		comb_var = max(sys.float_info.min, comb_var)
		tstats[iwin] = np.abs(post_sum / windowsize - pre_sum / windowsize) / np.sqrt(comb_var / windowsize)
	
	return tstats


def __detect_peak(short_detector, long_detector, peak_height):

	detectors = {"short": short_detector, "long": long_detector}

	if short_detector.signal_length != long_detector.signal_length:
		print("Signal length is not the same between short- and long-detector", file = sys.stderr)
		sys.exit()

	retval = list()
	for idx in range(short_detector.signal_length):
		for key, detector in detectors.items():
			if detector.masked_to >= idx:
				continue

			current_value = detector.signal[idx]

			if detector.peak_pos == detector.DEF_PEAK_POS:
				if current_value < detector.peak_value:
					detector.peak_value = current_value
				elif current_value - detector.peak_value > peak_height:
					detector.peak_value = current_value
					detector.peak_pos = idx
			else:
				if current_value > detector.peak_value:
					detector.peak_value = current_value
					detector.peak_pos = idx
				if key == "short":
					if detector.peak_value > detector.threshold:
						long_detector.masked_to  = detector.peak_pos + detector.window_length
						long_detector.peak_pos   = long_detector.DEF_PEAK_POS
						long_detector.peak_value = long_detector.DEF_PEAK_VAL
						long_detector.valid_peak = False
				if (detector.peak_value - current_value > peak_height) & (detector.peak_value > detector.threshold):
					detector.valid_peak = True
				if detector.valid_peak & ((idx - detector.peak_pos) > detector.window_length / 2):
					retval.append(detector.peak_pos)
					detector.peak_pos = detector.DEF_PEAK_POS
					detector.peak_value = current_value
					detector.valid_peak = False

	return np.array(retval, dtype = int)


def __create_event(cumsum, csumsq, istart, iend):

	length = iend - istart
	mean = (cumsum[iend] - cumsum[istart]) / length
	stdv = np.sqrt(max((csumsq[iend] - csumsq[istart]) / length - mean ** 2, 0.0))
	#return Event(istart, length, mean, stdv)
	return np.array([length, mean, stdv])


def __create_events(peaks, cumsum, csumsq):

	data_size = len(cumsum) - 1
	event_num = len(peaks)
	end = event_num

	retval = np.zeros((event_num, 3))
	retval[0, :] = __create_event(cumsum, csumsq, 0, peaks[0])
	for ipeak in range(1, event_num - 1):
		retval[ipeak, :] = __create_event(cumsum, csumsq, peaks[ipeak - 1], peaks[ipeak])
	retval[event_num - 1, :] = __create_event(cumsum, csumsq, peaks[event_num - 2], data_size)

	return retval


def detect_events(signal):

	params = Params()

	cumsum = np.cumsum(np.concatenate([[0], signal]))
	csumsq = np.cumsum(np.concatenate([[0], signal ** 2]))
	tstats1 = __compute_tstat(cumsum, csumsq, int(params.short_detector_windowsize))
	tstats2 = __compute_tstat(cumsum, csumsq, int(params.long_detector_windowsize))
	short_detector = EventDetector(tstats1, params.short_detector_threshold, int(params.short_detector_windowsize))
	long_detector  = EventDetector(tstats2, params.long_detector_threshold, int(params.long_detector_windowsize))
	peaks = __detect_peak(short_detector, long_detector, params.peak_height)
	events = __create_events(peaks, cumsum, csumsq)

	return events


def __test(fname):

	from matplotlib import pyplot as plt
	from util import normalization

	signals = __read_fast5(fname)
	keys = list(signals.keys())
#	data = list()
#	for isignal in range(10):
#		key = keys[isignal]
#		events = detect_events(signals[key])
#		data.append(events)
#		print("Key:", key)
#		print("Event number:", len(events))

	isignal = 0
	key = keys[isignal]
	signal = normalization(signals[key])
	events = detect_events(signal)
	print("Key:", key)
	print("Event number:", len(events))
	reconst = list()
	stdvs   = list()
	for event in events:
		reconst += [event[1] for _ in range(int(event[0]))]
		stdvs   += [event[2] for _ in range(int(event[0]))]


	fig, ax = plt.subplots()
	fig.subplots_adjust(left = 0.2)
	xs = np.arange(len(signal)) + 1
	reconst = np.array(reconst)
	stdvs   = np.array(stdvs)
	ax.plot(xs, signal, color = "grey", label = "Raw")
	ax.plot(xs, reconst, color = "red", label = "Fitting")
#	ax.fill_between(xs, reconst + stdvs, reconst - stdvs, color = "red", alpha = 0.5)
	ax.set_xlabel("Time", fontsize = 20)
	xticks = np.arange(0, 15001, 10000)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xticks, fontsize = 16)
	ax.set_xlim(5000, 7000)
	ax.set_ylabel("Normalized Currents", fontsize = 20)
	yticks = np.arange(-4.0, 2.1, 1.0)
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks, fontsize = 16)
	ax.set_ylim(-3.5, 2.5)
	ax.legend(loc = "lower center", bbox_to_anchor = (0.5, 1.0), ncol = 2, fontsize = 14)
	plt.show()


def __count_events(fast5dirs, out):

	from util import normalization, filesindir
	from fast5reader import read_fast5_file
	from tqdm import tqdm

	fnames = list()
	for dir in fast5dirs:
		fnames += filesindir(dir)
	
	retval = list()
	for fname in tqdm(fnames):
		signals = read_fast5_file(fname)
		keys = set(signals.keys())
		for key in keys:
			events = detect_events(normalization(signals[key]))
			retval.append(len(events))
	retval = np.array(retval)
	with open(out, 'w') as fout:
		for val in retval:
			fout.write("{:10d}\n".format(val))


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", '-i', nargs = '*', required = True)
	parser.add_argument("--output", '-o', required = True)
	args = parser.parse_args()

#	__test(sys.argv[1])
	__count_events(args.input, args.output)
