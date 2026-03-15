

def run(args):
	main(args.data_dir, args.alignment, args.save_dir, args.verbose)


def main(data_dir, alignment, save_dir, verbose_flag):
	import numpy as np
	import random
	import os
	import paf_reader as pf
	import signal_reader as sg
	import event_detection as ed
	import util
	import event_io as io
	from tqdm import tqdm
	from constant import DataConst as const

	alignment_keys = pf.read_ids(paf)

	fnames = list()
	for dir in data_dir:
		fnames += util.signalfile_in(dir)
	iters = tqdm(fnames) if verbose_flag else fnames

	for fname in iters:
		signals = sg.read_signal(fname)
		signal_keys = set(signals.keys())
		keys = alignment_keys & signal_keys
		events_data = dict()
		for key in keys:
			signal = signals[key]
			events = ed.detect_events(util.normalization(signal))
			if len(events) < 1:
				continue
			elif len(events) <= const.data_size:
				padding_size = const.data_size - len(events)
				events_data[key] = np.pad(events, ((0, padding_size), (0, 0)), "constant", constant_values = const.mask_value)
			else:
				istart = np.random.randint(0, len(events) - const.data_size + 1)
				events_data[key] = events[istart:istart + const.data_size]
		out = "{0:s}/{1:s}.{2:s}".format(save_dir.rstrip('/'), os.path.splitext(os.path.basename(fname))[0], "event5")
		io.dump_events(events_data, out)
