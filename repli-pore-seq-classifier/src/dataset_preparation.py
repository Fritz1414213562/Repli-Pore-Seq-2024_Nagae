import numpy as np
import h5py
import sys
from event_detection import detect_events
from signal_reader import read_signal
from paf_reader import label_reads
from util import normalization
from constant import DataConst
from tqdm import tqdm


def run(args):
	main(args.data_dir, args.label0, args.label1, args.output, args.verbose)


def dump(events_data, labels_data, out):

	with h5py.File(out, 'w') as fout:
		data_num = len(events_data)
		grp1 = fout.create_group("DataSize")
		grp1.create_dataset("shape", data = np.array([data_num, DataConst.data_size, 3]))
		grp2 = fout.create_group("DataSet")
		for idx, (label, events) in enumerate(zip(labels_data, events_data)):
			subgrp = grp2.create_group("Instance{:d}".format(idx))
			subgrp.create_dataset("signal", data = events)
			if isinstance(label, int):
				subgrp.create_dataset("label", data = np.array([label]))
			elif isinstance(label, np.ndarray):
				subgrp.create_dataset("label", data = label)
			else:
				print("Warning: The label format is invalid, then recording is skipped.", file = sys.stderr)
				continue


def main(fast5dirs, paf0, paf1, out, verbose_flag):
	import random

	if verbose_flag:
		print("Read Fast5 ....")
	fnames = list()
	for fast5dir in fast5dirs:
		fnames += util.signalfile_in(dir)
	signals = dict()
	for fname in fnames:
		signals.update(read_signal(fname))
	if verbose_flag:
		print("Done")

	signal_keys = set(signals.keys())
	if len(signal_keys) < 1:
		print("No signal input", file = sys.stderr)
		sys.exit()

	if verbose_flag:
		print("Signal number       : {:d}".format(len(signal_keys)))

	label0  = label_reads(paf0, 0)
	label0_keys = list(set(label0.keys()) & signal_keys)
	if verbose_flag:
		print("reads labeled as '0': {:d}".format(len(label0)))
		print("reads overlapped    : {:d}".format(len(label0_keys)))

	label1  = label_reads(paf1, 1)
	label1_keys = list(set(label1.keys()) & signal_keys)
	if verbose_flag:
		print("reads labeled as '1': {:d}".format(len(label1)))
		print("reads overlapped    : {:d}".format(len(label1_keys)))

	labels = dict(label0, **label1)

	if len(label0_keys) < len(label1_keys):
		label1_keys = random.sample(label1_keys, len(label0_keys))
	elif len(label0_keys) > len(label1_keys):
		label0_keys = random.sample(label0_keys, len(label1_keys))
	else:
		pass
	
	keys = label0_keys + label1_keys
	random.shuffle(keys)

	events_data = list()
	labels_data = list()
	iters = tqdm(keys) if verbose_flag else keys

	for key in iters:
		events = detect_events(normalization(signals[key]))
		label = labels[key]
		if len(events) <= DataConst.data_size:
			padding_size = DataConst.data_size - len(events)
			events_data.append(np.pad(events, ((0, padding_size), (0, 0)), "constant", constant_values = DataConst.mask_value))
			labels_data.append(label)
		else:
			remaining_num = len(events)
			istart = 0
			while remaining_num > DataConst.left_trim_size:
				sample_size = np.random.randint(DataConst.left_trim_size, min(DataConst.data_size, remaining_num))
				eventslice = events[istart:istart + sample_size]
				padding_size = DataConst.data_size - sample_size
				events_data.append(np.pad(eventslice, ((0, padding_size), (0, 0)), "constant", constant_values = DataConst.mask_value))
				labels_data.append(label)
				remaining_num -= sample_size
				istart += sample_size


	if verbose_flag:
		print("Output to {:s}".format(out))
	dump(events_data, labels_data, out)



if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", '-d', nargs = '*', required = True)
	parser.add_argument("--label0", '-0', required = True)
	parser.add_argument("--label1", '-1', required = True)
	parser.add_argument("--output", '-o', required = True)
	parser.add_argument("--verbose", '-v', action = "store_true")
	args = parser.parse_args()

	main(args.data_dir, args.label0, args.label1, args.output, args.verbose)
