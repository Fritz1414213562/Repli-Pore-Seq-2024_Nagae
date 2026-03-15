

def run(args):
	main(args.event_dir, args.alignment, args.model_dir, args.output)


def classify(data, model_dir):
	import tensorflow as tf
	from tensorflow import keras

	round = lambda x: np.round((2*x + 1) // 2)
	model = keras.models.load_model(model_dir)
	predicted = model.predict(data["signal"])
	predicted = round(predicted).astype(int)
	predicted = predicted.reshape(len(predicted))
	read_ids = data["read_id"]
	retval = dict()
	for idx, key in enumerate(read_ids):
		retval[key] = predicted[idx]

	return retval


def main(event_dir, alignment, model_dir, output_prefix):

	import os
	import event_io as io
	import paf_reader as pf
	import util

	fnames = util.eventfile_in(event_dir)
	predicted_labels = dict()
	for fname in fnames:
		data = io.read_events(fname)
		pred = classify(data, model_dir)
		predicted_labels.update(pred)

	datanum = len(predicted_labels)
	label_values = np.array(list(predicted_labels.values()))
	label_kinds  = list(set(predicted_labels.values()))
	labeled_nums = list()

	for label_kind in label_kinds:
		labeled_nums.append(np.count_nonzero(label_values == label_kind))
	with open(output_prefix + ".log", 'w') as ofs:
		ofs.write("Total: {:d}\n".format(datanum))
		for labeled_num, label_kind in zip(labeled_nums, label_kinds):
			ofs.write("Label{:d}: {:d}\n".format(label_kind, labeled_num))

	paf_data = pf.read_all(alignment)
	for label_kind in label_kinds:
		output_name = output_prefix + "_label{:d}.paf".format(label_kind)
		with open(output_name, 'w') as ofs:
			for key, value in predicted_labels.items():
				if value == label_kind:
					ofs.write(key + "\t" + '\t'.join(paf_data[key]))

