

def read_ids(fname):
	retval = set()
	with open(fname, 'r') as fin:
		for line in fin:
			words = line.split('\t')
			retval.add(words[0])
	return retval


def read_all(fname):
	retval = dict()
	with open(fname, 'r') as fin:
		for line in fin:
			words = line.split('\t')
			retval[words[0]] = words[1:]
	return retval


def label_reads(fname, label):
	retval = dict()
	with open(fname, 'r') as fin:
		for line in fin:
			words = line.split('\t')
			retval[words[0]] = label
	return retval
