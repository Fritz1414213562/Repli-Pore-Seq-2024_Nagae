def main():
	import argparse
	import fragmentation as fg
	import classification as cl
	import nn
	import dataset_preparation as pr

	parser = argparse.ArgumentParser(description = "Classification of biotinylated DNA fragments")
	subparsers = parser.add_subparsers()

## Fragmentation
	parser_fg = subparsers.add_parser('fragment', help = "see `fragment -h`")
	parser_fg.add_argument("--data_dir", '-d', nargs = '*', required = True)
	parser_fg.add_argument("--alignment", '-a', required = True)
	parser_fg.add_argument("--save_dir", '-s', required = True)
	parser_fg.add_argument("--verbose", '-v', action = "store_true")
	parser_fg.set_defaults(handler = fg.run)

## Classification
	parser_cl = subparsers.add_parser('classify', help = "see `classify -h`")
	parser_cl.add_argument("--event_dir", '-e', required = True)
	parser_cl.add_argument("--model_dir", '-m', required = True)
	parser_cl.add_argument("--alignment", '-a', required = True)
	parser_cl.add_argument("--output", '-o', required = True)
	parser_cl.set_defaults(handler = cl.run)

## Preparation of dataset
	parser_pr = subparsers.add_parser('preparation', help = "see `preparation -h`")
	parser_pr.add_argument("--data_dir", '-d', required = True, nargs = '*')
	parser_pr.add_argument("--label0", '-0', required = True)
	parser_pr.add_argument("--label1", '-1', required = True)
	parser_pr.add_argument("--output", '-o', required = True)
	parser_pr.add_argument("--verbose", '-v', required = True)
	parser_pr.set_defaults(handler = pr.run)

## Training DNN
	parser_nn = subparsers.add_parser('train', help = "see `train -h`")
	parser_nn.add_argument("--train_set", '-t', required = True)
	parser_nn.add_argument("--output", '-o', required = True)
	parser_nn.set_defaults(handler = nn.run)

	args = parser.parse_args()
	if hasattr(args, 'handler'):
		args.handler(args)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
