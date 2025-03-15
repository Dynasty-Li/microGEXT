import argparse


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-fs", "--frame_size", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument('--cuda', default=True, help='enables cuda for GPU training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--patience', default=5, type=int,
                        help='number of epochs to tolerate no improvement of val_loss')

    parser.add_argument('--dp_rate', type=float, default=0.2,
                        help='dropout rate')

    parser.add_argument("--training", dest='training', action='store_false')
    parser.add_argument('--no-training', '--no_training', dest='training', action='store_false')
    parser.set_defaults(training=True)

    parser.add_argument("--inference", dest='inference', action='store_true')
    parser.add_argument('--no-inference', '--no_inference', dest='inference', action='store_false')
    parser.set_defaults(inference=False)

    parser.add_argument("--calibration", dest='calibration', action='store_true')
    parser.add_argument('--no-calibration', '--no_calibration', dest='calibration', action='store_false')
    parser.set_defaults(calibration=False)

    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    args = parser.parse_args()
    return args
