from argparse import ArgumentParser
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from lib.generate_data import GenerateData
from lib.benchmark import Benchmark

def main():
    parser = ArgumentParser(description="MLTool v0.1")

    modes = [
        'generate-data',
        'train',
        'benchmark'
    ]

    parser.add_argument("--ratio", help="Ratio for training", type=float, default=0.8)
    parser.add_argument("--input-file", help="Input CSV file", type=str)
    parser.add_argument("--output-x-train-file", help="Output x train file", type=str, default="x_train.csv")
    parser.add_argument("--output-y-train-file", help="Output y train file", type=str, default="y_train.csv")
    parser.add_argument("--output-x-test-file", help="Output x test file", type=str, default="x_test.csv")
    parser.add_argument("--output-y-test-file", help="Output y test file", type=str, default="y_test.csv")
    parser.add_argument("--mode", help="Type of action for MLTool", type=str, choices=modes, required=True)
    parser.add_argument("--input-model-file", help="Input model file", type=str)
    parser.add_argument("--input-test-file", help="Input test file", type=str)
    parser.add_argument("--input-labels-file", help="Input labels file", type=str)

    args = parser.parse_args()

    mode                = args.mode
    ratio               = args.ratio
    test_size           = 1 - ratio
    input_file          = args.input_file
    output_x_train_file = args.output_x_train_file
    output_y_train_file = args.output_y_train_file
    output_x_test_file  = args.output_x_test_file
    output_y_test_file  = args.output_y_test_file

    # For benchmark
    input_model_file    = args.input_model_file
    input_test_file     = args.input_test_file
    input_labels_file   = args.input_labels_file

    if mode == 'generate-data':
        # Perform generate data
        params = {
            'ratio':                ratio,
            'test_size':            test_size,
            'input_file':           input_file,
            'output_x_train_file':  output_x_train_file,
            'output_y_train_file':  output_y_train_file,
            'output_x_test_file':   output_x_test_file,
            'output_y_test_file':   output_y_test_file
        }

        cmd = GenerateData(params)
        cmd.run()

    elif mode == 'train':
        # Perform train mode (See: Example 2 notebook)
        pass

    elif mode == 'benchmark':
        # Perform benchmark mode (See: Example 3 notebook)
        params = {
            'input_model_file':     input_model_file,
            'input_test_file':      input_test_file,
            'input_labels_file':    input_labels_file
        }

        cmd = Benchmark(params)
        cmd.run()

    print("Done.")

if __name__ == '__main__':
    main()
