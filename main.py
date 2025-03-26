import argparse
from training import run_training

def main():
    parser = argparse.ArgumentParser(
        description='Run QFL with a specific method and data distribution.'
    )
    parser.add_argument('method', type=str, 
                        choices=['centralized', 'fedavg', 'fedadam', 'loss_agg', 'loss_agg_adam'],
                        help='The QFL method to run')
    parser.add_argument('-l', '--load', type=str, help='Load from existing run directory (e.g., run_1)')
    parser.add_argument('-d', '--distribution', type=str, 
                        choices=['c1', 'c2', 'iid', 'quantity'], default='c2',
                        help='Data distribution type: c1 (1 class per client), c2 (2 classes per client), iid (homogenous distributed), or quantity (quantity skew)')
    parser.add_argument('-c', '--classes', type=int, choices=[8, 10], default=10,
                        help='Number of classes to use: 8 (classes 0-7) or 10 (all classes)')
    parser.add_argument('-ds', '--dataset', type=str, 
                        choices=['mnist', 'fmnist'], default='mnist',
                        help='Dataset to use: mnist (MNIST digits) or fmnist (Fashion MNIST)')
    args = parser.parse_args()
    
    run_training(args)

if __name__ == "__main__":
    main()
