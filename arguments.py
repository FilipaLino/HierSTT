import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical spatio-temporal ED demand forecasting.")

    # Add arguments
    parser.add_argument('--train', action='store_true', help='Set the model to training mode')
    parser.add_argument('--test', action='store_true', help='Set the model to testing mode')
    
    parser.add_argument("--data", default="data/dataset.csv", help="Path to the raw activity CSV (schema in docs/DATA.md).")
    
    parser.add_argument('--alpha', default=0.3, type=float, metavar='NAME', help='Weigth of he hierarchical term in the training loss')
    parser.add_argument('--batch', default=32, type=int, metavar='NAME', help='Batch size')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-lr", type=float, default=3e-4, help="OneCycleLR peak.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience.")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--checkpoint", default="checkpoint/model_seed_42_alpha_0_3.pth", help="Explicit checkpoint path.")
    

    args = parser.parse_args()
    if not (args.train or args.test):
        parser.error("Choose at least one of --train / --test.")
    
    return args