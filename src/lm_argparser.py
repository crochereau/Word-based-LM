import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument("--language", default="german", type=str)
parser.add_argument("--load_from", dest="load_from", type=str)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--word_embedding_size", type=int, default=random.choice([100, 200, 300]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([1, 2]))
parser.add_argument("--nonlinearity", default=random.choice(["tanh", "relu"]), type=str)

parser.add_argument("--weight_dropout_in", type=float,
                    default=random.choice([0.0, 0.0, 0.0, 0.01]))
parser.add_argument("--weight_dropout_hidden", type=float,
                    default=random.choice([0.0, 0.05, 0.15, 0.2, 0.3, 0.4]))

parser.add_argument("--char_dropout_prob", type=float,
                    default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_noise_prob", type = float, default=0)
parser.add_argument("--learning_rate", type = float,
                    default= random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))

parser.add_argument("--my_id", type=str, default=str(random.randint(0,1000000000)))
parser.add_argument("--sequence_length", type=int, default=50)
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float,
                    default=random.choice([0.5, 0.6, 0.7, 0.9, 0.95, 0.98, 1.0]))

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                       help='report interval')
parser.add_argument('--save', type=str, default='-model.pt',
                       help='path to save the final model')
parser.add_argument('--log', type=str, default='log.txt',
                       help='path to logging file')
