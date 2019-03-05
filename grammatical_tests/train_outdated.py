
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="german")
parser.add_argument("--load-from", dest="load_from", type=str, default="wiki-german-nospaces-bptt-words-966024846")
#parser.add_argument("--load-from-baseline", dest="load_from_baseline", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)
# parser.add_argument("--batchSize", type=int, default=random.choice([128, 128, 256]))
parser.add_argument("--batchSize", type=int, default=128)
# parser.add_argument("--char_embedding_size", type=int, default=random.choice([100, 200, 300]))
parser.add_argument("--char_embedding_size", type=int, default=200)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=2)
# parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01, 0.05, 0.1]))
parser.add_argument("--weight_dropout_in", type=float, default=0.1)
# parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2]))
parser.add_argument("--weight_dropout_hidden", type=float, default=0.2)
# parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
parser.add_argument("--char_dropout_prob", type=float, default=0)
# parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
parser.add_argument("--char_noise_prob", type = float, default=0.01)
# parser.add_argument("--learning_rate", type = float, default= random.choice([0.8, 0.9, 1.0,1.0,  1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5]))
parser.add_argument("--learning_rate", type=float, default=0.2)
parser.add_argument("--myID", type=int, default=random.randint(0, 1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50, 50, 80]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))

args=parser.parse_args([])
print(args)



learning_rate = args.learning_rate
optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

# FIXME: Only for training
train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(reduction='none', ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


"""
rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).to(device)
#rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
#print(rnn_parameter_names)
rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])
output = torch.nn.Linear(args.hidden_dim, len(itos)+3).to(device)
char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).to(device)
"""
