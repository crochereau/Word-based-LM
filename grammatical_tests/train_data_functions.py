import random

import torch


def prepare_dataset_chunks(data, stoi, args, device, train=True):
    count = 0
    print("Prepare chunks")
    numerified = []
    for chunk in data:
        for char in chunk:
            count += 1
            if (not train) or random.random() > args.char_noise_prob:
                numerified.append(random.randint(3, len(stoi)))
            else:
                numerified.append(stoi.get(char, 2))

        if len(numerified) > (args.batchSize * args.sequence_length):
            seq_length = args.sequence_length

            first_mult = (len(numerified) // (args.batchSize * seq_length))
            cutoff = first_mult * args.batch_size * seq_length
            numerified_current = numerified[:cutoff]
            numerified = numerified[cutoff:]

            numerified_current = torch.LongTensor(numerified_current)\
                .view(args.batch_size, -1, seq_length).transpose(0, 1).transpose(1, 2).to(device)
            number_of_sequences = numerified_current.size()[0]
            for i in range(number_of_sequences):
                yield numerified_current[i]
        else:
            print("Skipping")
