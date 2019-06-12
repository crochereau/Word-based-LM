import torch


def prepare_dataset_chunks(data, stoi, args, device, train=True):
    print("Prepare chunks")
    encoded_sentences = []
    for chunk in data:
        for word in chunk:

            # In eval or test condition, introduce a random word from the vocabulary
            # FIXME: as args.char_noise_prob = 0, introduces noise all the time?
            # FIXME: What is a good value for args.char_noise_prob? Which should be called noisy_word_prob...
            # Replaced numerified by encoded_sentence
            """
            if train or random.random() > args.char_noise_prob:
                numerified.append(random.randint(3, len(stoi)))
            else:
                # get word index in word-index mapping; returns 2 if OOV word
                numerified.append(stoi.get(word, 2))
            """
            encoded_sentences.append(stoi.get(word, 2))

        if len(encoded_sentences) > (args.batch_size * args.sequence_length):   # 40,001 > 128 * 10
            seq_length = args.sequence_length

            first_mult = (len(encoded_sentences) // (args.batch_size * seq_length)) # 31

            cutoff = first_mult * args.batch_size * seq_length   # 39,680 = 31 * 128 * 10
            selected_sentences = encoded_sentences[:cutoff]   # len = 39,680
            encoded_sentences = encoded_sentences[cutoff:]   # len = 321

            # size of selected sentences: (31, 10, 128)
            selected_sentences = torch.tensor(selected_sentences, dtype=torch.long,
                                    device=device).view(args.batch_size, -1, seq_length).transpose(0, 1).transpose(1, 2)
            number_of_sequences = selected_sentences.size()[0]   # 31

            for i in range(number_of_sequences):
                yield selected_sentences[i]   # size: 10 * 128 = args.sequence_length * args_batch.size
        else:
            print("Skipping")
