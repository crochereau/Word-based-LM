def prepare_dataset_chunks(data, train=True):
    numeric = [0]
    count = 0
    print("Prepare chunks")
    numerified = []
    for chunk in data:
        for char in chunk:
            count += 1
            numerified.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))

        if len(numerified) > (args.batchSize*args.sequence_length):
            sequence_length_here = args.sequence_length

            cutoff = int(len(numerified)/(args.batchSize*sequence_length_here)) * (args.batchSize*sequence_length_here)
            numerified_current = numerified[:cutoff]
            numerified = numerified[cutoff:]

            numerified_current = torch.LongTensor(numerified_current).view(args.batchSize, -1, sequence_length_here).transpose(0,1).transpose(1,2).to(device)
            number_of_sequences = numerified_current.size()[0]
            for i in range(number_of_sequences):
                yield numerified_current[i]
            hidden = None
        else:
            print("Skipping")

def prepare_dataset_chunks_previous(data, train=True):
    numeric = [0]
    count = 0
    print("Prepare chunks")
    for chunk in data:
        print(len(chunk))
        for char in chunk:
            if char == " ":
                continue
            count += 1
            #         if count % 100000 == 0:
            #             print(count/len(data))
            numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
            if len(numeric) > args.sequence_length:
                yield numeric
                numeric = [0]

def prepare_dataset(data, train=True):
    numeric = [0]
    count = 0
    for char in data:
        if char == " ":
            continue
        count += 1
        numeric.append((stoi[char]+3 if char in stoi else 2) if (not train) or random.random() > args.char_noise_prob else 2+random.randint(0, len(itos)))
        if len(numeric) > args.sequence_length:
            yield numeric
            numeric = [0]

