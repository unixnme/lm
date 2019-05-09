from collections import defaultdict

def unpack(data:list, batch_sizes:list):
    '''
    unpack given packed data
    '''
    batch = defaultdict(list)
    j = 0
    for batch_size in batch_sizes:
        for i in range(batch_size):
            batch[i].append(data[j])
            j += 1

    return [b for _,b in sorted(batch.items())]