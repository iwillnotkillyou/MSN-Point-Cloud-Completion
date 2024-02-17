import numpy as np
from tqdm import tqdm
from multiprocessing import Queue
from data_process import kill_data_processes
from shapenet import ShapenetDataProcess
import torch
def data_setup(args, phase, num_workers, repeat):
    if args.dataset == 'shapenet':
        DataProcessClass = ShapenetDataProcess
    # Initialize data processes
    data_queue = Queue(4 * num_workers)
    data_processes = []
    for i in range(num_workers):
        data_processes.append(DataProcessClass(data_queue, args, phase, repeat=repeat))
        data_processes[-1].start()
    return data_queue, data_processes


def test(split, args, its=100, printf = None):
    """ Evaluated model on test set """
    print("Testing....")
    args.model.eval()

    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
    losses = []
    N = len(data_processes[0].data_paths)
    Nb = int(N / args.batch_size)
    if Nb * args.batch_size < N:
        Nb += 1
    # iterate over dataset in batches
    Nb = min(Nb, its)
    for i in tqdm(range(Nb), total=Nb, position=0, leave=True):
        targets, clouds_data = data_queue.get()
        with torch.no_grad():
          loss, dist1, dist2, emd_cost, outputs = args.step(targets, clouds_data, i)

        losses.append(loss)

    kill_data_processes(data_queue, data_processes)
    return np.mean(losses)


def test_on_completion3D(args, download):
    dir = "/content/repo_folder/completion3d/data"
    test_data_queue, test_data_processes = data_setup(args, 'test', args.nworkers,
                                                      repeat=True)
    return test('train', args, 50)
