import torch
import torch.utils.data as td


class Dataset(td.Dataset):
    def __init__(self, n):
        self.n = n

    def __getitem__(self, item):
        return item, torch.tensor([14, 88], dtype=torch.int8)

    def __len__(self):
        return self.n


def test_dataset():
    ds = Dataset(50)
    loader = td.DataLoader(ds, num_workers=4, sampler=td.RandomSampler(ds), batch_size=4)
    for epoch in range(5):
        print(f'Epoch {epoch}')
        for i, batch in enumerate(loader):
            print(f'Batch: {batch}')


if __name__ == '__main__':
    from kwg_vqa.task.arg import parse_args
    from kwg_vqa.vqa.kwg_dataset import Dataset as Ds

    args = parse_args(['--dataset', 'dataset/tiny-trainA', '--batch-size', '2', '--yes'])
    ds = Ds(args)
