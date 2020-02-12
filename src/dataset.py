import torch
from torch.utils.data import DataLoader, Dataset


class SampleDataset(Dataset):
    def __init__(self, CONFIG, mode):
        assert mode in ("train", "val", "test")
        self.CONFIG = CONFIG
        self.data = []
        # sample data
        for id in range(100):
            self.data.append(
                {"id": id, "hoge": torch.randn(100), "label": torch.randn(1)}
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            {
                'id':           int;            the id of hoge
                'hoge':         torch.tensor;   features
                'label':        torch.tensor;   label of hoge
            }
        """
        id = self.data[index]["id"]
        hoge = self.data[index]["hoge"]
        label = self.data[index]["label"]
        return {"id": id, "hoge": hoge, "label": label}

    def __len__(self):
        return len(self.data)


def get_collater(mode):
    assert mode in ("train", "val", "test")

    def collater(datalist):
        ids = []
        hoges = []
        labels = []
        for data in datalist:
            ids.append(data["id"])
            hoges.append(data["hoge"])
            labels.append(data["label"])
        hoges = torch.stack(hoges)
        labels = torch.stack(labels)

        return {"id": ids, "hoge": hoges, "label": labels}

    return collater


# for debugging
if __name__ == "__main__":
    mode = "train"
    ds = SampleDataset(mode)
    loader = DataLoader(ds, batch_size=4, collate_fn=get_collater(mode))
    for data in loader:
        print(data)
        break
