from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


def get_dataset(dataset_name: str):
    if dataset_name == "cora":
        return CoraGraphDataset()
    elif dataset_name == "cite":
        return CiteseerGraphDataset()
    elif dataset_name == "pubmed":
        return PubmedGraphDataset()
    else:
        raise ValueError(f"No dataset corresponds to {dataset_name}")


if __name__ == "__main__":
    print(get_dataset("cora"))
