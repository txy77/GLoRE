from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.all_data = None  # all data
        self.all_problems = None
        self.all_inputs = None
        self.all_labels = None
        self.load_data()

    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        return self.all_data[index]