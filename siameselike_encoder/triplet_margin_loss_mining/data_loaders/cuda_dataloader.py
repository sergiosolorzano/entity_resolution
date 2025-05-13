import torch

#move data to cuda
class CudaDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield self.move_to_device(batch)

    def __len__(self):
        return len(self.dataloader)

    def move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self.move_to_device(x) for x in batch)
        elif isinstance(batch, dict):
            return {key: self.move_to_device(value) for key, value in batch.items()}
        else:
            return batch
        
    def check_cuda(self, entity):
        print(f"Devices for Object:")
        for idx, tensor in enumerate(entity):
            print(f"  Tensor {idx}: Device {tensor.device}, Type: {type(tensor)}")
        
def CheckAllInCuda(cuda_dataloader):
    for e in cuda_dataloader:
        print("***Batch type:", type(e))
        if isinstance(e, list):
            for idx, item in enumerate(e):
                if isinstance(item, torch.Tensor):
                    print(f"***Tensor {idx} shape:", item.shape)
        elif isinstance(e, dict):
            for key, value in e.items():
                print(f"***Key: {key}, Tensor shape: {value.shape}")
        break