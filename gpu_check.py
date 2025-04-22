import torch

def list_gpus():
    if not torch.cuda.is_available():
        print("No GPUs detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e6} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1e6} MB")

if __name__ == "__main__":
    list_gpus()