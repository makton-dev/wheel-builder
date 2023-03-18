import os

pytorch_version = None
python_version = None
cuda_version = None
arch = None
processor = None
torch_wheel = None

file_list = os.listdir("./wheels/")
if len(file_list) < 5:
    print("One of the wheels is missing. Unable to benchmark")

for file in file_list:
    if "torch-" in file:
        torch_wheel = file
        break

if torch_wheel is None:
    print("Missing Torch wheel.. Exiting..")
    exit(1)

data = torch_wheel.split('-')
pytorch_version = data[1].split('+')[0]
python_version = data[2].replace('cp','')
processor = data[1].split('+')[1]
if "aarch64" in data[4]:
    processor = "graviton"
elif processor != "cpu":
    cuda_version = processor.replace('cu','')
    processor = "gpu"


print(f"Python Version = {python_version}\n"
      f"PyTorch Version = {pytorch_version}\n"
      f"ML Processor = {processor}\n"
      f"Cuda Version = {cuda_version}")