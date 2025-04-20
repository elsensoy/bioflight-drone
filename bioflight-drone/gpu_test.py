import torch

print("=== GPU CHECK ===")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))
    x = torch.rand(10000, 10000).cuda()
    print("Matrix operation test (10K x 10K)...")
    torch.mm(x, x)
    print("Matrix operation succeeded on GPU.")
else:
    print("No GPU found. Running on CPU.")

'''
-build image
docker build -t bioflight-gpu .


-container Run
docker run --gpus all -it --rm -v $(pwd):/bioflight-drone bioflight-gpu

or use docker-compose
docker-compose up --build


inside the container 
python3 gpu_test.py
'''