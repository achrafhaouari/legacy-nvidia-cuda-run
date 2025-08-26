Running Python on Nvidia Geforce 740M on a docker Container
System : Ubuntu 24.04LTS
CuPy version: 9.6.0
CUDA runtime version: 10.2
GPU name: NVIDIA GeForce GT 740M (Legacy)

The different codes shows some benchmarks on the compute capability with large matrices (~10k x 10k) to see a clear GPU speedup, even for legacy GPUs

Notes:

  Use cp.random.rand() to avoid CPU→GPU transfer overhead.
  cp.cuda.Stream.null.synchronize() ensures the GPU finishes before measuring time.
  If N=4000 uses too much memory (~256 MB per matrix, 3 matrices = 768 MB), reduce to N=2000–3000.
  For even heavier workloads, you can do matrix powers or repeated multiplications.

Setup:
1- Download repo
2- run :  docker build -t your_docker_img_name .
3- ensure your "Run/Debug Config" on "Docker container setting" is at minimum: 
    --gpus all
    or
    --entrypoint= --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it

Results example : gpu_test.py

Test 1 :  matrix_size = 10000 x 10000 

  1. Running on CPU with NumPy...
     CPU Time: 23.1901 seconds
  
  2. Running on GPU with CuPy...
     GPU Time: 6.7922 seconds
  
  ✅ Verification successful: CPU and GPU results match.
  🚀 GPU was approximately 3.41 times faster than the CPU.

Test 2 :  matrix_size = 1000 x 1000 
  
  1. Running on CPU with NumPy...
     CPU Time: 0.0171 seconds
  
  2. Running on GPU with CuPy...
     GPU Time: 0.1233 seconds

smaller matrix size show that it's not neccessary to even bather with your Legacy CUDA GPU Compute Capability
