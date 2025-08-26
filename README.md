###Running Python on Nvidia Geforce 740M on a docker Container<br/>
##System : Ubuntu 24.04LTS<br/>
##CuPy version: 9.6.0<br/>
##CUDA runtime version: 10.2<br/>
##GPU name: NVIDIA GeForce GT 740M (Legacy)<br/>

The different codes shows some benchmarks on the compute capability with large matrices (~10k x 10k) to see a clear GPU speedup, even for legacy GPUs<br/>

#Notes:<br/>

  Use cp.random.rand() to avoid CPU→GPU transfer overhead.<br/>
  cp.cuda.Stream.null.synchronize() ensures the GPU finishes before measuring time.<br/>
  If N=4000 uses too much memory (~256 MB per matrix, 3 matrices = 768 MB), reduce to N=2000–3000.<br/>
  For even heavier workloads, you can do matrix powers or repeated multiplications.<br/>

##Setup:<br/>
1- Download repo<br/>
2- run :  docker build -t your_docker_img_name .<br/>
3- ensure your "Run/Debug Config" on "Docker container setting" is at minimum: <br/>
    --gpus all<br/>
    or<br/>
    --entrypoint= --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it<br/>

##Results example : gpu_test.py<br/>

#Test 1 :  matrix_size = 10000 x 10000 <br/>

  1. Running on CPU with NumPy...<br/>
     CPU Time: 23.1901 seconds<br/>
  
  2. Running on GPU with CuPy...<br/>
     GPU Time: 6.7922 seconds<br/>
  
  ✅ Verification successful: CPU and GPU results match.<br/>
  🚀 GPU was approximately 3.41 times faster than the CPU.<br/>

#Test 2 :  matrix_size = 1000 x 1000 <br/>
  
  1. Running on CPU with NumPy...<br/>
     CPU Time: 0.0171 seconds<br/>
  
  2. Running on GPU with CuPy...<br/>
     GPU Time: 0.1233 seconds<br/>

smaller matrix size show that it's not neccessary to even bather with your Legacy CUDA GPU Compute Capability<br/>
