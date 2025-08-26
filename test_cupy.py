import cupy as cp

print("CuPy version:", cp.__version__)

# Show CUDA runtime version (works in CuPy 9.x)
print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())

# Show GPU device info
dev = cp.cuda.Device(0)
print("GPU name:", cp.cuda.runtime.getDeviceProperties(dev.id)['name'])

x = cp.arange(10)
print("Array on GPU:", x * 2)
