import os

np = "OMP_NUM_THREADS"
openblas = "OPENBLAS_NUM_THREADS"
mkl = "MKL_NUM_THREADS"

print(os.environ.get(np))
print(os.environ.get(openblas))
print(os.environ.get(mkl))


np = "OMP_NUM_THREADS"
openblas = "OPENBLAS_NUM_THREADS"
mkl = "MKL_NUM_THREADS"

os.environ[np] = "1"
os.environ[openblas] = "1"
os.environ[mkl] = "1"

print(os.environ.get(np))
print(os.environ.get(openblas))
print(os.environ.get(mkl))

