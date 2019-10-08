import mpi4py
from mpi4py import MPI


comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = ccomm.Get_rank()

print("============")
print(size)
print(rank)
