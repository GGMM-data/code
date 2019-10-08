import mpi4py
from mpi4py import MPI


com = MPI.COMM_WORLD

print(com.Get_rank())
