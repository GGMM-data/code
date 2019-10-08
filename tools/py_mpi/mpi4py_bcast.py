import mpi4py
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 1:
    data = {"name": "mxx", "age": 23}
    print("data bcast to others")
else:
    data = None

data = comm.bcast(data, root=1)
print("process {} has received data".format(rank))

