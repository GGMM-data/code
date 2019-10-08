import mpi4py
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

recv_data = None
if rank == 1:
    send_data = range(size) 
    print("data bcast to others")
else:
    send_data = None

recv_data = comm.scatter(send_data, root=1)
print("process {} has received data {}".format(rank, recv_data))

