import mpi4py
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

send_data = rank
print("process {} send data {} to root.".format(rank, send_data))

recv_data = comm.gather(send_data, root=9)
if rank == 9:
    print("process {} gather all data {} to others.".format(rank, recv_data))

