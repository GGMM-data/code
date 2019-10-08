from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    data = np.ones((3, 4), dtype='i')
    comm.Send([data, MPI.INT], dest=1, tag=10)
    print("data has sent.")
else:
    data = np.empty((3, 4), dtype='i')
    data = comm.Recv([data, MPI.INT], source=0, tag=10)
    print("data has been receieved.")

