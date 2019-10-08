from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    data = {'name': "mxx", "age": 23}
    comm.send(data, dest=1, tag=10)
    print("data has sent.")
else:
    data = comm.recv(source=0, tag=10)
    print("data has been receieved.")

