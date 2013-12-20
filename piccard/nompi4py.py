# Dummy class for packages that have no MPI
class MPIDummy(object):
    def __init__(self):
        pass

    def Get_rank(self):
        pass

    def Get_size(self):
        pass

    def barrier(self):
        pass

    def send(self, lnlike0, dest=1, tag=55):
        pass

    def recv(self, source=1, tag=55):
        pass

    def Iprobe(self, source=1, tag=55):
        pass


# Global object representing no MPI:
COMM_WORLD = MPIDummy()
