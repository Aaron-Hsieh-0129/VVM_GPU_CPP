#ifndef VVM_IO_IOSERVER_HPP
#define VVM_IO_IOSERVER_HPP
#include <mpi.h>
namespace VVM {
namespace IO {
    void run_io_server(MPI_Comm io_comm);
}
}
#endif
