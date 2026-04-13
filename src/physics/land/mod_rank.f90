  module rank
#ifdef USE_CUDA
      use nccl
#endif

      implicit none

      public

      integer nsize,myrank


      common/com_mpe/nsize,myrank


! for io quilting
      integer nsize_all, nsize_gfs, nsize_io,          &
              myrank_all,myrank_gfs,myrank_io,         &
              root_gfs,root_io,                        &
#if defined(RSM) && defined(CWB_MPMD)
              MPI_COMM_gfs_all,root_rsm,itag,          &
#else
              root_rsm,itag,                           &
#endif
              MPI_COMM_gfs,MPI_COMM_io,ntag,Ngfs,Nio,MPI_COMM_atm

      common/com_quilt/nsize_all, nsize_gfs, nsize_io, &
              myrank_all,myrank_gfs,myrank_io,         &
              root_gfs,root_io,                        &
#if defined(RSM) && defined(CWB_MPMD)
              MPI_COMM_gfs_all,root_rsm,itag,          &
#else
              root_rsm,itag,                           &
#endif
              MPI_COMM_gfs,MPI_COMM_io,ntag,Ngfs,Nio,MPI_COMM_atm
#ifdef USE_CUDA
      type(ncclUniqueId) :: nccl_id
      type(ncclComm) :: nccl_comm_gfs
#endif
  end module rank
