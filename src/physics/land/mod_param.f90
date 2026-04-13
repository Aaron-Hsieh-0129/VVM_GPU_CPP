      module param

      implicit none

      public

!
!  definition of global model resolution
!  t79 : nx=240,  t120 : nx=360
!  t180: nx=540,  t240 : nx=720
!  t320: nx=960, 
!  t512: nx=1536,

! integer, parameter :: nx=1536 
! integer, parameter :: my=nx/2
! integer, parameter :: lev=60
! integer, parameter :: jtrun= 2*((1+(nx-1)/3)/2)
! integer, parameter :: mlmax= jtrun*(jtrun+1)/2

      integer  nco,nx,my,lev,jtrun,mlmax
      logical  octahedral
      common/comparam1/nco,nx,my,lev,jtrun,mlmax,octahedral

!
! specify the ncld = number of water spieces
!             ncld = 1  : only humidity
!                  = 2  : humidity and mixed cloud as qcirs
!                  = 3  : humidity, qci and qrs  (not yet)
!

!     integer, parameter :: ncld=2
      integer  ncld
      common/comparam2/ncld
!
      real     alpha,mwhd

!
! for nmccup
!      parameter(option_cup=2)
! for nmcpbl
!      parameter(option_pbl=2)

!  specify the number of processors to be used for running model
!

!     integer, parameter :: npe=96
!     integer, parameter :: jtmax=jtrun/npe+1
!     integer, parameter :: my_max=my/npe+1

!
!  specify the number for the "outdir" array request
!  the number must be greater than the number of keys in "ocards"
!  (modify at 2002/4/15)
!

!     integer, parameter :: nout=5000
      integer nout
      common/comparam3/nout

      logical io_quilting
      common/comparam31/io_quilting
 
      integer  npe,jtmax,my_max
!2Dmpi
      integer  nx_max,npex,npey

      common/comparam4/npe,jtmax,my_max,nx_max,npex,npey

      end module param
