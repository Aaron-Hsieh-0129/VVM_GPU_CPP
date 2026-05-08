    MODULE MACHINE

      IMPLICIT NONE
      SAVE
!  Machine dependant constants
      integer, parameter :: kind_io4  = 4, kind_io8  = 8 , kind_ior = 8 &
     &,                     kind_evod = 8, kind_dbl_prec = 8 &
! Users don't need to control anything here, the compiler and flags can tackle everything. Just define in the CMakePresets. 
#ifdef VVM_USE_DOUBLE_PRECISION
     &,                     kind_rad  = selected_real_kind(13,60) & ! 64-bit real
     &,                     kind_phys = selected_real_kind(13,60) & ! 64-bit real
     &,                     kind_REAL = 8                         & ! used in cmp_comm
#else
     &,                     kind_rad  = selected_real_kind(6,30)  & ! 32-bit real (Float)
     &,                     kind_phys = selected_real_kind(6,30)  & ! 32-bit real (Float)
     &,                     kind_REAL = 4                         & ! used in cmp_comm
#endif
     &,                     kind_INTEGER = 4                      ! -,,-
!
      real(kind=kind_evod), parameter :: mprec = 1.e-12           ! machine precision to restrict dep

    END MODULE MACHINE
