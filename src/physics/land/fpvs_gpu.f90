      function fpvs_gpu(t,c1xpvs,c2xpvs,tbpvs)
      !$acc routine seq
!
! Abstract: Compute saturation vapor pressure from the temperature.
!   A linear interpolation is done between values in a lookup table
!   computed in gpvs. See documentation for fpvsx for details.
!   Input values outside table range are reset to table extrema.
!   The interpolation accuracy is almost 6 decimal places.
!   On the Cray, fpvs is about 4 times faster than exact calculation.
!   This function should be expanded inline in the calling routine.
!
!   Input argument list:
!       t              Real temperature in Kelvin
!
!   Output functio:
!       fpvs        Real saturation vapor pressure in Pascals
!
      implicit none
      real fpvs_gpu
      real t
      integer jx
      real xj
      integer nxpvs
      parameter(nxpvs=7501)
      real c1xpvs,c2xpvs,tbpvs(nxpvs)
      
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      xj=min(max(c1xpvs+c2xpvs*t,1.),real(nxpvs))
      jx=min(xj,nxpvs-1.)
      fpvs_gpu=tbpvs(jx)+(xj-jx)*(tbpvs(jx+1)-tbpvs(jx))
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      end function
