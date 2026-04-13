      subroutine sfc_diag_gpu(myim, im, lev, ncld, ps, u1, v1, t1, q1, &
                          !    &                    tskin,qsurf,f10m,u10m,v10m,t2m,q2m,      &
                          tskin, qsurf, u10m, v10m, t2m, q2m, &
                          prslki, evap, fm, fh, fm10, fh2, fh10, rh2, rh10, async_id)
!
         !$acc routine(fpvs_gpu) seq
         use machine, only: kind_phys
!     use funcphys, only : fpvs
         use physcons, grav => con_g, cp => con_cp, &
            eps => con_eps, epsm1 => con_epsm1
         use const, only: RTYPE
         use param, only: my, my_max
         use index
         implicit none
!
         integer im, myim(my_max), lev, ncld, async_id
         real(kind=RTYPE), dimension(im, lev, my_max) ::     u1, v1, t1
         real(kind=RTYPE), dimension(im, lev*ncld, my_max) :: q1
         real, dimension(im, my_max) :: ps, tskin, qsurf, &
                          u10m, v10m, t2m, q2m, &
                          prslki, evap, fm, fh, fh10, fm10, fh2, rh2, rh10
         real :: f10m, t10m, q10m
!
!     locals
!
         real(kind=kind_phys), parameter :: qmin = 1.0e-8
         integer k, i, jj
!
         real(kind=kind_phys) fhi, qss, wrk, fpvs_gpu
         integer, parameter :: nxpvs = 7501
         real :: c1xpvs,c2xpvs,tbpvs(nxpvs)
         common/fpvscom/ c1xpvs,c2xpvs,tbpvs(nxpvs)
         real :: time1, time2
!     real(kind=kind_phys) sig2k, fhi, qss
!
!     real, parameter :: g=grav
!
!     estimate sigma ** k at 2 m
!
!     sig2k = 1. - 4. * g * 2. / (cp * 280.)
!
!  initialize variables. all units are supposedly m.k.s. unless specified
!  ps is in pascals
!
!!
         
         !!$acc enter data create(f10m, t10m, q10m) async(async_id)
         !$acc parallel loop gang vector collapse(2) &
         !$acc&         private(jj, i, wrk, qss, fhi) async(async_id)
         do jj = 1, jlistnum
            do i = 1, im
               if (i .le. myim(jj)) then
                  f10m = fm10(i, jj)/fm(i, jj)
      !       f10m(i) = min(f10m(i),1.)
                  u10m(i, jj) = f10m*u1(i, lev, jj)
                  v10m(i, jj) = f10m*v1(i, lev, jj)
                  fhi = min(fh2(i, jj)/fh(i, jj), 0.99)
      !       t2m(i)  = tskin(i)*(1. - fhi) + t1(i) * prslki(i) * fhi
      !       sig2k   = 1. - (grav+grav) / (cp * t2m(i))
      !       t2m(i)  = t2m(i) * sig2k
                  wrk = 1.0 - fhi

                  t2m(i, jj) = tskin(i, jj)*wrk + t1(i, lev, jj)*prslki(i, jj)*fhi - (grav + grav)/cp

                  if (evap(i, jj) >= 0.) then !  for evaporation>0, use inferred qsurf to deduce q2m
                     q2m(i, jj) = qsurf(i, jj)*wrk + max(qmin, q1(i, lev, jj))*fhi
                  else                   !  for dew formation, use saturated q at tskin
                     qss = fpvs_gpu(tskin(i, jj),c1xpvs,c2xpvs,tbpvs)
                     qss = eps*qss/(ps(i, jj) + epsm1*qss)
                     q2m(i, jj) = qss*wrk + max(qmin, q1(i, lev, jj))*fhi
                  end if
                  qss = fpvs_gpu(t2m(i, jj),c1xpvs,c2xpvs,tbpvs)
                  qss = eps*qss/(ps(i, jj) + epsm1*qss)
                  q2m(i, jj) = min(q2m(i, jj), qss)
      !
                  rh2(i, jj) = max(q2m(i, jj)/qss, 0.)

                  fhi = min(fh10(i, jj)/fh(i, jj), 0.99)
                  wrk = 1.0 - fhi
                  t10m = tskin(i, jj)*wrk + t1(i, lev, jj)*prslki(i, jj)*fhi - (grav + grav)/cp
                  if (evap(i, jj) >= 0.) then !  for evaporation>0, use inferred qsurf to deduce q2m
                     q10m = qsurf(i, jj)*wrk + max(qmin, q1(i, lev, jj))*fhi
                  else                   !  for dew formation, use saturated q at tskin
                     qss = fpvs_gpu(tskin(i, jj),c1xpvs,c2xpvs,tbpvs)
                     qss = eps*qss/(ps(i, jj) + epsm1*qss)
                     q10m = qss*wrk + max(qmin, q1(i, lev, jj))*fhi
                  end if
                  qss = fpvs_gpu(t10m,c1xpvs,c2xpvs,tbpvs)
                  qss = eps*qss/(ps(i, jj) + epsm1*qss)
                  q10m = min(q10m, qss)
      !
                  rh10(i, jj) = max(q10m/qss, 0.)
               end if
            end do
         end do
         !!$acc exit data delete(f10m, t10m, q10m) async(async_id)

         return
      end
