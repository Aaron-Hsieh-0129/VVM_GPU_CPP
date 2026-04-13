!-----------------------------------
      subroutine sfc_ocean_gpu &
         !...................................
         !  ---  inputs: &
         (myim, im, lev, ncld, ps, u1, v1, t1, q1, tskin, cm, ch, &
          prsl1, prslki, islimsk, ddvel, flag_iter, &
          !  ---  outputs:
          !    &       qsurf, cmm, chh, gflux, evap, hflx, ep                     &
#ifdef TIMCOMCPL
          qsurf, gflux, evap, hflx, ep, async_id, ssu, ssv)
#else
          qsurf, gflux, evap, hflx, ep, async_id)
#endif
      !$acc routine(fpvs_gpu) seq

! ===================================================================== !
!  description:                                                         !
!                                                                       !
!  usage:                                                               !
!                                                                       !
!    call sfc_ocean                                                     !
!       inputs:                                                         !
!          ( im, ps, u1, v1, t1, q1, tskin, cm, ch,                     !
!            prsl1, prslki, islimsk, ddvel, flag_iter,                  !
!       outputs:                                                        !
!            qsurf, cmm, chh, gflux, evap, hflx, ep )                   !
!                                                                       !
!                                                                       !
!  subprograms/functions called: fpvs                                   !
!                                                                       !
!                                                                       !
!  program history log:                                                 !
!         2005  -- created from the original progtm to account for      !
!                  ocean only                                           !
!    oct  2006  -- h. wei      added cmm and chh to the output          !
!    apr  2009  -- y.-t. hou   modified to match the modified gbphys.f  !
!                  reformatted the code and added program documentation !
!    sep  2009  -- s. moorthi removed rcl and made pa as pressure unit  !
!                  and furthur reformatted the code                     !
!                                                                       !
!                                                                       !
!  ====================  defination of variables  ====================  !
!                                                                       !
!  inputs:                                                       size   !
!     im       - integer, horizontal dimension                     1    !
!     ps       - real, surface pressure                            im   !
!     u1, v1   - real, u/v component of surface layer wind         im   !
!     t1       - real, surface layer mean temperature ( k )        im   !
!     q1       - real, surface layer mean specific humidity        im   !
!     tskin    - real, ground surface skin temperature ( k )       im   !
!     cm       - real, surface exchange coeff for momentum (m/s)   im   !
!     ch       - real, surface exchange coeff heat & moisture(m/s) im   !
!     prsl1    - real, surface layer mean pressure                 im   !
!     prslki   - real,                                             im   !
!     islimsk  - integer, sea/land/ice mask (=0/1/2)               im   !
!     ddvel    - real, wind enhancement due to convection (m/s)    im   !
!     flag_iter- logical,                                          im   !
!                                                                       !
!  outputs:                                                             !
!     qsurf    - real, specific humidity at sfc                    im   !
!     cmm      - real,                                             im   !
!     chh      - real,                                             im   !
!     gflux    - real, ground heat flux (zero for ocean)           im   !
!     evap     - real, evaporation from latent heat flux           im   !
!     hflx     - real, sensible heat flux                          im   !
!     ep       - real, potential evaporation                       im   !
!                                                                       !
! ===================================================================== !
!
         use machine, only: kind_phys
!     use funcphys, only : fpvs
         use physcons, only: cp => con_cp, rd => con_rd, eps => con_eps, &
                             epsm1 => con_epsm1, hvap => con_hvap, &
                             rvrdm1 => con_fvirt
         use const, only: RTYPE
         use param, only: my, my_max
         use index
!
         implicit none
!
!  ---  constant parameters:
         real(kind=kind_phys), parameter :: cpinv = 1.0/cp &
                                            , hvapi = 1.0/hvap &
                                            , elocp = hvap/cp

!  ---  inputs:
         integer, intent(in) :: im, myim(my_max), lev, ncld

         real(kind=RTYPE), dimension(im, lev, my_max) ::     u1, v1, t1
         real(kind=RTYPE), dimension(im, lev*ncld, my_max) :: q1
         real(kind=kind_phys), dimension(im, my_max), intent(in) :: ps, &
                           tskin, cm, ch, prsl1, prslki, ddvel
#ifdef TIMCOMCPL
         real(kind=kind_phys), dimension(im, my_max), intent(in) :: ssu, ssv
#endif
         integer, dimension(im, my_max), intent(in):: islimsk

         logical, intent(in) :: flag_iter(im, my_max)

!  ---  outputs:
!     real (kind=kind_phys), dimension(im), intent(out) :: qsurf,         &
         real(kind=kind_phys), dimension(im, my_max) :: qsurf, &
                     gflux, evap, hflx, ep
!  ---  locals:
#ifdef TIMCOMCPL
         real(kind=kind_phys), dimension(im, my_max) :: cpl_u1, cpl_v1
#endif

         real(kind=kind_phys) :: q0, qss, rch, rho, wind, tem, fpvs_gpu, &
            hflxr, evapr

         integer :: i, jj, async_id

         logical :: flag
         integer, parameter :: nxpvs = 7501
         real :: c1xpvs,c2xpvs,tbpvs(nxpvs)
         common/fpvscom/ c1xpvs,c2xpvs,tbpvs(nxpvs)
         real :: time1, time2
!
!===> ...  begin here
!
!  --- ...  flag for open water

#ifdef TIMCOMCPL
      !$acc enter data create(cpl_u1, cpl_v1) async(async_id)
#endif

         !$acc parallel loop gang vector collapse(2) private(jj, i, wind, &
         !$acc&         q0, rho, qss, rch, tem, hflxr, evapr) async(async_id)
         do jj = 1, jlistnum
            do i = 1, im
               if (i .le. myim(jj)) then
#ifdef TIMCOMCPL
                  cpl_u1(i, jj)=0.
                  cpl_v1(i, jj)=0.
                  cpl_u1(i, jj)=u1(i, lev, jj)
                  cpl_v1(i, jj)=v1(i, lev, jj)
#endif
                  flag = (islimsk(i, jj) == 0 .and. flag_iter(i, jj))

      !  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
      !           ps is in pascals, wind is wind speed,
      !           rho is density, qss is sat. hum. at surface

                  if (flag) then
#ifdef TIMCOMCPL
                     wind = max(sqrt((cpl_u1(i, jj)-ssu(i, jj))**2 + (cpl_v1(i, jj)-ssv(i, jj))**2)          &
                       + max( 0.0, min( ddvel(i, jj), 30.0 ) ), 1.0)
#else
                     wind = max(sqrt(u1(i, lev, jj)*u1(i, lev, jj) &
                                + v1(i, lev, jj)*v1(i, lev, jj)) &
                                + max(0.0, min(ddvel(i, jj), 30.0)), 1.0)
#endif          
                     q0 = max(q1(i, lev, jj), 1.0e-8)
                     rho = prsl1(i, jj)/(rd*t1(i, lev, jj)*(1.0 + rvrdm1*q0))

                     qss = fpvs_gpu(tskin(i, jj),c1xpvs,c2xpvs,tbpvs)
                     qss = eps*qss/(ps(i, jj) + epsm1*qss)

                     !evap(i, jj) = 0.0
                     !hflx(i, jj) = 0.0
                     ep(i, jj) = 0.0
                     gflux(i, jj) = 0.0

      !  --- ...    rcp  = rho cp ch v

                     rch = rho*cp*ch(i, jj)*wind
                     !cmm(i, jj) = cm(i, jj)*wind
                     !chh(i, jj) = rho*ch(i, jj)*wind

      !  --- ...  sensible and latent heat flux over open water

                     hflxr = rch*(tskin(i, jj) - t1(i, lev, jj)*prslki(i, jj))

                     evapr = elocp*rch*(qss - q0)
                     qsurf(i, jj) = qss

                     tem = 1.0/rho
                     hflx(i, jj) = hflxr*tem*cpinv
                     evap(i, jj) = evapr*tem*hvapi
                  end if
               end if
            end do
         end do
#ifdef TIMCOMCPL
         !$acc exit data delete(cpl_u1, cpl_v1) async(async_id)
#endif
!
         return
!...................................
      end subroutine sfc_ocean_gpu
!-----------------------------------
