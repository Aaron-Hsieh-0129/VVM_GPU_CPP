! ===================================================================== !
!  description:                                                         !
!                                                                       !
!  usage:                                                               !
!                                                                       !
!      call sfc_drv                                                     !
!  ---  inputs:                                                         !
!          ( im, km, ps, u1, v1, t1, q1, soiltyp, vegtype, sigmaf,      !
!            sfcemis, dlwflx, dswsfc, snet, delt, tg3, cm, ch,          !
!            prsl1, prslki, zf, islimsk, ddvel, slopetyp,               !
!            shdmin, shdmax, snoalb, sfalb, flag_iter, flag_guess,      !
!            isot, ivegsrc,                                             !
!  ---  in/outs:                                                        !
!            weasd, snwdph, tskin, tprcp, srflag, smc, stc, slc,        !
!            canopy, trans, tsurf, zorl,                                !
!  ---  outputs:                                                        !
!            sncovr1, qsurf, gflux, drain, evap, hflx, ep, runoff,      !
!            cmm, chh, evbs, evcw, sbsno, snowc, stm, snohf,            !
!            smcwlt2, smcref2, wet1 )                                   !
!                                                                       !
!                                                                       !
!  subprogram called:  sflx                                             !
!                                                                       !
!  program history log:                                                 !
!         xxxx  --             created                                  !
!         200x  -- sarah lu    modified                                 !
!    oct  2006  -- h. wei      modified                                 !
!    apr  2009  -- y.-t. hou   modified to include surface emissivity   !
!                     effect on lw radiation. replaced the comfussing   !
!                     slrad (net sw + dlw) with sfc net sw snet=dsw-usw !
!    sep  2009  -- s. moorthi modification to remove rcl and unit change!
!    nov  2011  -- sarah lu    corrected wet1 calculation
!                                                                       !
!  ====================  defination of variables  ====================  !
!                                                                       !
!  inputs:                                                       size   !
!     im       - integer, horiz dimention and num of used pts      1    !
!     km       - integer, vertical soil layer dimension            1    !
!     ps       - real, surface pressure (pa)                       im   !
!     u1, v1   - real, u/v component of surface layer wind         im   !
!     t1       - real, surface layer mean temperature (k)          im   !
!     q1       - real, surface layer mean specific humidity        im   !
!     soiltyp  - integer, soil type (integer index)                im   !
!     vegtype  - integer, vegetation type (integer index)          im   !
!     sigmaf   - real, areal fractional cover of green vegetation  im   !
!     sfcemis  - real, sfc lw emissivity ( fraction )              im   !
!     dlwflx   - real, total sky sfc downward lw flux ( w/m**2 )   im   !
!     dswflx   - real, total sky sfc downward sw flux ( w/m**2 )   im   !
!     snet     - real, total sky sfc netsw flx into ground(w/m**2) im   !
!     delt     - real, time interval (second)                      1    !
!     tg3      - real, deep soil temperature (k)                   im   !
!     cm       - real, surface exchange coeff for momentum (m/s)   im   !
!     ch       - real, surface exchange coeff heat & moisture(m/s) im   !
!     prsl1    - real, sfc layer 1 mean pressure (pa)              im   !
!     prslki   - real,                                             im   !
!     zf       - real, height of bottom layer (m)                  im   !
!     islimsk  - integer, sea/land/ice mask (=0/1/2)               im   !
!     ddvel    - real,                                             im   !
!     slopetyp - integer, class of sfc slope (integer index)       im   !
!     shdmin   - real, min fractional coverage of green veg        im   !
!     shdmax   - real, max fractnl cover of green veg (not used)   im   !
!     snoalb   - real, upper bound on max albedo over deep snow    im   !
!     sfalb    - real, mean sfc diffused sw albedo (fractional)    im   !
!     flag_iter- logical,                                          im   !
!     flag_guess-logical,                                          im   !
!     isot     - integer, sfc soil type data source zobler or statsgo   !
!     ivegsrc  - integer, sfc veg type data source umd or igbp          !
!                                                                       !
!  input/outputs:                                                       !
!     weasd    - real, water equivalent accumulated snow depth (mm) im  !
!     snwdph   - real, snow depth (water equiv) over land          im   !
!     tskin    - real, ground surface skin temperature ( k )       im   !
!     tprcp    - real, total precipitation                         im   !
!     srflag   - real, snow/rain flag for precipitation            im   !
!     smc      - real, total soil moisture content (fractional)   im,km !
!     stc      - real, soil temp (k)                              im,km !
!     slc      - real, liquid soil moisture                       im,km !
!     canopy   - real, canopy moisture content (m)                 im   !
!     trans    - real, total plant transpiration (m/s)             im   !
!     tsurf    - real, surface skin temperature (after iteration)  im   !
!                                                                       !
!  outputs:                                                             !
!     sncovr1  - real, snow cover over land (fractional)           im   !
!     qsurf    - real, specific humidity at sfc                    im   !
!     gflux    - real, soil heat flux (w/m**2)                     im   !
!     drain    - real, subsurface runoff (mm/s)                    im   !
!     evap     - real, evaperation from latent heat flux           im   !
!     hflx     - real, sensible heat flux                          im   !
!     ep       - real, potential evaporation                       im   !
!     runoff   - real, surface runoff (m/s)                        im   !
!     cmm      - real,                                             im   !
!     chh      - real,                                             im   !
!     evbs     - real, direct soil evaporation (m/s)               im   !
!     evcw     - real, canopy water evaporation (m/s)              im   !
!     sbsno    - real, sublimation/deposit from snopack (m/s)      im   !
!     snowc    - real, fractional snow cover                       im   !
!     stm      - real, total soil column moisture content (m)      im   !
!     snohf    - real, snow/freezing-rain latent heat flux (w/m**2)im   !
!     smcwlt2  - real, dry soil moisture threshold                 im   !
!     smcref2  - real, soil moisture threshold                     im   !
!     zorl     - real, surface roughness                           im   !
!     wet1     - real, normalized soil wetness                     im   !
!                                                                       !
!  ====================    end of description    =====================  !

!-----------------------------------
      subroutine sfc_drv_gpu                                                &
!...................................
!  ---  inputs: &
           ( myim, im, km, lev, ncld, ps, u1, v1, t1, q1, soiltyp, vegtype, sigmaf,  &  
             sfcemis, dlwflx, dswsfc, snet, delt, tg3, cm, ch,          &  
             prsl1, prslki, zf, islimsk, ddvel, slopetyp,               & 
             shdmin, shdmax, snoalb, sfalb, flag_iter, flag_guess,      & 
             isot, ivegsrc,                                             &
!  ---  in/outs: &
             weasd, snwdph, tskin, tprcp, srflag, smc, stc, slc,        & 
             canopy,  tsurf, zorl,                                &
!            canopy, trans, tsurf, zorl,                                &
!  ---  outputs: &
             sncovr1, qsurf, gflux, drain, evap, hflx, ep, runoff,      &
!    &       cmm, chh, evbs, evcw, sbsno, snowc, stm, snohf,            &
!    &       smcwlt2, smcref2, wet1                                     & 
             albedo2, j, io, jo, async_id)
!
      !$acc routine(fpvs_gpu) seq
      use machine , only : kind_phys
!     use funcphys, only : fpvs
      use physcons, only : grav   => con_g,    cp   => con_cp,          & 
                           hvap   => con_hvap, rd   => con_rd,          & 
                           eps    => con_eps, epsm1 => con_epsm1,       & 
                           rvrdm1 => con_fvirt
      use const,    only : RTYPE
      use param,    only : my_max
      use index,    only : jlistnum
      use rank, only: myrank

      implicit none

!  ---  constant parameters:
      real(kind=kind_phys), parameter :: cpinv   = 1.0/cp
      real(kind=kind_phys), parameter :: hvapi   = 1.0/hvap
      real(kind=kind_phys), parameter :: elocp   = hvap/cp
      real(kind=kind_phys), parameter :: rhoh2o  = 1000.0
      real(kind=kind_phys), parameter :: a2      = 17.2693882
      real(kind=kind_phys), parameter :: a3      = 273.16
      real(kind=kind_phys), parameter :: a4      = 35.86
      real(kind=kind_phys), parameter :: a23m4   = a2*(a3-a4)

      real(kind=kind_phys), save         :: zsoil_noah(4)
      data zsoil_noah / -0.1, -0.4, -1.0, -2.0 /

!  ---  input:
      integer, intent(in) :: im, km, isot, ivegsrc, lev, ncld, async_id
      integer, dimension(my_max), intent(in) :: myim
      integer, dimension(im, my_max), intent(in) :: soiltyp, vegtype, slopetyp

      real(kind=RTYPE), dimension(im, lev, my_max) ::     u1, v1, t1
      real(kind=RTYPE), dimension(im, lev*ncld, my_max) :: q1
      real (kind=kind_phys), dimension(im, my_max), intent(in) :: ps,           & 
             sigmaf, sfcemis, dlwflx, dswsfc, snet, tg3, cm,            & 
             ch, prsl1, prslki, ddvel, shdmin, shdmax,                  & 
             snoalb, sfalb, zf

      integer, dimension(im, my_max), intent(in) :: islimsk
      real (kind=kind_phys),  intent(in) :: delt

      logical, dimension(im, my_max), intent(in) :: flag_iter, flag_guess

!  ---  in/out:
!     real (kind=kind_phys), dimension(im), intent(inout) :: weasd,     & 
      real (kind=kind_phys), dimension(im, my_max) :: weasd,     & 
             snwdph, tskin, tprcp, srflag, canopy, tsurf
      real (kind=kind_phys), dimension(im, my_max) :: trans
      real (kind=kind_phys), dimension(im, km, my_max), intent(inout) ::         & 
             smc, stc, slc

!  ---  output:
!     real (kind=kind_phys), dimension(im), intent(out) :: sncovr1,     & 
      real (kind=kind_phys), dimension(im, my_max)  :: sncovr1,     & 
             qsurf, gflux, drain, evap, hflx, ep, runoff,    & 
             zorl, albedo2
      real (kind=kind_phys), dimension(im, my_max)  :: chh,     & 
             evbs, evcw, sbsno, snowc, stm, snohf, smcwlt2, smcref2,    & 
             wet1
!  ---  locals:
      real (kind=kind_phys), dimension(km) :: stc1d, smc1d, slc1d, et1d
      real (kind=kind_phys), dimension(im, my_max) :: rch, rho,                 & 
             q0, qs1, theta1, wind
      real (kind=kind_phys), dimension(6, im, my_max) :: old_2d

      real (kind=kind_phys), dimension(im, km, my_max) :: et
      real (kind=kind_phys), dimension(km) :: sldpth

      real (kind=kind_phys), dimension(3, im, km, my_max) :: old_3d

      real (kind=kind_phys) , dimension(im, my_max) :: beta, chx, cmx,        & 
             dew, drip, dqsdt2,         & 
             flx1, flx2, flx3, ffrozp, pc, prcp,         & 
             rc, rcs, rct, rcq, rcsoil, rsmin,           & 
             runoff3, sfctmp,         & 
             shdfac,          & 
             smcdry, smcmax, snowh,            & 
             snomlt, soilw, tbot,      & 
             xlai
      real (kind=kind_phys) :: tem, fpvs_gpu, ttmp
      
      integer, parameter :: nxpvs = 7501
      real :: c1xpvs,c2xpvs,tbpvs(nxpvs)
      common/fpvscom/ c1xpvs,c2xpvs,tbpvs(nxpvs)

      real :: q0r, qs1r, weasdr, snowhr
      integer , dimension(im, my_max) :: couple, ice, nroot
      integer :: nsoil
      integer :: i, k, cnt, j, ii
      integer :: jj, io,jo, check, check2

      logical :: flag(im, my_max)
!
!===> ...  begin here
!


      !$acc enter data create(flag, old_2d, old_3d, evbs, evcw, &
      !$acc&      trans, sbsno, snowc, snohf, wind, &
      !$acc&      q0, theta1, rho, qs1, couple, ffrozp, ice, &
      !$acc&      prcp, sfctmp, &
      !$acc&      dqsdt2, &
      !$acc&      shdfac, tbot, &
      !$acc&      snowh, chx, cmx, &
      !$acc&      chh, nroot, et, &
      !$acc&      drip, dew, beta, flx1, flx2, flx3, &
      !$acc&      runoff3, snomlt, rc, pc, rsmin, zsoil_noah, &
      !$acc&      xlai, rcs, rct, rcq, rcsoil, soilw, smcdry, sldpth, &
      !$acc&      smcmax, stm, smcwlt2, smcref2, rch) async(async_id)
      !!$acc enter data copyin(sldpth, idx2) async(async_id)

      nsoil = km
      !$acc kernels async(async_id)
      zsoil_noah(1) = -0.1
      zsoil_noah(2) = -0.4
      zsoil_noah(3) = -1.0
      zsoil_noah(4) = -2.0
      sldpth(1) = - zsoil_noah(1)
      do k = 2, km
         sldpth(k) = zsoil_noah(k-1) - zsoil_noah(k)
      enddo
      !$acc end kernels
      !$acc parallel loop gang vector collapse(2) private(ttmp, jj, i, k, ii, &
      !$acc&         q0r, qs1r, weasdr, snowhr) async(async_id)
      do jj = 1, jlistnum
         do i = 1, im
            if (i .le. myim(jj)) then
!  --- ...  set flag for land points
              flag(i, jj) = (islimsk(i, jj) == 1)
              
   !  --- ...  save land-related prognostic fields for guess run
               if (flag(i, jj) .and. flag_guess(i, jj)) then
                  old_2d(1, i, jj)  = weasd(i, jj)
                  old_2d(2, i, jj) = snwdph(i, jj)
                  old_2d(3, i, jj)  = tskin(i, jj)
                  old_2d(4, i, jj) = canopy(i, jj)
                  old_2d(5, i, jj)  = tprcp(i, jj)
                  old_2d(6, i, jj) = srflag(i, jj)
                  !$acc loop seq
                  do k = 1, km
                     old_3d(1, i, k, jj) = smc(i, k, jj)
                     old_3d(2, i, k, jj) = stc(i, k, jj)
                     old_3d(3, i, k, jj) = slc(i, k, jj)
                  enddo
               endif
               
               if (flag_iter(i, jj) .and. flag(i, jj)) then
                  weasdr = weasd(i, jj)
   !  --- ...  initialization block
                  ep(i, jj)     = 0.0
                  evap (i, jj)  = 0.0
                  hflx (i, jj)  = 0.0
                  gflux(i, jj)  = 0.0
                  drain(i, jj)  = 0.0
                  canopy(i, jj) = max(canopy(i, jj), 0.0)

                  evbs (i, jj)  = 0.0
                  evcw (i, jj)  = 0.0
                  trans(i, jj)  = 0.0
                  sbsno(i, jj)  = 0.0
                  snowc(i, jj)  = 0.0
                  snohf(i, jj)  = 0.0
   !  --- ...  initialize variables
                  wind(i, jj) = max(sqrt( u1(i, lev, jj)*u1(i, lev, jj) + v1(i, lev, jj)*v1(i, lev, jj) )               & 
                              + max(0.0, min(ddvel(i, jj), 30.0)), 1.0)

                  q0r   = max(q1(i, lev, jj), 1.e-8)   !* q1=specific humidity at level 1 (kg/kg)
                  theta1(i, jj) = t1(i, lev, jj) * prslki(i, jj) !* adiabatic temp at level 1 (k)

                  rho(i, jj) = prsl1(i, jj) / (rd*t1(i, lev, jj)*(1.0+rvrdm1*q0r))
                  ttmp   = t1(i, lev, jj)
                  qs1r = fpvs_gpu( ttmp,c1xpvs,c2xpvs,tbpvs )        !* qs1=sat. humidity at level 1 (kg/kg)
                  qs1r = max(eps*qs1r / (prsl1(i, jj)+epsm1*qs1r), 1.e-8)
                  q0r = min(qs1r, q0r)
                  

               !  --- ...  noah: prepare variables to run noah lsm
               !   1. configuration information (c):
               !      ------------------------------
               !    couple  - couple-uncouple flag (=1: coupled, =0: uncoupled)
               !    ffrozp  - flag for snow-rain detection (1.=snow, 0.=rain)
               !    ice     - sea-ice flag (=1: sea-ice, =0: land)
               !    dt      - timestep (sec) (dt should not exceed 3600 secs) = delt
               !    zlvl    - height (m) above ground of atmospheric forcing variables
               !    nsoil   - number of soil layers (at least 2)
               !    sldpth  - the thickness of each soil layer (m)

               couple(i, jj) = 1                      ! run noah lsm in 'couple' mode
               if     (srflag(i, jj) == 1.0) then  ! snow phase
               ffrozp(i, jj) = 1.0
               elseif (srflag(i, jj) == 0.0) then  ! rain phase
               ffrozp(i, jj) = 0.0
               endif
               ice(i, jj) = 0



               !   2. forcing data (f):
               !      -----------------
               !    lwdn    - lw dw radiation flux (w/m2)
               !    solnet  - net sw radiation flux (dn-up) (w/m2)
               !    sfcprs  - pressure at height zlvl above ground (pascals)
               !    prcp    - precip rate (kg m-2 s-1)
               !    sfctmp  - air temperature (k) at height zlvl above ground
               !    th2     - air potential temperature (k) at height zlvl above ground
               !    q2      - mixing ratio at height zlvl above ground (kg kg-1)


               prcp(i, jj)   = rhoh2o * tprcp(i, jj) / delt
               sfctmp(i, jj) = t1(i, lev, jj)

               !   3. other forcing (input) data (i):
               !      ------------------------------
               !    sfcspd  - wind speed (m s-1) at height zlvl above ground
               !    q2sat   - sat mixing ratio at height zlvl above ground (kg kg-1)
               !    dqsdt2  - slope of sat specific humidity curve at t=sfctmp (kg kg-1 k-1)

               dqsdt2(i, jj) = qs1r * a23m4/(sfctmp(i, jj)-a4)**2

               !   4. canopy/soil characteristics (s):
               !      --------------------------------
               !    vegtyp  - vegetation type (integer index)                       -> vtype
               !    soiltyp - soil type (integer index)                             -> stype
               !    slopetyp- class of sfc slope (integer index)                    -> slope
               !    shdfac  - areal fractional coverage of green vegetation (0.0-1.0)
               !    shdmin  - minimum areal fractional coverage of green vegetation -> shdmin1d
               !    ptu     - photo thermal unit (plant phenology for annuals/crops)
               !    alb     - backround snow-free surface albedo (fraction)
               !    snoalb  - upper bound on maximum albedo over deep snow          -> snoalb1d
               !    tbot    - bottom soil temperature (local yearly-mean sfc air temp)

               shdfac(i, jj) = sigmaf(i, jj)
               tbot(i, jj) = tg3(i, jj)

               !   5. history (state) variables (h):
               !      ------------------------------
               !    cmc     - canopy moisture content (m)
               !    t1      - ground/canopy/snowpack) effective skin temperature (k)   -> tsea
               !    stc(nsoil) - soil temp (k)                                         -> stsoil
               !    smc(nsoil) - total soil moisture content (volumetric fraction)     -> smsoil
               !    sh2o(nsoil)- unfrozen soil moisture content (volumetric fraction)  -> slsoil
               !    snowh   - actual snow depth (m)
               !    sneqv   - liquid water-equivalent snow depth (m)
               !    albedo  - surface albedo including snow effect (unitless fraction)
               !    ch      - surface exchange coefficient for heat and moisture (m s-1) -> chx
               !    cm      - surface exchange coefficient for momentum (m s-1)          -> cmx

               canopy(i, jj) = canopy(i, jj) * 0.001            ! convert from mm to m


               snowhr = snwdph(i, jj) * 0.001         ! convert from mm to m
               weasdr = weasdr  * 0.001         ! convert from mm to m
               if (weasdr /= 0.0 .and. snowhr == 0.0) then
               snowhr = 10.0 * weasdr
               endif

               chx(i, jj)    = ch(i, jj)  * wind(i, jj)              ! compute conductance
               cmx(i, jj)    = cm(i, jj)  * wind(i, jj)
               chh(i, jj) = chx(i, jj) * rho(i, jj)

               !  ---- ... outside sflx, roughness uses cm as unit
               zorl(i, jj) = zorl(i, jj)/100.
                  
               q0(i, jj) = q0r
               qs1(i, jj) = qs1r
               weasd(i, jj) = weasdr
               snowh(i, jj) = snowhr
 
              end if
            end if
         end do
      end do
      

   !  --- ...  call noah lsm
      !!$acc update self(idx2) async(async_id)
      !!$acc wait(async_id)

       call sflx_gpu                                                     &
!  ---  inputs: &
        ( nsoil, im, myim, flag, flag_iter, &
          couple, ice, ffrozp, delt, zf, sldpth,            & 
          dswsfc, snet, dlwflx, sfcemis, prsl1, sfctmp,                & 
          wind, prcp, q0, qs1, dqsdt2, theta1, ivegsrc,             & 
          vegtype, soiltyp, slopetyp, shdmin, sfalb, snoalb,              &
!  ---  input/outputs: &
          tbot, canopy, tsurf, stc, &
          smc, slc, weasd, chx, cmx,  & 
          zorl,                                                        &
!  ---  outputs: &
          nroot, shdfac, snowh, albedo2, evap, hflx, evcw,              & 
          evbs, et, trans, sbsno, drip, dew, beta, ep, gflux,         & 
          flx1, flx2, flx3, runoff, drain, runoff3,               & 
          snomlt, sncovr1, rc, pc, rsmin, xlai, rcs, rct, rcq,        & 
          rcsoil, soilw, stm, smcwlt2, smcdry, smcref2, smcmax,      &
          async_id)

      !!$acc wait(async_id)
      !!$acc update self(soiltyp, vegtype, slopetyp, sfcemis, dlwflx, &
      !!$acc&       dswsfc, snet, snoalb, sfalb, zf, prsl1, shdmin, weasd, &
      !!$acc&       canopy, tsurf, trans, sncovr1, gflux, drain, evap, &
      !!$acc&       hflx, ep, runoff, zorl, albedo2, evbs, evcw, sbsno, &
      !!$acc&       stm, smcwlt2, smcref2, q0, qs1, theta1, wind, beta, &
      !!$acc&       chx, cmx, dew, drip, dqsdt2, flx1, flx2, flx3, ffrozp, &
      !!$acc&       pc, prcp, rc, rcs, rct, rcq, rcsoil, rsmin, runoff3, &
      !!$acc&       sfctmp, shdfac, smcdry, smcmax, snowh, snomlt, soilw, &
      !!$acc&       tbot, xlai, couple, ice, nroot, sldpth, stc, smc, slc, &
      !!$acc&       et, myim, flag_iter, flag) &
      !!$acc&       async(async_id)
      !!$acc wait(async_id)
      !do jj = 1, jlistnum
      !   do i = 1, myim(jj)
      !      if (flag_iter(i, jj) .and. flag(i, jj)) then
      !         do k = 1, km
      !            stc1d(k) = stc(i, k, jj)
      !            smc1d(k) = smc(i, k, jj)
      !            slc1d(k) = slc(i, k, jj)
      !            et1d(k) = et(i, k, jj)
      !         end do
      !          call sflx                                                     &
      !   !  ---  inputs: &
      !           ( nsoil, couple(i, jj), ice(i, jj), ffrozp(i, jj), delt, zf(i, jj), sldpth,            & 
      !             dswsfc(i, jj), snet(i, jj), dlwflx(i, jj), sfcemis(i, jj), prsl1(i, jj), sfctmp(i, jj),                & 
      !             wind(i, jj), prcp(i, jj), q0(i, jj), qs1(i, jj), dqsdt2(i, jj), theta1(i, jj), ivegsrc,             & 
      !             vegtype(i, jj), soiltyp(i, jj), slopetyp(i, jj), shdmin(i, jj), sfalb(i, jj), snoalb(i, jj),              &
      !   !  ---  input/outputs: &
      !             tbot(i, jj), canopy(i, jj), tsurf(i, jj), stc1d, &
      !             smc1d, slc1d, weasd(i, jj), chx(i, jj), cmx(i, jj),  & 
      !             zorl(i, jj),                                                        &
      !   !  ---  outputs: &
      !             nroot(i, jj), shdfac(i, jj), snowh(i, jj), albedo2(i, jj), evap(i, jj), hflx(i, jj), evcw(i, jj),              & 
      !             evbs(i, jj), et1d, trans(i, jj), sbsno(i, jj), drip(i, jj), dew(i, jj), beta(i, jj), ep(i, jj), gflux(i, jj),         & 
      !             flx1(i, jj), flx2(i, jj), flx3(i, jj), runoff(i, jj), drain(i, jj), runoff3(i, jj),               & 
      !             snomlt(i, jj), sncovr1(i, jj), rc(i, jj), pc(i, jj), rsmin(i, jj), xlai(i, jj), rcs(i, jj), rct(i, jj), rcq(i, jj),        & 
      !             rcsoil(i, jj), soilw(i, jj), stm(i, jj), smcwlt2(i, jj), smcdry(i, jj), smcref2(i, jj), smcmax(i, jj))
      !         do k = 1, km
      !            stc(i, k, jj) = stc1d(k)
      !            smc(i, k, jj) = smc1d(k)
      !            slc(i, k, jj) = slc1d(k)
      !            et(i, k, jj) = et1d(k)
      !         end do
      !      end if
      !   end do
      !end do
      !!$acc wait(async_id)
      !!$acc update device(soiltyp, vegtype, slopetyp, sfcemis, &
      !!$acc&       dlwflx, dswsfc, snet, snoalb, sfalb, zf, prsl1, shdmin, &
      !!$acc&       weasd, canopy, tsurf, trans, sncovr1, gflux, drain, &
      !!$acc&       evap, hflx, ep, runoff, zorl, albedo2, evbs, evcw, &
      !!$acc&       sbsno, stm, smcwlt2, smcref2, q0, qs1, theta1, wind, &
      !!$acc&       beta, chx, cmx, dew, drip, dqsdt2, flx1, flx2, flx3, &
      !!$acc&       ffrozp, pc, prcp, rc, rcs, rct, rcq, rcsoil, rsmin, &
      !!$acc&       runoff3, sfctmp, shdfac, smcdry, smcmax, snowh, snomlt, &
      !!$acc&       soilw, tbot, xlai, couple, ice, nroot, sldpth, stc, &
      !!$acc&       smc, slc, et, myim, flag_iter, flag) &
      !!$acc&       async(async_id)
      !!$acc wait(async_id)
      !write(*,*) check
   !  --- ...  noah: prepare variables for return to parent mode
   !   6. output (o):
   !      -----------
   !    eta     - actual latent heat flux (w m-2: positive, if upward from sfc)
   !    sheat   - sensible heat flux (w m-2: positive, if upward from sfc)
   !    beta    - ratio of actual/potential evap (dimensionless)
   !    etp     - potential evaporation (w m-2)
   !    ssoil   - soil heat flux (w m-2: negative if downward from surface)
   !    runoff1 - surface runoff (m s-1), not infiltrating the surface
   !    runoff2 - subsurface runoff (m s-1), drainage out bottom

      !$acc parallel loop gang vector collapse(2) private(k, jj, i, tem) async(async_id)
      do jj = 1, jlistnum
         do i = 1, im
            if (i .le. myim(jj)) then
               if (flag_iter(i, jj) .and. flag(i, jj)) then
                snohf(i, jj) = flx1(i, jj) + flx2(i, jj) + flx3(i, jj)

                wet1(i, jj) = smc(i, 1, jj) / smcmax(i, jj) !Sarah Lu added 09/09/2010 (for GOCART)

      !  --- ...  unit conversion (from m s-1 to mm s-1)
                runoff(i, jj)  = runoff(i, jj) * 1000.0
                drain (i, jj)  = drain(i, jj) * 1000.0

      !  --- ...  unit conversion (from m to mm)
                canopy(i, jj)  = canopy(i, jj)   * 1000.0
                snwdph(i, jj)  = snowh(i, jj) * 1000.0
                weasd(i, jj)   = weasd(i, jj) * 1000.0
      !  ---- ... outside sflx, roughness uses cm as unit (update after snow's
      !  effect)
                zorl(i, jj) = zorl(i, jj)*100.

      !  --- ...  do not return the following output fields to parent model
      !    ec      - canopy water evaporation (m s-1)
      !    edir    - direct soil evaporation (m s-1)
      !    et(nsoil)-plant transpiration from a particular root layer (m s-1)
      !    ett     - total plant transpiration (m s-1)
      !    esnow   - sublimation from (or deposition to if <0) snowpack (m s-1)
      !    drip    - through-fall of precip and/or dew in excess of canopy
      !              water-holding capacity (m)
      !    dew     - dewfall (or frostfall for t<273.15) (m)
      !    beta    - ratio of actual/potential evap (dimensionless)
      !    flx1    - precip-snow sfc (w m-2)
      !    flx2    - freezing rain latent heat flux (w m-2)
      !    flx3    - phase-change heat flux from snowmelt (w m-2)
      !    snomlt  - snow melt (m) (water equivalent)
      !    sncovr  - fractional snow cover (unitless fraction, 0-1)
      !    runoff3 - numerical trunctation in excess of porosity (smcmax)
      !              for a given soil layer at the end of a time step
      !    rc      - canopy resistance (s m-1)
      !    pc      - plant coefficient (unitless fraction, 0-1) where pc*etp
      !              = actual transp
      !    xlai    - leaf area index (dimensionless)
      !    rsmin   - minimum canopy resistance (s m-1)
      !    rcs     - incoming solar rc factor (dimensionless)
      !    rct     - air temperature rc factor (dimensionless)
      !    rcq     - atmos vapor pressure deficit rc factor (dimensionless)
      !    rcsoil  - soil moisture rc factor (dimensionless)
      !    soilw   - available soil moisture in root zone (unitless fraction
      !              between smcwlt and smcmax)
      !    soilm   - total soil column moisture content (frozen+unfrozen) (m)
      !    smcwlt  - wilting point (volumetric)
      !    smcdry  - dry soil moisture threshold where direct evap frm top
      !              layer ends (volumetric)
      !    smcref  - soil moisture threshold where transpiration begins to
      !              stress (volumetric)
      !    smcmax  - porosity, i.e. saturated value of soil moisture
      !              (volumetric)
      !    nroot   - number of root layers, a function of veg type, determined
      !              in subroutine redprm.

   !   --- ...  compute qsurf (specific humidity at sfc)
                  rch(i, jj)   = rho(i, jj) * cp * ch(i, jj) * wind(i, jj)
                  qsurf(i, jj) = q1(i, lev, jj)  + evap(i, jj) / (elocp * rch(i, jj))
               
                  tem     = 1.0 / rho(i, jj)
                  hflx(i, jj) = hflx(i, jj) * tem * cpinv
                  evap(i, jj) = evap(i, jj) * tem * hvapi

              end if
   !  --- ...  restore land-related prognostic fields for guess run
               if (flag(i, jj)) then
                  if (flag_guess(i, jj)) then
                     weasd(i, jj)  = old_2d(1, i, jj)
                     snwdph(i, jj) = old_2d(2, i, jj)
                     tskin(i, jj)  = old_2d(3, i, jj)
                     canopy(i, jj) = old_2d(4, i, jj)
                     tprcp(i, jj)  = old_2d(5, i, jj)
                     srflag(i, jj) = old_2d(6, i, jj)
                     !$acc loop seq
                     do k = 1, km
                        smc(i, k, jj) = old_3d(1, i, k, jj)
                        stc(i, k, jj) = old_3d(2, i, k, jj)
                        slc(i, k, jj) = old_3d(3, i, k, jj)
                     end do
                  else
                     tskin(i, jj) = tsurf(i, jj)
                  end if
               end if
           end if   ! end if_flag_iter_and_flag_block
         end do   ! end do_i_loop
      end do

      
      !$acc exit data delete(flag, old_2d, old_3d, &
      !$acc&     evbs, evcw, trans, sbsno, snowc, snohf, wind, &
      !$acc&     q0, theta1, rho, qs1, couple, ffrozp, ice, &
      !$acc&     sldpth, prcp, sfctmp, &
      !$acc&     dqsdt2,  &
      !$acc&     shdfac, tbot, &
      !$acc&     snowh, chx, cmx, &
      !$acc&     chh, nroot, et, &
      !$acc&     drip, dew, beta, flx1, flx2, flx3, &
      !$acc&     runoff3, snomlt, rc, pc, rsmin, zsoil_noah, &
      !$acc&     xlai, rcs, rct, rcq, rcsoil, soilw, smcdry, sldpth, &
      !$acc&     smcmax, stm, smcwlt2, smcref2, rch) async(async_id)
!
      return
!...................................
      end subroutine sfc_drv_gpu
!-----------------------------------

