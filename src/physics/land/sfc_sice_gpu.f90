!-----------------------------------
      subroutine sfc_sice_gpu                                               &
!...................................
!  ---  inputs: &
           ( myim, im, km, lev, ncld, ps, u1, v1, t1, q1, delt,                     & 
             sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,                   & 
             cm, ch, prsl1, prslki, islimsk, ddvel,                     & 
             flag_iter, mom4ice, lsm,                                   &
!    &       flag_iter, mom4ice, lsm, lprnt,ipr,                        &
!  ---  input/outputs: &
             hice, fice, tice, weasd, tskin, tprcp, stc, ep,            &
!  ---  outputs:
!    &       snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx         & 
             snwdph, qsurf, snowmt, gflux, evap, hflx,                   & 
           async_id)
      !$acc routine(fpvs_gpu) seq

! ===================================================================== !
!  description:                                                         !
!                                                                       !
!  usage:                                                               !
!                                                                       !
!    call sfc_sice                                                      !
!       inputs:                                                         !
!          ( im, km, ps, u1, v1, t1, q1, delt,                          !
!            sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,                   !
!            cm, ch, prsl1, prslki, islimsk, ddvel,                     !
!            flag_iter, mom4ice, lsm,                                   !
!       input/outputs:                                                  !
!            hice, fice, tice, weasd, tskin, tprcp, stc, ep,            !
!       outputs:                                                        !
!            snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx )       !
!                                                                       !
!  subprogram called:  ice3lay.                                         !
!                                                                       !
!  program history log:                                                 !
!         2005  --  xingren wu created  from original progtm and added  !
!                     two-layer ice model                               !
!         200x  -- sarah lu    added flag_iter                          !
!    oct  2006  -- h. wei      added cmm and chh to output              !
!         2007  -- x. wu modified for mom4 coupling (i.e. mom4ice)      !
!         2007  -- s. moorthi micellaneous changes                      !
!    may  2009  -- y.-t. hou   modified to include surface emissivity   !
!                     effect on lw radiation. replaced the confusing    !
!                     slrad with sfc net sw sfcnsw (dn-up). reformatted !
!                     the code and add program documentation block.     !
!    sep  2009 -- s. moorthi removed rcl, changed pressure units and    !
!                     further optimized                                 !
!    jan  2015 -- x. wu change "cimin = 0.15" for both                  !
!                              uncoupled and coupled case               !
!                                                                       !
!                                                                       !
!  ====================  defination of variables  ====================  !
!                                                                       !
!  inputs:                                                       size   !
!     im, km   - integer, horiz dimension and num of soil layers   1    !
!     ps       - real, surface pressure                            im   !
!     u1, v1   - real, u/v component of surface layer wind         im   !
!     t1       - real, surface layer mean temperature ( k )        im   !
!     q1       - real, surface layer mean specific humidity        im   !
!     delt     - real, time interval (second)                      1    !
!     sfcemis  - real, sfc lw emissivity ( fraction )              im   !
!     dlwflx   - real, total sky sfc downward lw flux ( w/m**2 )   im   !
!     sfcnsw   - real, total sky sfc netsw flx into ground(w/m**2) im   !
!     sfcdsw   - real, total sky sfc downward sw flux ( w/m**2 )   im   !
!     srflag   - real, snow/rain flag for precipitation            im   !
!     cm       - real, surface exchange coeff for momentum (m/s)   im   !
!     ch       - real, surface exchange coeff heat & moisture(m/s) im   !
!     prsl1    - real, surface layer mean pressure                 im   !
!     prslki   - real,                                             im   !
!     islimsk  - integer, sea/land/ice mask (=0/1/2)               im   !
!     ddvel    - real,                                             im   !
!     flag_iter- logical,                                          im   !
!     mom4ice  - logical,                                          im   !
!     lsm      - integer, flag for land surface model scheme       1    !
!                =0: use osu scheme; =1: use noah scheme                !
!                                                                       !
!  input/outputs:                                                       !
!     hice     - real, sea-ice thickness                           im   !
!     fice     - real, sea-ice concentration                       im   !
!     tice     - real, sea-ice surface temperature                 im   !
!     weasd    - real, water equivalent accumulated snow depth (mm)im   !
!     tskin    - real, ground surface skin temperature ( k )       im   !
!     tprcp    - real, total precipitation                         im   !
!     stc      - real, soil temp (k)                              im,km !
!     ep       - real, potential evaporation                       im   !
!                                                                       !
!  outputs:                                                             !
!     snwdph   - real, water equivalent snow depth (mm)            im   !
!     qsurf    - real, specific humidity at sfc                    im   !
!     snowmt   - real, snow melt (m)                               im   !
!     gflux    - real, soil heat flux (w/m**2)                     im   !
!     cmm      - real,                                             im   !
!     chh      - real,                                             im   !
!     evap     - real, evaperation from latent heat flux           im   !
!     hflx     - real, sensible heat flux                          im   !
!                                                                       !
! ===================================================================== !
!
      use machine , only : kind_phys
!     use funcphys, only : fpvs
      use physcons, only : sbc => con_sbc, hvap => con_hvap,            & 
                           tgice => con_tice, cp => con_cp,             & 
                           eps => con_eps, epsm1 => con_epsm1,          & 
                           grav => con_g, rvrdm1 => con_fvirt,          & 
                           t0c => con_t0c, rd => con_rd
      use const,    only : RTYPE
      use param,    only: my_max
      use index,    only: jlistnum
      use rank, only:myrank
!
      implicit none
!
!  ---  constant parameters:
      integer,              parameter :: kmi   = 2        ! 2-layer of ice
      real(kind=kind_phys), parameter :: cpinv = 1.0/cp
      real(kind=kind_phys), parameter :: hvapi = 1.0/hvap
      real(kind=kind_phys), parameter :: elocp = hvap/cp
      real(kind=kind_phys), parameter :: himax = 8.0      ! maximum ice thickness allowed
      real(kind=kind_phys), parameter :: himin = 0.1      ! minimum ice thickness required
      real(kind=kind_phys), parameter :: hsmax = 2.0      ! maximum snow depth allowed
      real(kind=kind_phys), parameter :: timin = 173.0    ! minimum temperature allowed for snow/ice
      real(kind=kind_phys), parameter :: albfw = 0.06     ! albedo for lead
      real(kind=kind_phys), parameter :: dsi   = 1.0/0.33
      real(kind=kind_phys), parameter :: cimin = 0.15     !  --- minimum ice concentration

!  ---  constant parameters: (properties of ice, snow, and seawater)
!  ---  ice3lay
      real (kind=kind_phys), parameter :: ds   = 330.0    ! snow (ov sea ice) density (kg/m^3)
      real (kind=kind_phys), parameter :: dw   =1000.0    ! fresh water density  (kg/m^3)
      real (kind=kind_phys), parameter :: dsdw = ds/dw
      real (kind=kind_phys), parameter :: dwds = dw/ds
      real (kind=kind_phys), parameter :: t0c1  =273.15    ! freezing temp of fresh ice (k)
      real (kind=kind_phys), parameter :: ks   = 0.31     ! conductivity of snow   (w/mk)
      real (kind=kind_phys), parameter :: i0   = 0.3      ! ice surface penetrating solar fraction
      real (kind=kind_phys), parameter :: ki   = 2.03     ! conductivity of ice  (w/mk)
      real (kind=kind_phys), parameter :: di   = 917.0    ! density of ice   (kg/m^3)
      real (kind=kind_phys), parameter :: didw = di/dw
      real (kind=kind_phys), parameter :: dsdi = ds/di
      real (kind=kind_phys), parameter :: ci   = 2054.0   ! heat capacity of fresh ice (j/kg/k)
      real (kind=kind_phys), parameter :: li   = 3.34e5   ! latent heat of fusion (j/kg-ice)
      real (kind=kind_phys), parameter :: si   = 1.0      ! salinity of sea ice
      real (kind=kind_phys), parameter :: mu   = 0.054    ! relates freezing temp to salinity
      real (kind=kind_phys), parameter :: tfi  = -mu*si   ! sea ice freezing temp = -mu*salinity
      real (kind=kind_phys), parameter :: tfw  = -1.8     ! tfw - seawater freezing temp (c)
      real (kind=kind_phys), parameter :: tfi0 = tfi-0.0001
      real (kind=kind_phys), parameter :: dici = di*ci
      real (kind=kind_phys), parameter :: dili = di*li
      real (kind=kind_phys), parameter :: dsli = ds*li
      real (kind=kind_phys), parameter :: ki4  = ki*4.0


!  ---  inputs:
      integer, intent(in) :: im, km, lsm, myim(my_max), lev, ncld, async_id
!     logical, intent(in) :: lprnt

      real(kind=RTYPE), dimension(im, lev, my_max) ::     u1, v1, t1
      real(kind=RTYPE), dimension(im, lev*ncld, my_max) :: q1
      real (kind=kind_phys), dimension(im, my_max), intent(in) :: ps,           & 
             sfcemis, dlwflx, sfcnsw, sfcdsw, srflag, cm, ch,           & 
             prsl1, prslki, ddvel

      integer, dimension(im, my_max), intent(in) :: islimsk
      real (kind=kind_phys), intent(in)  :: delt

      logical, intent(in) :: flag_iter(im, my_max), mom4ice

!  ---  input/outputs:
      real (kind=kind_phys), dimension(im, my_max), intent(inout) :: hice,      & 
             fice, tice, weasd, tskin, tprcp, ep

      real (kind=kind_phys), dimension(im, km, my_max), intent(inout) :: stc

!  ---  outputs:
!     real (kind=kind_phys), dimension(im), intent(out) :: snwdph,      & 
      real (kind=kind_phys), dimension(im, my_max) :: snwdph,      & 
             qsurf, snowmt, gflux, evap, hflx

!  ---  locals:
      real (kind=kind_phys) :: ffw, evapi, evapw,        & 
             sneti, snetw, hfd, hfi,                                    &
!    &       hflxi, hflxw, sneti, snetw, qssi, qssw, hfd, hfi, hfw,     & 
             focn, snof, hi_save, hs_save,                 rch, rho,    & 
             snowd, theta1, cmm, chh
      real(kind_phys) :: ttmp
      real (kind=kind_phys) :: t12, t14, tem, stsice(kmi) &
      ,                   hflxi, hflxw, q0, qs1, wind, qssi, qssw,fpvs_gpu

      integer :: i, k, ipr, jj
      ! ice3lay
      real (kind=kind_phys) :: dt2, dt4, dt6, h1, h2, dh, wrk, wrk1,      &
                               dt2i, hdi, hsni, ai, bi, a1, b1, a10, b10  &
      ,                        c1, ip, k12, k32, tsf, f1, tmelt, bmelt

      logical :: flag, lprnt
      integer, parameter :: nxpvs = 7501
      real :: c1xpvs,c2xpvs,tbpvs(nxpvs)
      common/fpvscom/ c1xpvs,c2xpvs,tbpvs(nxpvs)
      
      !register of GPU
      real(kind=RTYPE) :: u1r, v1r, t1r, q1r
      real(kind=kind_phys) :: psr, sfcemisr, dlwflxr, sfcnswr, sfcdswr, &
         srflagr, cmr, chr, prsl1r, prslkir, ddvelr
      integer :: islimskr
      logical :: flag_iterr
      real(kind=kind_phys) :: hicer, ficer, ticer, weasdr, tskinr, tprcpr, epr
      real(kind=kind_phys) :: snwdphr, qsurfr, snowmtr, gfluxr, evapr, hflxr
      
!
!===> ...  begin here
!
      dt2  = 2.0 * delt
      dt4  = 4.0 * delt
      dt6  = 6.0 * delt
      dt2i = 1.0 / dt2


      !$acc parallel loop gang vector collapse(2) private(jj, i, k, wind, q0, &
      !$acc&         ttmp, qs1, qssi, qssw, t12, t14, hdi, hsni, &
      !$acc&         ip, tsf, bi, ai, k12, k32, wrk, a10, b10, wrk1, a1, b1, &
      !$acc&         c1, tmelt, bmelt, h1, h2, dh, f1, hflxi, hflxw, &
      !$acc&         tem, hi_save, hs_save, flag, theta1, rho, ffw, &
      !$acc&         snowd, cmm, chh, rch, evapi, evapw, snetw, sneti, hfi, hfd, &
      !$acc&         focn, snof, stsice) async(async_id)
      do jj = 1, jlistnum
         do i = 1, im
           if (i .le. myim(jj)) then
               flag = (islimsk(i, jj) >= 2) .and. flag_iter(i, jj)
               if (flag_iter(i, jj) .and. islimsk(i, jj) < 2) then
                  hice(i, jj) = 0.0
                  fice(i, jj) = 0.0
               endif
               if (flag) then
                  !$acc loop seq
                   do  k = 1, kmi
      !  --- ...  update sea ice temperature
                     stsice(k) = stc(i,k, jj)
                  enddo
                  if (mom4ice) then
                     hi_save = hice(i, jj)
                     hs_save = weasd(i, jj) * 0.001
                  elseif (lsm > 0) then           !  --- ...  snow-rain detection
                     if (srflag(i, jj) == 1.0) then
                        ep(i, jj) = 0.0
                        weasd(i, jj) = weasd(i, jj) + 1.e3*tprcp(i, jj)
                        tprcp(i, jj)  = 0.0
                     endif
                  endif
   !  --- ...  initialize variables. all units are supposedly m.k.s. unless specifie
   !           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
   !           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
   !           is sat. hum. at surface
   !           convert slrad to the civilized unit from langley minute-1 k-4
         !        psurf(i) = 1000.0 * ps(i)
         !        ps1(i)   = 1000.0 * prsl1(i)

         !        dlwflx has been given a negative sign for downward longwave
         !        sfcnsw is the net shortwave flux (direction: dn-up)

                  wind      = max(sqrt(u1(i, lev, jj)*u1(i, lev, jj) + v1(i, lev, jj)*v1(i, lev, jj))               & 
                                + max(0.0, min(ddvel(i, jj), 30.0)), 1.0)

                  q0        = max(q1(i, lev, jj), 1.0e-8)
         !        tsurf(i)  = tskin(i)
                  theta1 = t1(i, lev, jj) * prslki(i, jj)
                  rho    = prsl1(i, jj) / (rd*t1(i, lev, jj)*(1.0+rvrdm1*q0))
                  ttmp      = t1(i, lev, jj)
                  qs1       = fpvs_gpu(ttmp, c1xpvs,c2xpvs,tbpvs)
                  qs1       = max(eps*qs1 / (prsl1(i, jj) + epsm1*qs1), 1.e-8)
                  q0        = min(qs1, q0)

                  ffw    = 1.0 - fice(i, jj)
                  if (fice(i, jj) < cimin) then
                     print *,'warning: ice fraction is low:', fice(i, jj)
                     fice(i, jj) = cimin
                     ffw  = 1.0 - fice(i, jj)
                     tice(i, jj) = tgice
                     tskin(i, jj)= tgice
                     print *,'fix ice fraction: reset it to:', fice(i, jj)
                  endif

                  qssi = fpvs_gpu(tice(i, jj), c1xpvs,c2xpvs,tbpvs)
                  qssi = eps*qssi / (ps(i, jj) + epsm1*qssi)
                  qssw = fpvs_gpu(tgice, c1xpvs,c2xpvs,tbpvs)
                  qssw = eps*qssw / (ps(i, jj) + epsm1*qssw)

         !  --- ...  snow depth in water equivalent is converted from mm to m unit

                  if (mom4ice) then
                     snowd = weasd(i, jj) * 0.001 / fice(i, jj)
                  else
                     snowd = weasd(i, jj) * 0.001
                  endif
         !         flagsnw(i) = .false.

         !  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
         !           soil is allowed to interact with the atmosphere.
         !           we should eventually move to a linear combination of soil and
         !           snow under the condition of patchy snow.

         !  --- ...  rcp = rho cp ch v

                  cmm = cm(i, jj)  * wind
                  chh = rho * ch(i, jj) * wind
                  rch = chh * cp

         !  --- ...  sensible and latent heat flux over open water & sea ice

                  evapi = elocp * rch * (qssi - q0)
                  evapw = elocp * rch * (qssw - q0)
         !        evap(i)  = fice(i)*evapi(i) + ffw(i)*evapw(i)

         !     if (lprnt) write(0,*)' tice=',tice(ipr)

                  snetw = sfcdsw(i, jj) * (1.0 - albfw)
                  snetw = min(3.0*sfcnsw(i, jj)/(1.0+2.0*ffw), snetw)
                  sneti = (sfcnsw(i, jj) - ffw*snetw) / fice(i, jj)

                  t12 = tice(i, jj) * tice(i, jj)
                  t14 = t12 * t12

         !  --- ...  hfi = net non-solar and upir heat flux @ ice surface

                  hfi = -dlwflx(i, jj) + sfcemis(i, jj)*sbc*t14 + evapi           & 
                         + rch*(tice(i, jj) - theta1)
                  hfd = 4.0*sfcemis(i, jj)*sbc*tice(i, jj)*t12                       & 
                         + (1.0 + elocp*eps*hvap*qs1/(rd*t12)) * rch

                  t12 = tgice * tgice
                  t14 = t12 * t12

         !  --- ...  hfw = net heat flux @ water surface (within ice)

         !         hfw(i) = -dlwflx(i) + sfcemis(i)*sbc*t14 + evapw(i)           &
         !    &           + rch(i)*(tgice - theta1(i)) - snetw(i)

                  focn = 2.0     ! heat flux from ocean - should be from ocn model
                  snof = 0.0     ! snowfall rate - snow accumulates in gbphys

                  hice(i, jj) = max( min( hice(i, jj), himax ), himin )
                  snowd = min( snowd, hsmax )

                  if (snowd > (2.0*hice(i, jj))) then
                     print *, 'warning: too much snow :',snowd
                     snowd = hice(i, jj) + hice(i, jj)
                     print *,'fix: decrease snow depth to:',snowd
                  endif
         !     if (lprnt) write(0,*)' tice2=',tice(ipr)
         !call ice3lay
   !  ---  inputs:                                                         !
   !    &     ( myim(jj), im, kmi, fice, flag, hfi, hfd, sneti, focn, delt,     !
   !  ---  outputs:                                                        !
   !    &       snowd, hice, stsice, tice, snof, snowmt, gflux )           !

   !     if (lprnt) write(0,*)' tice3=',tice(ipr)

                  snowd = snowd * dwds
                  hdi      = (dsdw*snowd + didw*hice(i, jj))

                  if (hice(i, jj) < hdi) then
                     snowd = snowd + hice(i, jj) - hdi
                     hsni     = (hdi - hice(i, jj)) * dsdi
                     hice (i, jj) = hice(i, jj) + hsni
                  endif

                  snof     = snof * dwds
                  tice(i, jj)     = tice(i, jj) - t0c1
                  stsice(1) = min(stsice(1)-t0c1, tfi0)     ! degc
                  stsice(2) = min(stsice(2)-t0c1, tfi0)     ! degc

                  ip = i0 * sneti         ! ip +v (in winton ip=-i0*sneti as sol -v)
                  if (snowd > 0.0) then
                     tsf = 0.0
                     ip  = 0.0
                  else
                     tsf = tfi
                     ip  = i0 * sneti      ! ip +v here (in winton ip=-i0*sneti)
                  endif
                  tice(i, jj) = min(tice(i, jj), tsf)

         !  --- ...  compute ice temperature

                  bi   = hfd
                  ai   = hfi - sneti + ip - tice(i, jj)*bi  ! +v sol input here
                  k12  = ki4*ks / (ks*hice(i, jj) + ki4*snowd)
                  k32  = (ki+ki) / hice(i, jj)

                  wrk    = 1.0 / (dt6*k32 + dici*hice(i, jj))
                  a10    = dici*hice(i, jj)*dt2i + k32*(dt4*k32 + dici*hice(i, jj))*wrk
                  b10    = -di*hice(i, jj) * (ci*stsice(1) + li*tfi/stsice(1))    &
                         * dt2i - ip                                              &
                         - k32*(dt4*k32*tfw + dici*hice(i, jj)*stsice(2)) * wrk

                  wrk1  = k12 / (k12 + bi)
                  a1    = a10 + bi * wrk1
                  b1    = b10 + ai * wrk1
                  c1    = dili * tfi * dt2i * hice(i, jj)

                  stsice(1) = -(sqrt(b1*b1 - 4.0*a1*c1) + b1)/(a1+a1)
                  tice(i, jj) = (k12*stsice(1) - ai) / (k12 + bi)

                  !!! top ice melt
                  if (tice(i, jj) > tsf) then
                     a1 = a10 + k12
                     b1 = b10 - k12*tsf
                     stsice(1) = -(sqrt(b1*b1 - 4.0*a1*c1) + b1)/(a1+a1)
                     tice(i, jj) = tsf
                     tmelt   = (k12*(stsice(1)-tsf) - (ai+bi*tsf)) * delt
                  else
                     tmelt    = 0.0
                     snowd = snowd + snof*delt
                  endif

                  stsice(2) = (dt2*k32*(stsice(1) + tfw + tfw)                &
                              +  dici*hice(i, jj)*stsice(2)) * wrk

                  !!! bottom melt
                  bmelt = (focn + ki4*(stsice(2) - tfw)/hice(i, jj)) * delt

         !  --- ...  resize the ice ...

                  h1 = 0.5 * hice(i, jj)
                  h2 = 0.5 * hice(i, jj)

         !  --- ...  top ...

                  if (tmelt <= snowd*dsli) then
                     snowmt(i, jj) = tmelt / dsli
                     snowd  = snowd - snowmt(i, jj)
                  else
                     snowmt(i, jj) = snowd
                     h1 = h1 - (tmelt - snowd*dsli)                             &
                        / (di * (ci - li/stsice(1)) * (tfi - stsice(1)))
                     snowd = 0.0
                  endif

         !  --- ...  and bottom

                  if (bmelt < 0.0) then
                     dh = -bmelt / (dili + dici*(tfi - tfw))
                     stsice(2) = (h2*stsice(2) + dh*tfw) / (h2 + dh)
                     h2 = h2 + dh
                  else
                     h2 = h2 - bmelt / (dili + dici*(tfi - stsice(2)))
                  endif

         !  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

                  hice(i, jj) = h1 + h2

                  if (hice(i, jj) > 0.0) then
                     if (h1 > 0.5*hice(i, jj)) then
                        f1 = 1.0 - (h2+h2) / hice(i, jj)
                        stsice(2) = f1 * (stsice(1) + li*tfi/(ci*stsice(1)))  &
                                    + (1.0 - f1)*stsice(2)

                        if (stsice(2) > tfi) then
                           hice(i, jj) = hice(i, jj) - h2*ci*(stsice(2) - tfi)/ (li*delt)
                           stsice(2) = tfi
                        endif
                     else
                        f1 = (h1+h1) / hice(i, jj)
                        stsice(1) = f1 * (stsice(1) + li*tfi/(ci*stsice(1)))  &
                                    + (1.0 - f1)*stsice(2)
                        stsice(1) = (stsice(1) - sqrt(stsice(1)*stsice(1)   &
                                    - 4.0*tfi*li/ci)) * 0.5
                     endif

                     k12      = ki4*ks / (ks*hice(i, jj) + ki4*snowd)
                     gflux(i, jj) = k12 * (stsice(1) - tice(i, jj))
                  else
                     snowd = snowd + (h1*(ci*(stsice(1) - tfi)             &
                              - li*(1.0 - tfi/stsice(1)))                        &
                              + h2*(ci*(stsice(2) - tfi) - li)) / li

                     hice(i, jj)     = max(0.0, snowd*dsdi)
                     snowd    = 0.0
                     stsice(1) = tfw
                     stsice(2) = tfw
                     gflux(i, jj)    = 0.0
                  endif   ! end if_hice_block

                  gflux(i, jj)    = fice(i, jj) * gflux(i, jj)
                  snowmt(i, jj)   = snowmt(i, jj) * dsdw
                  snowd    = snowd * dsdw
                  tice(i, jj)     = tice(i, jj) + t0c1
                  stsice(1) = stsice(1) + t0c1
                  stsice(2) = stsice(2) + t0c1
                  
                  
                  if (mom4ice) then
                     hice(i, jj)  = hi_save
                     snowd = hs_save
                  endif
                
               if (tice(i, jj) < timin) then
                  print *,'warning: snow/ice temperature is too low:',tice(i, jj) &
            ,' i=',i
                  tice(i, jj) = timin
                  print *,'fix snow/ice temperature: reset it to:',tice(i, jj)
               endif

               if (stsice(1) < timin) then
                  print *,'warning: layer 1 ice temp is too low:',stsice(1) &
            ,' i=',i
                  stsice(1) = timin
                  print *,'fix layer 1 ice temp: reset it to:',stsice(1)
               endif

               if (stsice(2) < timin) then
                  print *,'warning: layer 2 ice temp is too low:',stsice(2)
                  stsice(2) = timin
                  print *,'fix layer 2 ice temp: reset it to:',stsice(2)
               endif

               tskin(i, jj) = tice(i, jj)*fice(i, jj) + tgice*ffw
                
                  do k = 1, kmi
                     stc(i,k, jj) = min(stsice(k), t0c)
                  enddo
                  
         !  --- ...  calculate sensible heat flux (& evap over sea ice)

                  hflxi    = rch * (tice(i, jj) - theta1)
                  hflxw    = rch * (tgice - theta1)
                  hflx(i, jj)  = fice(i, jj)*hflxi    + ffw*hflxw
                  evap(i, jj)  = fice(i, jj)*evapi + ffw*evapw
         !
         !  --- ...  the rest of the output

                  qsurf(i, jj) = q1(i, lev, jj) + evap(i, jj) / (elocp*rch)

         !  --- ...  convert snow depth back to mm of water equivalent

                  weasd(i, jj)  = snowd * 1000.0
                  snwdph(i, jj) = weasd(i, jj) * dsi             ! snow depth in mm

                  tem     = 1.0 / rho
                  hflx(i, jj) = hflx(i, jj) * tem * cpinv
                  evap(i, jj) = evap(i, jj) * tem * hvapi
               endif
           end if
         enddo
      end do

   !

              



!
      return

! =================
      !contains
! =================


!-----------------------------------
      !subroutine ice3lay
!...................................
!  ---  inputs:
!    &     ( myim(jj), im, kmi, fice, flag, hfi, hfd, sneti, focn, delt,          &
!  ---  input/outputs:
!    &       snowd, hice, stsice, tice, snof,                           &
!  ---  outputs:
!    &       snowmt, gflux                                              &
!    &     )

!**************************************************************************
!                                                                         *
!            three-layer sea ice vertical thermodynamics                  *
!                                                                         *
! based on:  m. winton, "a reformulated three-layer sea ice model",       *
! journal of atmospheric and oceanic technology, 2000                     *
!                                                                         *
!                                                                         *
!        -> +---------+ <- tice - diagnostic surface temperature ( <= 0c )*
!       /   |         |                                                   *
!   snowd   |  snow   | <- 0-heat capacity snow layer                     *
!       \   |         |                                                   *
!        => +---------+                                                   *
!       /   |         |                                                   *
!      /    |         | <- t1 - upper 1/2 ice temperature; this layer has *
!     /     |         |         a variable (t/s dependent) heat capacity  *
!   hice    |...ice...|                                                   *
!     \     |         |                                                   *
!      \    |         | <- t2 - lower 1/2 ice temp. (fixed heat capacity) *
!       \   |         |                                                   *
!        -> +---------+ <- base of ice fixed at seawater freezing temp.   *
!                                                                         *
!  =====================  defination of variables  =====================  !
!                                                                         !
!  inputs:                                                         size   !
!     im, kmi  - integer, horiz dimension and num of ice layers      1    !
!     fice     - real, sea-ice concentration                         im   !
!     flag     - logical, ice mask flag                              1    !
!     hfi      - real, net non-solar and heat flux @ surface(w/m^2)  im   !
!     hfd      - real, heat flux derivatice @ sfc (w/m^2/deg-c)      im   !
!     sneti    - real, net solar incoming at top  (w/m^2)            im   !
!     focn     - real, heat flux from ocean    (w/m^2)               im   !
!     delt     - real, timestep                (sec)                 1    !
!                                                                         !
!  input/outputs:                                                         !
!     snowd    - real, surface pressure                              im   !
!     hice     - real, sea-ice thickness                             im   !
!     stsice   - real, temp @ midpt of ice levels  (deg c)          im,kmi!
!     tice     - real, surface temperature     (deg c)               im   !
!     snof     - real, snowfall rate           (m/sec)               im   !
!                                                                         !
!  outputs:                                                               !
!     snowmt   - real, snow melt during delt   (m)                   im   !
!     gflux    - real, conductive heat flux    (w/m^2)               im   !
!                                                                         !
!  locals:                                                                !
!     hdi      - real, ice-water interface     (m)                        !
!     hsni     - real, snow-ice                (m)                        !
!                                                                         !
! ======================================================================= !
!

!  ---  constant parameters: (properties of ice, snow, and seawater)

!  ---  inputs:
!     integer, intent(in) :: im, kmi

!     real (kind=kind_phys), dimension(im), intent(in) :: fice, hfi,    &
!    &       hfd, sneti, focn

!     real (kind=kind_phys), intent(in) :: delt

!     logical, dimension(im), intent(in) :: flag

!  ---  input/outputs:
!     real (kind=kind_phys), dimension(im), intent(inout) :: snowd,     &
!    &       hice, tice, snof

!     real (kind=kind_phys), dimension(im,kmi), intent(inout) :: stsice

!  ---  outputs:
!     real (kind=kind_phys), dimension(im), intent(out) :: snowmt,      &
!    &       gflux

!  ---  locals:


      !integer :: i
!
!===> ...  begin here
!

      !return
!...................................
      !end subroutine ice3lay
!-----------------------------------

! =========================== !
!     end contain programs    !
! =========================== !

!...................................
      end subroutine sfc_sice_gpu
!-----------------------------------
