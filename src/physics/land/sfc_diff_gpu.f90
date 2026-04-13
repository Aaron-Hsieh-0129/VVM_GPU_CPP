      subroutine sfc_diff_gpu(myim, im, lev, ncld, ps, u1, v1, t1, q1, z1, &
                          snwdph, tskin, z0rl, cm, ch, rb, &
                          prsl1, prslki, islimsk, &
                          stress, fm, fh, &
                          ustar, wind, ddvel, fm10, fh2, fh10, &
                          sigmaf, vegtype, shdmax, ivegsrc, &
#ifdef TIMCOMCPL
                          tsurf, flag_iter, redrag, async_id, ustress,vstress,ssu,ssv)
#else 
                          tsurf, flag_iter, redrag, async_id)
#endif
      !$acc routine(fpvs_gpu) seq
!
         use machine, only: kind_phys
!     use funcphys, only : fpvs
         use physcons, grav => con_g, cp => con_cp &
           , rvrdm1 => con_fvirt, rd => con_rd &
           , eps => con_eps, epsm1 => con_epsm1
         use const, only: RTYPE
         use param, only: my, my_max
         use index

         implicit none
!
         integer im, ivegsrc, myim(my_max), lev, ncld, jj, async_id, idx(1), ii
         real(kind=RTYPE), dimension(im, lev, my_max) ::     u1, v1, t1
         real(kind=RTYPE), dimension(im, lev*ncld, my_max) :: q1
         real(kind=kind_phys), dimension(im, my_max) :: ps &
                                       , z1, tskin, z0rl &
                                       , cm, ch, rb, prsl1, prslki &
                                       , stress, fm, fh, ustar &
                                       , wind, ddvel, fm10, fh2, fh10 &
                                       , sigmaf, shdmax, tsurf, snwdph
                                       
         !real(kind=kind_phys), dimension(im, my_max) :: ustress, vstress
#ifdef TIMCOMCPL
         real(kind=kind_phys), dimension(im, my_max) :: ustress, vstress, ssu, ssv, &
                                                        cpl_u1, cpl_v1
#endif      
         integer, dimension(im, my_max) ::  vegtype, islimsk
         integer, dimension(im*my_max) :: jj_idx, i_idx

         logical flag_iter(im, my_max) ! added by s.lu
         logical redrag                ! reduced drag coeff. flag for high wind over sea (j.han)
!
!     locals
!
         integer i
!
         real(kind=kind_phys) aa, aa0, bb, bb0, dtv, adtv, qs1, &
            hl1, hl12, pm, ph, pm10, ph2, ph10, rat, &
            thv1, tvs, z1i, z0, z0max, ztmax, &
            fms, fhs, hl0, hl0inf, hlinf, &
            hl110, hlt, hltinf, olinf, &
            restar, czilc, tem1, tem2, &
            ztmin1, ztmax1, beta, hmgn, fpvs_gpu
            
!
         real(kind=kind_phys), parameter :: &
            charnock = .014, ca = .4 &! ca - von karman constant
            , z0s_max = .317e-2 &! a limiting value at high winds over sea
            , alpha = 5., a0 = -3.975, a1 = 12.32, alpha4 = 4.0*alpha &
            , b1 = -7.755, b2 = 6.041 &
            , a0p = -7.941, a1p = 24.75, b1p = -8.705, b2p = 7.899 &
            , vis = 1.4e-5, rnu = 1.51e-5, visi = 1.0/vis &
            , log01 = log(0.01), log05 = log(0.05), log07 = log(0.07)

         ! GPU register
         real(kind=kind_phys) :: z0rlr, cmr, chr, rbr, stressr, fmr, &
            fhr, ustarr, windr, fm10r, fh2r, fh10r, u1r, v1r, t1r, q1r, &
            psr, z1r, tskinr, prsl1r, prslkir, ddvelr, sigmafr, shdmaxr, &
            tsurfr, snwdphr
         integer::  vegtyper, islimskr
#ifdef TIMCOMCPL
         real(kind=kind_phys) :: cpl_u1r, cpl_v1r, ssur, ssvr, ustressr, vstressr
#endif         


         integer, parameter :: nxpvs = 7501
         real :: c1xpvs,c2xpvs,tbpvs(nxpvs)
         common/fpvscom/ c1xpvs,c2xpvs,tbpvs(nxpvs)
         real :: time1, time2


!     parameter (charnock=.014,ca=.4)!c ca is the von karman constant
!     parameter (alpha=5.,a0=-3.975,a1=12.32,b1=-7.755,b2=6.041)
!     parameter (a0p=-7.941,a1p=24.75,b1p=-8.705,b2p=7.899,vis=1.4e-5)

!     real(kind=kind_phys) aa1,bb1,bb2,cc,cc1,cc2,arnu
!     parameter (aa1=-1.076,bb1=.7045,cc1=-.05808)
!     parameter (bb2=-.1954,cc2=.009999)
!     parameter (arnu=.135*rnu)
!
!    z0s_max=.196e-2 for u10_crit=25 m/s
!    z0s_max=.317e-2 for u10_crit=30 m/s
!    z0s_max=.479e-2 for u10_crit=35 m/s
!
! mbek -- toga-coare flux algorithm
!     parameter (rnu=1.51e-5,arnu=0.11*rnu)
!
!  initialize variables. all units are supposedly m.k.s. unless specified
!  ps is in pascals, wind is wind speed,
!  surface roughness length is converted to m from cm
!
         
         

         !call cpu_time(time1)
         !!$acc enter data create(idx, jj_idx, i_idx) async(async_id)
         !!$acc serial async(async_id)
         !idx(1) = 0
         !!$acc end serial
         !!$acc parallel loop gang vector collapse(2) private(jj,i,ii) async(async_id) 
         !do jj = 1, jlistnum
         !   do i = 1, im
         !      if (i .le. myim(jj)) then
         !         if (flag_iter(i, jj)) then
         !            !$acc atomic capture
         !            idx(1) = idx(1) + 1
         !            ii = idx(1)
         !            !$acc end atomic
         !            jj_idx(ii) = jj
         !            i_idx(ii) = i
         !         end if
         !      end if
         !   end do
         !end do
         !call cpu_time(time2)
         
         
#ifdef TIMCOMCPL
      !$acc enter data create(cpl_u1, cpl_v1) async(async_id)
#endif
         
         !$acc parallel loop gang vector collapse(2) private(tem1, thv1, tvs,&
         !$acc&         z0, z0max, ii, jj, i, &
         !$acc&         z1i, restar, rat, ztmax, tem2, ztmin1, beta, hmgn, &
         !$acc&         dtv, adtv, hlinf, ztmax1, hl1, hl0inf, hltinf, aa, &
         !$acc&         aa0, bb, bb0, pm, ph, fms, fhs, hl0, hlt, hl110, pm10, &
         !$acc&         hl12, ph2, ph10, olinf, qs1, z0rlr, cmr, chr, rbr, &
         !$acc&         stressr, fmr, fhr, ustarr, windr, fm10r, fh2r, fh10r, &
         !$acc&         u1r, v1r, t1r, q1r, vegtyper, islimskr, psr, z1r, &
         !$acc&         tskinr, prsl1r, prslkir, ddvelr, sigmafr, shdmaxr, &
#ifdef TIMCOMCPL
         !$acc&         ustressr, vstressr, cpl_u1r, cpl_v1r, ssur, ssvr, &
#endif
         !$acc&         tsurfr, snwdphr) &
         !$acc&         async(async_id)       
         do jj = 1, jlistnum
            do i = 1, im
               if (i .le. myim(jj)) then
#ifdef TIMCOMCPL
                  cpl_u1(i, jj)=0.
                  cpl_v1(i, jj)=0.
                  cpl_u1(i, jj)=u1(i, lev, jj)
                  cpl_v1(i, jj)=v1(i, lev, jj)
#endif
                  if (flag_iter(i, jj)) then
                  !do ii = 1, idx(1)
                  !   jj = jj_idx(ii)
                  !   i = i_idx(ii)
                     ! from global memory to register
                     u1r = u1(i, lev, jj)
                     v1r = v1(i, lev, jj)
                     t1r = t1(i, lev, jj)
                     q1r = q1(i, lev, jj)
                     z0rlr = z0rl(i, jj)
                     ustarr = ustar(i, jj)
                     vegtyper = vegtype(i, jj)
                     islimskr = islimsk(i, jj)
                     psr = ps(i, jj)
                     z1r = z1(i, jj)
                     tskinr = tskin(i, jj)
                     prsl1r = prsl1(i, jj)
                     prslkir = prslki(i, jj)
                     ddvelr = ddvel(i, jj)
                     sigmafr = sigmaf(i, jj)
                     shdmaxr = shdmax(i, jj)
                     tsurfr = tsurf(i, jj)
                     snwdphr = snwdph(i, jj)
#ifdef TIMCOMCPL
                     cpl_u1r = cpl_u1(i, jj)
                     cpl_v1r = cpl_v1(i, jj)
                     ssur = ssu(i, jj)
                     ssvr = ssv(i, jj)
#endif

#ifdef TIMCOMCPL
                     windr = max(sqrt((cpl_u1r-ssur)**2 + (cpl_v1r-ssvr)**2) &
                             + max(0.0, min(ddvelr, 30.0)), 1.0)
#else
                     windr = max(sqrt(u1r*u1r &
                                   + v1r*v1r) &
                                   + max(0.0, min(ddvelr, 30.0)), 1.0)
#endif
                     tem1 = 1.0 + rvrdm1*max(q1r, 1.e-8)
                     thv1 = t1r*prslkir*tem1
                     tvs = 0.5*(tsurfr + tskinr)*tem1
                     qs1 = fpvs_gpu(t1r,c1xpvs,c2xpvs,tbpvs)
                     qs1 = max(1.0e-8, eps*qs1/(prsl1r + epsm1*qs1))

                     z0 = 0.01*z0rlr
                     z0max = max(1.0e-6, min(z0, z1r))
                     z1i = 1.0/z1r

         !  compute stability dependent exchange coefficients
         !  this portion of the code is presently suppressed
         !

                     if (islimskr == 0) then            ! over ocean
                        ustarr = sqrt(grav*z0/charnock)

         !**  test xubin's new z0

         !           ztmax  = z0max

                        restar = max(ustarr*z0max*visi, 0.000001)

         !           restar = log(restar)
         !           restar = min(restar,5.)
         !           restar = max(restar,-5.)
         !           rat    = aa1 + (bb1 + cc1*restar) * restar
         !           rat    = rat    / (1. + (bb2 + cc2*restar) * restar))
         !  rat taken from zeng, zhao and dickinson 1997

                        rat = min(7.0, 2.67*sqrt(sqrt(restar)) - 2.57)
                        ztmax = z0max*exp(-rat)

                     else                                ! over land and sea ice
         !** xubin's new z0  over land and sea ice
                        tem1 = 1.0 - shdmaxr
                        tem2 = tem1*tem1
                        tem1 = 1.0 - tem2

                        if (ivegsrc .ge. 1) then

                           if (vegtyper == 10) then
                              z0max = exp(tem2*log01 + tem1*log07)
                           elseif (vegtyper == 6) then
                              z0max = exp(tem2*log01 + tem1*log05)
                           elseif (vegtyper == 7) then
         !           z0max = exp( tem2*log01 + tem1*log01 )
                              z0max = 0.01
                           elseif (vegtyper == 16) then
         !           z0max = exp( tem2*log01 + tem1*log01 )
                              z0max = 0.01
                           else
                              if (islimskr == 2) then
                                 z0max = exp(tem2*log(0.0002) + tem1*log(z0max))
                              else
                                 z0max = exp(tem2*log01 + tem1*log(z0max))
                              end if
                           end if

                        elseif (ivegsrc == 0) then

                           if (vegtyper == 7) then
                              z0max = exp(tem2*log01 + tem1*log07)
                           elseif (vegtyper == 8) then
                              z0max = exp(tem2*log01 + tem1*log05)
                           elseif (vegtyper == 9) then
         !             z0max = exp( tem2*log01 + tem1*log01 )
                              z0max = 0.01
                           elseif (vegtyper == 11) then
         !             z0max = exp( tem2*log01 + tem1*log01 )
                              z0max = 0.01
                           else
                              if (islimskr == 2) then
                                 z0max = exp(tem2*log(0.0002) + tem1*log(z0max))
                              else
                                 z0max = exp(tem2*log01 + tem1*log(z0max))
                              end if
                           end if

                        end if
                        z0max = max(z0max, 1.0e-6)
         !
         !           czilc = 10.0 ** (- (0.40/0.07) * z0) ! fei's canopy height dependance of czil
                        czilc = 0.8

                        tem1 = 1.0 - sigmafr
                        ztmax = z0max*exp(-tem1*tem1 &
                                          *czilc*ca*sqrt(ustarr*(0.01/1.5e-05)))

                     end if       ! end of if(islimsk(i) == 0) then
                     ztmax = max(ztmax, 1.0e-6)
                     ztmin1 = -999.0
                     beta = 1.0
         !xb118>>
                     tem1 = z0max/z1r
                     if (abs(1.0 - tem1) > 1.0e-6) then
                        hmgn = -beta*log(tem1)/(2.*alpha*(1.-tem1))
                     else
                        hmgn = 99.0
                     end if
         !            hmgn   = beta*log(z1(i)/z0max)/(2.*alpha*(1.-z0max/z1(i)))
         !xb118<<
                     if (z0max .lt. 0.05 .and. snwdphr .lt. 10.0) hmgn = 99.0

         !  compute stability indices (rb and hlinf)

                     dtv = thv1 - tvs
                     adtv = max(abs(dtv), 0.001)
                     dtv = sign(1., dtv)*adtv
                     rbr = max(-5000.0, (grav + grav)*dtv*z1r &
                                 /((thv1 + tvs)*windr*windr))
                     tem1 = 1.0/z0max
                     tem2 = 1.0/ztmax
                     fmr = log((z0max + z1r)*tem1)
                     fhr = log((ztmax + z1r)*tem2)
                     fm10r = log((z0max + 10.)*tem1)
                     fh2r = log((ztmax + 2.)*tem2)
                     fh10r = log((ztmax + 10.)*tem2)
                     hlinf = rbr*fmr*fmr/fhr
                     ztmax1 = hmgn
                     hlinf = min(max(hlinf, ztmin1), ztmax1)
         !
         !  stable case
         !
                     if (dtv >= 0.0) then
                        hl1 = hlinf
                        if (hlinf > .25) then
                           tem1 = hlinf*z1i
                           hl0inf = z0max*tem1
                           hltinf = ztmax*tem1
                           aa = sqrt(1.+alpha4*hlinf)
                           aa0 = sqrt(1.+alpha4*hl0inf)
                           bb = aa
                           bb0 = sqrt(1.+alpha4*hltinf)
                           pm = aa0 - aa + log((aa + 1.)/(aa0 + 1.))
                           ph = bb0 - bb + log((bb + 1.)/(bb0 + 1.))
                           fms = fmr - pm
                           fhs = fhr - ph
                           hl1 = fms*fms*rbr/fhs
                           ztmax1 = hmgn
                           hl1 = min(max(hl1, ztmin1), ztmax1)
                        end if
         !
         !  second iteration
         !
                        tem1 = hl1*z1i
                        hl0 = z0max*tem1
                        hlt = ztmax*tem1
                        aa = sqrt(1.+alpha4*hl1)
                        aa0 = sqrt(1.+alpha4*hl0)
                        bb = aa
                        bb0 = sqrt(1.+alpha4*hlt)
                        pm = aa0 - aa + log((1.0 + aa)/(1.0 + aa0))
                        ph = bb0 - bb + log((1.0 + bb)/(1.0 + bb0))
                        hl110 = hl1*10.*z1i
                        ztmax1 = hmgn
                        hl110 = min(max(hl110, ztmin1), ztmax1)
                        aa = sqrt(1.+alpha4*hl110)
                        pm10 = aa0 - aa + log((1.0 + aa)/(1.0 + aa0))
                        hl12 = (hl1 + hl1)*z1i
                        hl12 = min(max(hl12, ztmin1), ztmax1)
         !           aa    = sqrt(1. + alpha4 * hl12)
                        bb = sqrt(1.+alpha4*hl12)
                        ph2 = bb0 - bb + log((1.0 + bb)/(1.0 + bb0))
                        bb = sqrt(1.+alpha4*hl110)
                        ph10 = bb0 - bb + log((1.0 + bb)/(1.0 + bb0))
         !
         !  unstable case - check for unphysical obukhov length
         !
                     else                          ! dtv < 0 case
                        olinf = z1r/hlinf
                        tem1 = 50.0*z0max
                        if (abs(olinf) <= tem1) then
                           hlinf = -z1r/tem1
                           ztmax1 = hmgn
                           hlinf = min(max(hlinf, ztmin1), ztmax1)
                        end if
         !
         !  get pm and ph
         !
                        if (hlinf >= -0.5) then
                           ztmax1 = hmgn
                           hl1 = hlinf
                           pm = (a0 + a1*hl1)*hl1/(1.+(b1 + b2*hl1)*hl1)
                           ph = (a0p + a1p*hl1)*hl1/(1.+(b1p + b2p*hl1)*hl1)
                           hl110 = hl1*10.*z1i
                           hl110 = min(max(hl110, ztmin1), ztmax1)
                           pm10 = (a0 + a1*hl110)*hl110/(1.+(b1 + b2*hl110)*hl110)
                           hl12 = (hl1 + hl1)*z1i
                           hl12 = min(max(hl12, ztmin1), ztmax1)
                           ph2 = (a0p + a1p*hl12)*hl12/(1.+(b1p + b2p*hl12)*hl12)
                           ph10 = (a0p + a1p*hl110)*hl110/(1.+(b1p + b2p*hl110)*hl110)
                        else                       ! hlinf < 0.05
                           hl1 = -hlinf
                           tem1 = 1.0/sqrt(hl1)
                           pm = log(hl1) + 2.*sqrt(tem1) - .8776
                           ph = log(hl1) + .5*tem1 + 1.386
         !             pm    = log(hl1) + 2.0 * hl1 ** (-.25) - .8776
         !             ph    = log(hl1) + 0.5 * hl1 ** (-.5) + 1.386
                           hl110 = hl1*10.*z1i
                           hl110 = min(max(hl110, ztmin1), ztmax1)
                           pm10 = log(hl110) + 2.0/sqrt(sqrt(hl110)) - .8776
         !             pm10  = log(hl110) + 2. * hl110 ** (-.25) - .8776
                           hl12 = (hl1 + hl1)*z1i
                           hl12 = min(max(hl12, ztmin1), ztmax1)
                           ph2 = log(hl12) + 0.5/sqrt(hl12) + 1.386
                           ph10 = log(hl110) + 0.5/sqrt(hl110) + 1.386
         !             ph2   = log(hl12) + .5 * hl12 ** (-.5) + 1.386
                        end if

                     end if          ! end of if (dtv >= 0 ) then loop
         !
         !  finish the exchange coefficient computation to provide fm and fh
         !
                     fmr = fmr - pm
                     fhr = fhr - ph
                     fm10r = fm10r - pm10
                     fh2r = fh2r - ph2
                     fh10r = fh10r - ph10
                     cmr = ca*ca/(fmr*fmr)
                     chr = ca*ca/(fmr*fhr)
                     cmr = max(cmr, 0.00001/z1r)
                     chr = max(chr, 0.00001/z1r)
                     stressr = cmr*windr*windr
                     ustarr = sqrt(stressr)
         !! jwhwu 20110311
#ifdef TIMCOMCPL
                     ustressr = - stressr * (cpl_u1r-ssur) / windr
                     vstressr = - stressr * (cpl_v1r-ssvr) / windr
#else
                     !ustress(i, jj) = -stress(i, jj)*u1(i, lev, jj)/wind(i, jj)
                     !vstress(i, jj) = -stress(i, jj)*v1(i, lev, jj)/wind(i, jj)
#endif 
         !
         !  update z0 over ocean
         !
                     if (islimskr == 0) then
                        z0 = (charnock/grav)*ustarr*ustarr

         ! mbek -- toga-coare flux algorithm
         !           z0 = (charnock / grav) * ustar(i)*ustar(i) +  arnu/ustar(i)
         !  new implementation of z0
         !           cc = ustar(i) * z0 / rnu
         !           pp = cc / (1. + cc)
         !           ff = grav * arnu / (charnock * ustar(i) ** 3)
         !           z0 = arnu / (ustar(i) * ff ** pp)

                        if (redrag) then
                           z0rlr = 100.0*max(min(z0, z0s_max), 1.e-7)
                        else
                           z0rlr = 100.0*max(min(z0, .1), 1.e-7)
                        end if
                     end if
                  
                  !return value from register to global memory
                  z0rl(i, jj) = z0rlr
                  cm(i, jj) = cmr
                  ch(i, jj) = chr
                  rb(i, jj) = rbr
                  stress(i, jj) = stressr
                  fm(i, jj) = fmr
                  fh(i, jj) = fhr
                  ustar(i, jj) = ustarr
                  wind(i, jj) = windr
                  fm10(i, jj) = fm10r
                  fh2(i, jj) = fh2r
                  fh10(i, jj) = fh10r
#ifdef TIMCOMCPL
                  ustress(i, jj) = ustressr
                  vstress(i, jj) = vstressr
#endif 
                  end if
               end if
            end do
         end do
#ifdef TIMCOMCPL
         !$acc exit data delete(cpl_u1, cpl_v1) async(async_id)
#endif
         return
      end
