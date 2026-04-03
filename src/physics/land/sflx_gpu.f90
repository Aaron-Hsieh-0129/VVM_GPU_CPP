#define checkboundary(idx, low, upper, j1, i1) call check_boundary(idx, low, upper, j1, i1, __LINE__)

subroutine check_boundary(idx, low, upper, j1, i1, line)
   use rank, only: myrank
   implicit none

   integer, intent(in) :: line, j1, i1, idx, low, upper

   if (idx .gt. upper .or. idx .lt. low) then
      print '(7g0)', j1, ' ', i1, " Line: ", line, ' ', idx
   end if

end subroutine check_boundary


!-----------------------------------
      subroutine sflx_gpu &
         !...................................
         !  ---  inputs: &
         (nsoil, nx, myim, flag, flag_iter, couple, icein, ffrozp, dt, zlvl, sldpth, &
          swdn, swnet, lwdn, sfcems, sfcprs, sfctmp, &
          sfcspd, prcp, q2, q2sat, dqsdt2, th2, ivegsrc, &
          vegtyp, soiltyp, slopetyp, shdmin, alb, snoalb, &
          !  ---  input/outputs: &
          tbot, cmc, t1, stc, smc, sh2o, sneqv, ch, cm, z0, &
          !  ---  outputs: &
          nroot, shdfac, snowh, albedo, eta, sheat, ec, &
          edir, et, ett, esnow, drip, dew, beta, etp, ssoil, &
          flx1, flx2, flx3, runoff1, runoff2, runoff3, &
          snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq, &
          rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax, &
          async_id)

! ===================================================================== !
!  description:                                                         !
!                                                                       !
!  subroutine sflx - version 2.7:                                       !
!  sub-driver for "noah/osu lsm" family of physics subroutines for a    !
!  soil/veg/snowpack land-surface model to update soil moisture, soil   !
!  ice, soil temperature, skin temperature, snowpack water content,     !
!  snowdepth, and all terms of the surface energy balance and surface   !
!  water balance (excluding input atmospheric forcings of downward      !
!  radiation and precip)                                                !
!                                                                       !
!  usage:                                                               !
!                                                                       !
!      call sflx                                                        !
!  ---  inputs:                                                         !
!          ( nsoil, couple, icein, ffrozp, dt, zlvl, sldpth,            !
!            swdn, swnet, lwdn, sfcems, sfcprs, sfctmp,                 !
!            sfcspd, prcp, q2, q2sat, dqsdt2, th2,ivegsrc,              !
!            vegtyp, soiltyp, slopetyp, shdmin, alb, snoalb,            !
!  ---  input/outputs:                                                  !
!            tbot, cmc, t1, stc, smc, sh2o, sneqv, ch, cm,              !
!  ---  outputs:                                                        !
!            nroot, shdfac, snowh, albedo, eta, sheat, ec,              !
!            edir, et, ett, esnow, drip, dew, beta, etp, ssoil,         !
!            flx1, flx2, flx3, runoff1, runoff2, runoff3,               !
!            snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq,        !
!            rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax )     !
!                                                                       !
!                                                                       !
!  subprograms called:  redprm, snow_new, csnow, snfrac, alcalc,        !
!            tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.     !
!                                                                       !
!                                                                       !
!  program history log:                                                 !
!    jun  2003  -- k. mitchell et. al -- created version 2.7            !
!         200x  -- sarah lu    modified the code including:             !
!                       added passing argument, couple; replaced soldn  !
!                       and solnet by radflx; call sfcdif if couple=0;  !
!                       apply time filter to stc and tskin; and the     !
!                       way of namelist inport.                         !
!    feb  2004 -- m. ek noah v2.7.1 non-linear weighting of snow vs     !
!                       non-snow covered portions of gridbox            !
!    apr  2009  -- y.-t. hou   added lw surface emissivity effect,      !
!                       streamlined and reformatted the code, and       !
!                       consolidated constents/parameters by using      !
!                       module physcons, and added program documentation!               !
!    sep  2009 -- s. moorthi minor fixes
!                                                                       !
!  ====================  defination of variables  ====================  !
!                                                                       !
!  inputs:                                                       size   !
!     nsoil    - integer, number of soil layers (>=2 but <=nsold)  1    !
!     couple   - integer, =0:uncoupled (land model only)           1    !
!                         =1:coupled with parent atmos model            !
!     icein    - integer, sea-ice flag (=1: sea-ice, =0: land)     1    !
!     ffrozp   - real,                                             1    !
!     dt       - real, time step (<3600 sec)                       1    !
!     zlvl     - real, height abv atmos ground forcing vars (m)    1    !
!     sldpth   - real, thickness of each soil layer (m)          nsoil  !
!     swdn     - real, downward sw radiation flux (w/m**2)         1    !
!     swnet    - real, downward sw net (dn-up) flux (w/m**2)       1    !
!     lwdn     - real, downward lw radiation flux (w/m**2)         1    !
!     sfcems   - real, sfc lw emissivity (fractional)              1    !
!     sfcprs   - real, pressure at height zlvl abv ground(pascals) 1    !
!     sfctmp   - real, air temp at height zlvl abv ground (k)      1    !
!     sfcspd   - real, wind speed at height zlvl abv ground (m/s)  1    !
!     prcp     - real, precip rate (kg m-2 s-1)                    1    !
!     q2       - real, mixing ratio at hght zlvl abv grnd (kg/kg)  1    !
!     q2sat    - real, sat mixing ratio at zlvl abv grnd (kg/kg)   1    !
!     dqsdt2   - real, slope of sat specific humidity curve at     1    !
!                      t=sfctmp (kg kg-1 k-1)                           !
!     th2      - real, air potential temp at zlvl abv grnd (k)     1    !
!     ivegsrc  - integer, sfc veg type data source umd or igbp          !
!     vegtyp   - integer, vegetation type (integer index)          1    !
!     soiltyp  - integer, soil type (integer index)                1    !
!     slopetyp - integer, class of sfc slope (integer index)       1    !
!     shdmin   - real, min areal coverage of green veg (fraction)  1    !
!     alb      - real, bkground snow-free sfc albedo (fraction)    1    !
!     snoalb   - real, max albedo over deep snow     (fraction)    1    !
!                                                                       !
!  input/outputs:                                                       !
!     tbot     - real, bottom soil temp (k)                        1    !
!                      (local yearly-mean sfc air temp)                 !
!     cmc      - real, canopy moisture content (m)                 1    !
!     t1       - real, ground/canopy/snowpack eff skin temp (k)    1    !
!     stc      - real, soil temp (k)                             nsoil  !
!     smc      - real, total soil moisture (vol fraction)        nsoil  !
!     sh2o     - real, unfrozen soil moisture (vol fraction)     nsoil  !
!                      note: frozen part = smc-sh2o                     !
!     sneqv    - real, water-equivalent snow depth (m)             1    !
!                      note: snow density = snwqv/snowh                 !
!     ch       - real, sfc exchange coeff for heat & moisture (m/s)1    !
!                      note: conductance since it's been mult by wind   !
!     cm       - real, sfc exchange coeff for momentum (m/s)       1    !
!                      note: conductance since it's been mult by wind   !
!                                                                       !
!  outputs:                                                             !
!     nroot    - integer, number of root layers                    1    !
!     shdfac   - real, aeral coverage of green veg (fraction)      1    !
!     snowh    - real, snow depth (m)                              1    !
!     albedo   - real, sfc albedo incl snow effect (fraction)      1    !
!     eta      - real, downward latent heat flux (w/m2)            1    !
!     sheat    - real, downward sensible heat flux (w/m2)          1    !
!     ec       - real, canopy water evaporation (w/m2)             1    !
!     edir     - real, direct soil evaporation (w/m2)              1    !
!     et       - real, plant transpiration     (w/m2)            nsoil  !
!     ett      - real, total plant transpiration (w/m2)            1    !
!     esnow    - real, sublimation from snowpack (w/m2)            1    !
!     drip     - real, through-fall of precip and/or dew in excess 1    !
!                      of canopy water-holding capacity (m)             !
!     dew      - real, dewfall (or frostfall for t<273.15) (m)     1    !
!     beta     - real, ratio of actual/potential evap              1    !
!     etp      - real, potential evaporation (w/m2)                1    !
!     ssoil    - real, upward soil heat flux (w/m2)                1    !
!     flx1     - real, precip-snow sfc flux  (w/m2)                1    !
!     flx2     - real, freezing rain latent heat flux (w/m2)       1    !
!     flx3     - real, phase-change heat flux from snowmelt (w/m2) 1    !
!     snomlt   - real, snow melt (m) (water equivalent)            1    !
!     sncovr   - real, fractional snow cover                       1    !
!     runoff1  - real, surface runoff (m/s) not infiltrating sfc   1    !
!     runoff2  - real, sub sfc runoff (m/s) (baseflow)             1    !
!     runoff3  - real, excess of porosity for a given soil layer   1    !
!     rc       - real, canopy resistance (s/m)                     1    !
!     pc       - real, plant coeff (fraction) where pc*etp=transpi 1    !
!     rsmin    - real, minimum canopy resistance (s/m)             1    !
!     xlai     - real, leaf area index  (dimensionless)            1    !
!     rcs      - real, incoming solar rc factor  (dimensionless)   1    !
!     rct      - real, air temp rc factor        (dimensionless)   1    !
!     rcq      - real, atoms vapor press deficit rc factor         1    !
!     rcsoil   - real, soil moisture rc factor   (dimensionless)   1    !
!     soilw    - real, available soil mois in root zone            1    !
!     soilm    - real, total soil column mois (frozen+unfrozen) (m)1    !
!     smcwlt   - real, wilting point (volumetric)                  1    !
!     smcdry   - real, dry soil mois threshold (volumetric)        1    !
!     smcref   - real, soil mois threshold     (volumetric)        1    !
!     smcmax   - real, porosity (sat val of soil mois)             1    !
!                                                                       !
!  ====================    end of description    =====================  !
!
         use machine, only: kind_phys
         use namelist_soilveg
         use rank, only: myrank
         use physcons, only: con_cp, con_rd, con_t0c, con_g, con_pi, &
                             con_cliq, con_csol, con_hvap, con_hfus, &
                             con_sbc
         use param,    only : my_max
         use index,    only : jlistnum
!
         implicit none

!  ---  constant parameters:
!      *** note: some of the constants are different in subprograms and need to
!          be consolidated with the standard def in module physcons at sometime
!          at the present time, those diverse values are kept temperately to
!          provide the same result as the original codes.  -- y.t.h.  may09

         integer, parameter :: nsold = 4           ! max soil layers

!     real (kind=kind_phys), parameter :: gs      = con_g       ! con_g   =9.80665
         real(kind=kind_phys), parameter :: gs1 = 9.8         ! con_g in sfcdif
         real(kind=kind_phys), parameter :: gs2 = 9.81        ! con_g in snowpack, frh2o
         real(kind=kind_phys), parameter :: tfreez = con_t0c     ! con_t0c =275.15
         real(kind=kind_phys), parameter :: lsubc = 2.501e+6    ! con_hvap=2.5000e+6
         real(kind=kind_phys), parameter :: lsubf = 3.335e5     ! con_hfus=3.3358e+5
         real(kind=kind_phys), parameter :: lsubs = 2.83e+6     ! ? in sflx, snopac
         real(kind=kind_phys), parameter :: elcp = 2.4888e+3   ! ? in penman
!     real (kind=kind_phys), parameter :: rd      = con_rd      ! con_rd  =287.05
         real(kind=kind_phys), parameter :: rd1 = 287.04      ! con_rd in sflx, penman, canres
         real(kind=kind_phys), parameter :: cp = con_cp      ! con_cp  =1004.6
         real(kind=kind_phys), parameter :: cp1 = 1004.5      ! con_cp in sflx, canres
         real(kind=kind_phys), parameter :: cp2 = 1004.0      ! con_cp in htr
!     real (kind=kind_phys), parameter :: cph2o   = con_cliq    ! con_cliq=4.1855e+3
         real(kind=kind_phys), parameter :: cph2o1 = 4.218e+3    ! con_cliq in penman, snopac
         real(kind=kind_phys), parameter :: cph2o2 = 4.2e6       ! con_cliq in hrt *unit diff!
         real(kind=kind_phys), parameter :: cpice = con_csol    ! con_csol=2.106e+3
         real(kind=kind_phys), parameter :: cpice1 = 2.106e6     ! con_csol in hrt *unit diff!
!     real (kind=kind_phys), parameter :: sigma   = con_sbc     ! con_sbc=5.6704e-8
         real(kind=kind_phys), parameter :: sigma1 = 5.67e-8     ! con_sbc in penman, nopac, snopac
!  ---  frh2o_gpu
         real(kind=kind_phys), parameter :: ck = 8.0
!     real (kind=kind_phys), parameter :: ck    = 0.0
         real(kind=kind_phys), parameter :: blim = 5.5
         real(kind=kind_phys), parameter :: error = 0.005
!  ---  srt_gpu
         integer, parameter :: cvfrz = 3
!  ---  snksrc_gpu
         real(kind=kind_phys), parameter :: dh2o = 1.0000e3
!  ---  snowpack_gpu
         real(kind=kind_phys), parameter :: c1 = 0.01
         real(kind=kind_phys), parameter :: c2 = 21.0
!  ---  csnow_gpu
         real(kind=kind_phys), parameter :: unit = 0.11631
!  ---  sfcdif_gpu
         integer, parameter :: itrmx = 5
         real(kind=kind_phys), parameter :: wwst = 1.2
         real(kind=kind_phys), parameter :: wwst2 = wwst*wwst
         real(kind=kind_phys), parameter :: vkrm = 0.40
         real(kind=kind_phys), parameter :: excm = 0.001
         real(kind=kind_phys), parameter :: beta_1 = 1.0/270.0
         real(kind=kind_phys), parameter :: btg = beta_1*gs1
         real(kind=kind_phys), parameter :: elfc = vkrm*btg
         real(kind=kind_phys), parameter :: wold = 0.15
         real(kind=kind_phys), parameter :: wnew = 1.0 - wold
         real(kind=kind_phys), parameter :: pihf = 3.14159265/2.0  ! con_pi/2.0

         real(kind=kind_phys), parameter :: epsu2 = 1.e-4
         real(kind=kind_phys), parameter :: epsust = 0.07
         real(kind=kind_phys), parameter :: ztmin = -5.0
         real(kind=kind_phys), parameter :: ztmax = 1.0
         real(kind=kind_phys), parameter :: hpbl = 1000.0
         real(kind=kind_phys), parameter :: sqvisc = 258.2

         real(kind=kind_phys), parameter :: ric = 0.183
         real(kind=kind_phys), parameter :: rric = 1.0/ric
         real(kind=kind_phys), parameter :: fhneu = 0.8
         real(kind=kind_phys), parameter :: rfc = 0.191
         real(kind=kind_phys), parameter :: rfac = ric/(fhneu*rfc*rfc)
!  ---  shflx_gpu
         real(kind=kind_phys), parameter :: ctfil1 = 0.5
         real(kind=kind_phys), parameter :: ctfil2 = 1.0 - ctfil1
!  ---  snopac_gpu
         real, parameter :: esdmin = 1.e-6


!  ---  inputs:
         integer, intent(in) :: nsoil, ivegsrc, nx, async_id
         integer, dimension(my_max), intent(in) :: myim
         logical, dimension(nx, my_max), intent(in) :: flag_iter, flag
         integer, dimension(nx, my_max), intent(in) :: couple, icein, &
            vegtyp, soiltyp, slopetyp

         real(kind=kind_phys), dimension(nx, my_max), intent(in) :: ffrozp, &
            zlvl, lwdn, swdn, swnet, sfcprs, sfctmp, sfcems, &
            sfcspd, prcp, q2, q2sat, dqsdt2, th2, shdmin, alb, snoalb
         real(kind=kind_phys), intent(in) :: dt
         real(kind=kind_phys), dimension(nsoil), intent(in) :: sldpth

!  ---  input/outputs:
         real(kind=kind_phys), dimension(nx, my_max), intent(inout) :: tbot, &
            cmc, t1, sneqv, ch, cm, z0, shdfac, snowh
         real(kind=kind_phys), dimension(nx, nsoil, my_max), intent(inout) :: &
            stc, smc, sh2o

!  ---  outputs:
         integer, dimension(nx, my_max), intent(out) :: nroot

         real(kind=kind_phys), dimension(nx, my_max), intent(out) :: albedo, &
            eta, sheat, ec, edir, ett, esnow, drip, dew, &
            beta, etp, ssoil, flx1, flx2, flx3, snomlt, sncovr, &
            runoff1, runoff2, runoff3, rc, pc, rsmin, xlai, rcs, &
            rct, rcq, rcsoil, soilw, soilm, smcwlt, smcdry, smcref, &
            smcmax
         real(kind=kind_phys), dimension(nx, nsoil, my_max), intent(out) :: et

!  ---  dummys:
            real(kind=kind_phys), dimension(7) :: gx
            real(kind=kind_phys), dimension(nsold) :: ciin, dmax, &
               ai, bi, ci, part
               
            real(kind=kind_phys), dimension(nsoil) :: rhsttin, &
               rhstsin, stsoil, et1, rtdis, zsoil, rhstt, sice, sh2oa, &
               sh2ofg, rhsts, stcf

!  ---  locals:
!     real (kind=kind_phys) ::  df1h,
         real(kind=kind_phys) ::  bexp, cfactr, cmcmax, csoil, czil, &
            df1, df1a, dksat, dwsat, dsoil, dtot, frcsno, &
            frcsoi, epsca, fdown, f1, fxexp, frzx, hs, kdt, prcp1, &
            psisat, quartz, rch, refkdt, rr, rgl, rsmax, sndens, &
            sncond, sbeta, sn_new, slope, snup, salp, soilwm, soilww, &
            t1v, t24, t2v, th2v, topt, tsnow, zbot
         real(kind=kind_phys) :: gammd, thkdry, ake, thkice, thko, &
            thkqtz, thksat, thks, thkw, satratio, xu, xunfroz ! tdfcnd_gpu
         real(kind=kind_phys) :: z0s ! snowz0_gpu
         real(kind=kind_phys) :: c ! csnow_gpu
         real(kind=kind_phys) :: rsnow, z0n ! snfrac_gpu
         real(kind=kind_phys) :: frzfact, frzk, refdk ! redprm_gpu
         real(kind=kind_phys) :: dsnew, snowhc, hnewc, newsnc, tempc !snow_new_gpu
         real(kind=kind_phys) :: a, delta, fnet, rad, rho ! penman_gpu
         real(kind=kind_phys) :: delta_1, ff, gx_1, rr_1 ! canres_gpu
         real(kind=kind_phys) :: zilfc, zu, zt, rdz, cxch, dthv, du2, &
            btgh, wstar2, ustar, zslu, zslt, rlogu, rlogt, rlmo, &
            zetalt, zetalu, zetau, zetat, xlu4, xlt4, xu4, xt4, &
            xlu, xlt, xu_1, xt, psmz, simm, pshz, simh, ustark, &
            rlmn, rlma ! sfcdif_gpu
         real(kind=kind_phys) :: df2, eta1, etp1, prcp1_1, yy, yynum, &
                                 zz1, ec1, edir1, ett1 ! nopac_gpu
         real(kind=kind_phys) :: cmc2ms ! evapo_gpu
         real(kind=kind_phys) :: fx, sratio ! devap_gpu

         real(kind=kind_phys) :: denom_3, etp1a, rtx, sgx ! transp_gpu
         real(kind=kind_phys) :: dummy, excess, pcpdrp, rhsct, trhsct ! smflx_gpu
         real(kind=kind_phys) :: acrt, dd, ddt, ddz_1, ddz2, denom_4, denom2, &
            dice, dsmdz, dsmdz2, dt1, fcr, infmax, mxsmc, mxsmc2, px, &
            numer, pddum, sicemax, slopx, smcav, sstt, sum, val, wcnd, &
            wcnd2, wdf, wdf2 ! srt_gpu
         real(kind=kind_phys) :: expon, factr1, factr2, vkwgt !wdfcnd_gpu
         real(kind=kind_phys) :: ddz, stot, wplus ! sstep_gpu

         real(kind=kind_phys) :: oldt1 ! shflx_gpu
         real(kind=kind_phys) :: ddz_2, ddz2_1, denom_2, df1n, df1k, dtsdz_1, &
            dtsdz2_1, hcpct_1, qtot, ssoil_2, sice_1, tavg, tbk, tbk1, &
            tsnsr, tsurf, csoil_loc ! hrt_gpu
         real(kind=kind_phys) :: dz_1, free, xh2o ! snksrc_gpu
         real(kind=kind_phys) :: bx, denom_1, df, dswl, fk, swl, swlk ! frh2o_gpu

         real(kind=kind_phys) :: dz, dzh, x0, xdn, xup ! tmpavg_gpu
         real(kind=kind_phys) :: zb, zup ! tbnd_gpu
         real(kind=kind_phys) :: ddz_3, ddz2_2, denom_5, dtsdz, dtsdz2, &
                                 hcpct, ssoil_1, zbot_1 ! hrtice_gpu

         real(kind=kind_phys):: denom, dsoil_1, dtot_1, ssoil1, &
            snoexp, ex, t11, t12, t12a, t12b, seh, t14, &
            etns, etns1, esnow1, esnow2, etanrg ! snopac_gpu
         real(kind=kind_phys) :: bfac, dsx, dthr, dw, pexp, &
                                 tavgc, tsnowc, tsoilc, esdc, esdcx ! snowpack_gpu




         integer :: ipol, j ! snowpack_gpu
         integer :: ilech, itr     ! sfcdif_gpu        

         logical :: frzgra, snowng
         logical :: itavg ! hrt_gpu
         integer :: ice, k, kz, jj, i, ii


         integer :: nlog, kcount ! frh2o_gpu

         integer :: ialp1, iohinf, j1, ks ! srt_gpu
         integer :: kk11 ! sstep_gpu
         integer :: kk ! rosr12_gpu

         integer :: i1 ! evapo_gpu
         data snoexp/2.0/    !!! <----- for noah v2.7.1

!
!===> ...  begin here
!

         !$acc parallel loop gang vector collapse(2) private(bexp, cfactr, cmcmax, &
         !$acc&         csoil, czil, df1, df1a, dksat, dwsat, dsoil, dtot, frcsno, &
         !$acc&         frcsoi, epsca, fdown, f1, fxexp, frzx, hs, kdt, prcp1, &
         !$acc&         psisat, quartz, rch, refkdt, rr, rgl, rsmax, sndens, &
         !$acc&         sncond, sbeta, sn_new, slope, snup, salp, soilwm, soilww, &
         !$acc&         t1v, t24, t2v, th2v, topt, tsnow, zbot, gammd, thkdry, ake, &
         !$acc&         thkice, thko, thkqtz, thksat, thks, thkw, satratio, xu, &
         !$acc&         xunfroz, z0s, c, rsnow, z0n, frzfact, frzk, refdk, dsnew, &
         !$acc&         snowhc, hnewc, newsnc, tempc, a, delta, fnet, rad, rho, &
         !$acc&         delta_1, ff, gx_1, rr_1, zilfc, zu, zt, rdz, cxch, dthv, &
         !$acc&         du2, btgh, wstar2, ustar, zslu, zslt, rlogu, rlogt, rlmo, &
         !$acc&         zetalt, zetalu, zetau, zetat, xlu4, xlt4, xu4, xt4, xlu, &
         !$acc&         xlt, xu_1, xt, psmz, simm, pshz, simh, ustark, rlmn, rlma, &
         !$acc&         df2, eta1, etp1, prcp1_1, yy, yynum, zz1, ec1, edir1, ett1, &
         !$acc&         cmc2ms, fx, sratio, denom_3, etp1a, rtx, sgx, dummy, excess, &
         !$acc&         pcpdrp, rhsct, trhsct, acrt, dd, ddt, ddz_1, ddz2, denom_4, &
         !$acc&         denom2, dice, dsmdz, dsmdz2, dt1, fcr, infmax, mxsmc, mxsmc2, &
         !$acc&         px, numer, pddum, sicemax, slopx, smcav, sstt, sum, val, wcnd, &
         !$acc&         wcnd2, wdf, wdf2, expon, factr1, factr2, vkwgt, ddz, stot, &
         !$acc&         wplus, oldt1, ddz_2, ddz2_1, denom_2, df1n, df1k, dtsdz_1, &
         !$acc&         dtsdz2_1, hcpct_1, qtot, ssoil_2, sice_1, tavg, tbk, tbk1, &
         !$acc&         tsnsr, tsurf, csoil_loc, dz_1, free, xh2o, bx, denom_1, df, &
         !$acc&         dswl, fk, swl, swlk, dz, dzh, x0, xdn, xup, zb, zup, ddz_3, &
         !$acc&         ddz2_2, denom_5, dtsdz, dtsdz2, hcpct, ssoil_1, zbot_1, denom, &
         !$acc&         dsoil_1, dtot_1, ssoil1, snoexp, ex, t11, t12, t12a, t12b, &
         !$acc&         seh, t14, etns, etns1, esnow1, esnow2, etanrg, bfac, dsx, &
         !$acc&         dthr, dw, pexp, tavgc, tsnowc, tsoilc, esdc, esdcx, ipol, &
         !$acc&         j, ilech, itr, frzgra, snowng, itavg, ice, k, kz, nlog, &
         !$acc&         kcount, ialp1, iohinf, j1, ks, kk11, kk, i1, i, jj, gx, ciin, &
         !$acc&         dmax, ai, bi, ci, part, rhsttin, rhstsin, stsoil, et1, rtdis, &
         !$acc&         zsoil, rhstt, sice, sh2oa, sh2ofg, rhsts, stcf, ii) &
         !$acc&         vector_length(128) &
         !$acc&         async(async_id)
         do jj = 1, jlistnum
            do i = 1, nx
               if (i .le. myim(jj)) then
                  if (flag(i, jj) .and. flag_iter(i, jj)) then


!  --- ...  initialization
         runoff1(i, jj) = 0.0
         runoff2(i, jj) = 0.0
         runoff3(i, jj) = 0.0
         snomlt(i, jj) = 0.0

!  --- ...  define local variable ice to achieve:
!             sea-ice case,          ice =  1
!             non-glacial land,      ice =  0
!             glacial-ice land,      ice = -1
!             if vegtype=15 (glacial-ice), re-set ice flag = -1 (glacial-ice)
!    note - for open-sea, sflx should *not* have been called. set green
!           vegetation fraction (shdfac(i, jj)) = 0.

         ice = icein(i, jj)
         !if (myrank .eq. 0) write(*,*) 'df1 in sflx:', loc(df1)
         if (ivegsrc == 0) then
            if (vegtyp(i, jj) == 13) then
               ice = -1
               shdfac(i, jj) = 0.0
            end if
         end if

         if (ivegsrc .ge. 1) then
            if (vegtyp(i, jj) == 15) then
               ice = -1
               shdfac(i, jj) = 0.0
            end if
         end if

         if (ice == 1) then
            shdfac(i, jj) = 0.0

!  --- ...  set green vegetation fraction (shdfac(i, jj)) = 0.
!           set sea-ice layers of equal thickness and sum to 3 meters
            !$acc loop seq
            do kz = 1, nsoil
               zsoil(kz) = -3.0*float(kz)/float(nsoil)
            end do

         else

!  --- ...  calculate depth (negative) below ground from top skin sfc to
!           bottom of each soil layer.
!    note - sign of zsoil is negative (denoting below ground)
            zsoil(1) = -sldpth(1)
            !$acc loop seq
            do kz = 2, nsoil
               zsoil(kz) = -sldpth(kz) + zsoil(kz - 1)
            end do

         end if   ! end if_ice_block

!  --- ...  next is crucial call to set the land-surface parameters,
!           including soil-type and veg-type dependent parameters.
!           set shdfac(i, jj)=0.0 for bare soil surfaces

!         call redprm_gpu &                                                               !%f
!!  ---  inputs:                                                                          !%f
!    &     ( nsoil, vegtyp, soiltyp, slopetyp, sldpth, zsoil, &                           !%f
!!  ---  outputs:                                                                         !%f
!    &       cfactr, cmcmax, rsmin, rsmax, topt, refkdt, kdt,              &              !%f
!    &       sbeta, shdfac, rgl, hs, zbot, frzx, psisat, slope,            &              !%f
!    &       snup, salp, bexp, dksat, dwsat, smcmax, smcwlt,               &              !%f
!    &       smcref, smcdry, f1, quartz, fxexp, rtdis, nroot,              &              !%f
!    &       z0, czil, xlai, csoil                                         &              !%f
!    )                                                                                    !%f
            if (soiltyp(i, jj) > defined_soil) then                                       !%f
               write (*, *) 'warning: too many soil types,soiltyp(i, jj)=', &             !%f
                  soiltyp(i, jj), 'defined_soil=', defined_soil                           !%f
               stop 333                                                                   !%f
            end if                                                                        !%f
            if (vegtyp(i, jj) > defined_veg) then                                         !%f
               write (*, *) 'warning: too many veg types'                                 !%f
               stop 333                                                                   !%f
            end if                                                                        !%f
            if (slopetyp(i, jj) > defined_slope) then                                     !%f
               write (*, *) 'warning: too many slope types'                               !%f
               stop 333                                                                   !%f
            end if                                                                        !%f
                                                                                          !%f
!  --- ...  set-up universal parameters (not dependent on soiltyp(i, jj),                 !%f
!           vegtyp(i, jj) or slopetyp(i, jj))                                             !%f
                                                                                          !%f
            zbot = zbot_data                                                              !%f
            salp = salp_data                                                              !%f
            cfactr = cfactr_data                                                          !%f
            cmcmax = cmcmax_data                                                          !%f
            sbeta = sbeta_data                                                            !%f
            rsmax = rsmax_data                                                            !%f
            topt = topt_data                                                              !%f
            refdk = refdk_data                                                            !%f
            frzk = frzk_data                                                              !%f
            fxexp = fxexp_data                                                            !%f
            refkdt = refkdt_data                                                          !%f
            czil = czil_data                                                              !%f
            csoil = csoil_data                                                            !%f
                                                                                          !%f
!  --- ...  set-up soil parameters                                                        !%f
                                                                                          !%f
            bexp = bb(soiltyp(i, jj))                                                     !%f
            dksat = satdk(soiltyp(i, jj))                                                 !%f
            dwsat = satdw(soiltyp(i, jj))                                                 !%f
            f1 = f11(soiltyp(i, jj))                                                      !%f
            kdt = refkdt*dksat/refdk                                                      !%f
                                                                                          !%f
            psisat = satpsi(soiltyp(i, jj))                                               !%f
            quartz = qtz(soiltyp(i, jj))                                                  !%f
            smcdry(i, jj) = drysmc(soiltyp(i, jj))                                        !%f
            smcmax(i, jj) = maxsmc(soiltyp(i, jj))                                        !%f
            smcref(i, jj) = refsmc(soiltyp(i, jj))                                        !%f
            smcwlt(i, jj) = wltsmc(soiltyp(i, jj))                                        !%f
                                                                                          !%f
            frzfact = (smcmax(i, jj)/smcref(i, jj))*(0.412/0.468)                         !%f
                                                                                          !%f
!  --- ...  to adjust frzk parameter to actual soil type: frzk * frzfact                  !%f
            frzx = frzk*frzfact                                                           !%f
                                                                                          !%f
!  --- ...  set-up vegetation parameters                                                  !%f
            nroot(i, jj) = nroot_data(vegtyp(i, jj))                                      !%f
            snup = snupx(vegtyp(i, jj))                                                   !%f
            rsmin(i, jj) = rsmtbl(vegtyp(i, jj))                                          !%f
            rgl = rgltbl(vegtyp(i, jj))                                                   !%f
            hs = hstbl(vegtyp(i, jj))                                                     !%f
! roughness lengthe is defined in sfcsub                                                  !%f
!     z0(i, jj)  = z0_data(vegtyp(i, jj))                                                 !%f
            xlai(i, jj) = lai_data(vegtyp(i, jj))                                         !%f
!     sfcems(i, jj)= ems1_data(vegtyp(i, jj))      !for summer season                     !%f
!     sfcems(i, jj)= ems2_data(vegtyp(i, jj))      !for winter season                     !%f
            if (vegtyp(i, jj) == bare) then                                               !%f
               shdfac(i, jj) = 0.0                                                        !%f
            end if                                                                        !%f
                                                                                          !%f
            if (nroot(i, jj) > nsoil) then                                                !%f
               write (*, *) 'warning: too many root layers'                               !%f
               stop 333                                                                   !%f
            end if                                                                        !%f
                                                                                          !%f
!  --- ...  calculate root distribution.  present version assumes uniform                 !%f
!           distribution based on soil layer depths.                                      !%f
            !$acc loop seq                                                                !%f
            do i1 = 1, nroot(i, jj)                                                       !%f
               rtdis(i1) = -sldpth(i1)/zsoil(nroot(i, jj))                                !%f
            end do                                                                        !%f
                                                                                          !%f
!  --- ...  set-up slope parameter                                                        !%f
            slope = slope_data(slopetyp(i, jj))                                           !%f
            !end call redprm_gpu                                                          !%f

         if (ivegsrc .ge. 1) then
!only igbp type has urban
!urban
            if (vegtyp(i, jj) == 13) then
               shdfac(i, jj) = 0.05
               rsmin(i, jj) = 400.0
               smcmax(i, jj) = 0.45
               smcref(i, jj) = 0.42
               smcwlt(i, jj) = 0.40
               smcdry(i, jj) = 0.40
            end if
         end if

!  ---  inputs:                                                            !
!          ( nsoil, vegtyp, soiltyp, slopetyp, sldpth, zsoil,              !
!  ---  outputs:                                                           !
!            cfactr, cmcmax, rsmin, rsmax, topt, refkdt, kdt,              !
!            sbeta, shdfac, rgl, hs, zbot, frzx, psisat, slope,            !
!            snup, salp, bexp, dksat, dwsat, smcmax, smcwlt,               !
!            smcref, smcdry, f1, quartz, fxexp, rtdis, nroot,              !
!            z0, czil, xlai, csoil )                                       !

!  --- ...  initialize precipitation logicals.

         snowng = .false.
         frzgra = .false.

!  --- ...  over sea-ice or glacial-ice, if s.w.e. (sneqv(i, jj)) below threshold
!           lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then
!           set at lower bound and store the source increment in subsurface
!           runoff/baseflow (runoff2(i, jj)).
!    note - runoff2(i, jj) is then a negative value (as a flag) over sea-ice or
!           glacial-ice, in order to achieve water balance.
         if (ice == 1) then
            if (sneqv(i, jj) < 0.01) then
               sneqv(i, jj) = 0.01
               snowh(i, jj) = 0.10
!         snowh(i, jj) = sneqv(i, jj) / sndens
            end if
         elseif (ice == -1) then
            if (sneqv(i, jj) < 0.10) then
               
!         sndens = sneqv(i, jj) / snowh(i, jj)
!         runoff2(i, jj) = -(0.10 - sneqv(i, jj)) / dt
               sneqv(i, jj) = 0.10
               snowh(i, jj) = 1.00
!         snowh(i, jj) = sneqv(i, jj) / sndens
            end if

         end if   ! end if_ice_block

!  --- ...  for sea-ice and glacial-ice cases, set smc and sh2o values = 1
!           as a flag for non-soil medium
         if (ice /= 0) then
            !$acc loop seq
            do kz = 1, nsoil
               smc(i, kz, jj) = 1.0
               sh2o(i, kz, jj) = 1.0
            end do
         end if

!  --- ...  if input snowpack is nonzero, then compute snow density "sndens"
!           and snow thermal conductivity "sncond" (note that csnow is a
!           function subroutine)
         if (sneqv(i, jj) .eq. 0.0) then
            
            sndens = 0.0
            snowh(i, jj) = 0.0
            sncond = 1.0
         else
            sndens = sneqv(i, jj)/snowh(i, jj)
            sndens = max(0.0, min(1.0, sndens))   ! added by moorthi

            !call csnow_gpu &                                                             !%c
!  ---  inputs:                                                         !                 !%c
         ! ( sndens,  &                                                                   !%c
!  ---  outputs:                                                        !                 !%c
         !   sncond)                                                   !                  !%c
            c = 0.328*10**(2.25*sndens)                                                   !%c
            sncond = unit*c                                                               !%c
            !end call csnow_gpu                                                           !%c

         end if

!  --- ...  determine if it's precipitating and what kind of precip it is.
!           if it's prcping and the air temp is colder than 0 c, it's snowing!
!           if it's prcping and the air temp is warmer than 0 c, but the grnd
!           temp is colder than 0 c, freezing rain is presumed to be falling.
         if (prcp(i, jj) > 0.0) then
            if (ffrozp(i, jj) > 0.5) then
               snowng = .true.
            else
               if (t1(i, jj) <= tfreez) then
                  frzgra = .true.
               end if
            end if
         end if

!  --- ...  if either prcp(i, jj) flag is set, determine new snowfall (converting
!           prcp(i, jj) rate from kg m-2 s-1 to a liquid equiv snow depth in meters)
!           and add it to the existing snowpack.
!    note - that since all precip is added to snowpack, no precip infiltrates
!           into the soil so that prcp1 is set to zero.
         if (snowng .or. frzgra) then
            sn_new = prcp(i, jj)*dt*0.001
            sneqv(i, jj) = sneqv(i, jj) + sn_new
            prcp1 = 0.0

!  --- ...  update snow density based on new snowfall, using old and new
!           snow.  update snow thermal conductivity

            !call snow_new_gpu &                                                          !%j
!  ---  inputs:                                                         !                 !%j
          !( sfctmp, sn_new,  &                                                           !%j
!  ---  input/outputs:                                                  !                 !%j
          !  snowh, sndens)                                            !                  !%j
!  --- ...  conversion into simulation units                                              !%j
                                                                                          !%j
            snowhc = snowh(i, jj)*100.0                                                   !%j
            newsnc = sn_new*100.0                                                         !%j
            tempc = sfctmp(i, jj) - tfreez                                                !%j
                                                                                          !%j
!  --- ...  calculating new snowfall density depending on temperature                     !%j
!           equation from gottlib l. 'a general runoff model for                          !%j
!           snowcovered and glacierized basin', 6th nordic hydrological                   !%j
!           conference, vemadolen, sweden, 1980, 172-177pp.                               !%j
                                                                                          !%j
            if (tempc <= -15.0) then                                                      !%j
               dsnew = 0.05                                                               !%j
            else                                                                          !%j
               dsnew = 0.05 + 0.0017*(tempc + 15.0)**1.5                                  !%j
            end if                                                                        !%j
                                                                                          !%j
!  --- ...  adjustment of snow density depending on new snowfall                          !%j
            hnewc = newsnc/dsnew                                                          !%j
            sndens = (snowhc*sndens + hnewc*dsnew)/(snowhc + hnewc)                       !%j
            snowhc = snowhc + hnewc                                                       !%j
            snowh(i, jj) = snowhc*0.01                                                    !%j
            !end call snow_new_gpu                                                        !%j

            !call csnow_gpu &                                                             !%c
!  ---  inputs:                                                         !                 !%c
          !( sndens,    &                                                                 !%c
!  ---  outputs:                                                        !                 !%c
          !  sncond)                                                   !                  !%c
            c = 0.328*10**(2.25*sndens)                                                   !%c
            sncond = unit*c                                                               !%c
            !end call csnow_gpu                                                           !%c

         else

!  --- ...  precip is liquid (rain), hence save in the precip variable
!           that later can wholely or partially infiltrate the soil (along
!           with any canopy "drip(i, jj)" added to this later)

            prcp1 = prcp(i, jj)

         end if   ! end if_snowng_block

!  --- ...  determine snowcover fraction and albedo(i, jj) fraction over land.
         if (ice /= 0) then

!  --- ...  snow cover, albedo(i, jj) over sea-ice, glacial-ice

            sncovr(i, jj) = 1.0
            albedo(i, jj) = 0.65
         else

!  --- ...  non-glacial land
!           if snow depth=0, set snowcover fraction=0, albedo(i, jj)=snow free 
!           albedo(i, jj).
            if (sneqv(i, jj) == 0.0) then
               sncovr(i, jj) = 0.0
               albedo(i, jj) = alb(i, jj)
            else

!  --- ...  determine snow fraction cover.
!           determine surface albedo(i, jj) modification due to snowdepth state.

               !call snfrac_gpu &                                                         !%h
!  ---  inputs:                                                         !                 !%h
          !( sneqv(i, jj), snup, salp, snowh(i, jj), &                                    !%h
!  ---  outputs:                                                        !                 !%h
          !  sncovr(i, jj))                                                   !           !%h
            if (sneqv(i, jj) < snup) then                                                 !%h
               rsnow = sneqv(i, jj)/snup                                                  !%h
               sncovr(i, jj) = 1.0 - (exp(-salp*rsnow) - rsnow*exp(-salp))                !%h
            else                                                                          !%h
               sncovr(i, jj) = 1.0                                                        !%h
            end if                                                                        !%h
                                                                                          !%h
            z0n = 0.035                                                                   !%h
            !end call snfrac_gpu                                                          !%h


          !     call alcalc_gpu &                                                         !%a
!  ---  inputs:                                                         !                 !%a
          !( alb, snoalb, shdfac, shdmin, sncovr, tsnow,       &                          !%a
!  ---  outputs:                                                        !                 !%a
          !  albedo(i, jj))                                                               !%a
            albedo(i, jj) = alb(i, jj) + sncovr(i, jj)*(snoalb(i, jj) - alb(i, jj))       !%a
            if (albedo(i, jj) > snoalb(i, jj)) then                                       !%a
               albedo(i, jj) = snoalb(i, jj)                                              !%a
            end if                                                                        !%a
            ! end call alcalc_gpu                                                         !%a

            end if   ! end if_sneqv_block

         end if   ! end if_ice_block

!  --- ...  thermal conductivity for sea-ice case, glacial-ice case
         if (ice /= 0) then
            df1 = 2.2

         else

!  --- ...  next calculate the subsurface heat flux, which first requires
!           calculation of the thermal diffusivity.  treatment of the
!           latter follows that on pages 148-149 from "heat transfer in
!           cold climates", by v. j. lunardini (published in 1981
!           by van nostrand reinhold co.) i1.e. treatment of two contiguous
!           "plane parallel" mediums (namely here the first soil layer
!           and the snowpack layer, if any). this diffusivity treatment
!           behaves well for both zero and nonzero snowpack, including the
!           limit of very thin snowpack.  this treatment also eliminates
!           the need to impose an arbitrary upper bound on subsurface
!           heat flux when the snowpack becomes extremely thin.

!  --- ...   first calculate thermal diffusivity of top soil layer, using
!            both the frozen and liquid soil moisture, following the
!            soil thermal diffusivity function of peters-lidard et al.
!            (1998,jas, vol 55, 1209-1224), which requires the specifying
!            the quartz content of the given soil class (see routine redprm)

            !call tdfcnd_gpu &                                                            !%l
            !   !  ---  inputs: &                                                         !%l
            !   (smc(1), quartz, smcmax, sh2o(1), &                                       !%l
            !    !  ---  outputs: &                                                       !%l
            !    df1)                                                                     !%l
            k = 1                                                                         !%l
            !  --- ...  if the soil has any moisture content compute a partial            !%l
!           sum/product otherwise use a constant value which works well with most         !%l
!           soils                                                                         !%l
!  --- ...  saturation ratio:                                                             !%l
            satratio = smc(i, k, jj)/smcmax(i, jj)                                        !%l
                                                                                          !%l
!  --- ...  parameters  w/(m.k)                                                           !%l
            thkice = 2.2                                                                  !%l
            thkw = 0.57                                                                   !%l
            thko = 2.0                                                                    !%l
!     if (quartz <= 0.2) thko = 3.0                                                       !%l
            thkqtz = 7.7                                                                  !%l
                                                                                          !%l
!  --- ...  solids' conductivity                                                          !%l
                                                                                          !%l
            thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                                !%l
                                                                                          !%l
!  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0. (100% frozen))           !%l
                                                                                          !%l
            xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)                    !%l
                                                                                          !%l
!  --- ...  unfrozen volume for saturation (porosity*xunfroz)                             !%l
                                                                                          !%l
            xu = xunfroz*smcmax(i, jj)                                                    !%l
                                                                                          !%l
!  --- ...  saturated thermal conductivity                                                !%l
                                                                                          !%l
            thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu)*thkw**(xu)     !%l
                                                                                          !%l
!  --- ...  dry density in kg/m3                                                          !%l
                                                                                          !%l
            gammd = (1.0 - smcmax(i, jj))*2700.0                                          !%l
                                                                                          !%l
!  --- ...  dry thermal conductivity in w.m-1.k-1                                         !%l
                                                                                          !%l
            thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                          !%l
            if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then         ! frozen            !%l
               ake = satratio                                                             !%l
                                                                                          !%l
            else                                  ! unfrozen                              !%l
                                                                                          !%l
!  --- ...  range of validity for the kersten number (ake)                                !%l
               if (satratio > 0.1) then                                                   !%l
                                                                                          !%l
!  --- ...  kersten number (using "fine" formula, valid for soils containing              !%l
!           at least 5% of particles with diameter less than 2.e-6 meters.)               !%l
!           (for "coarse" formula, see peters-lidard et al., 1998).                       !%l
                                                                                          !%l
                  ake = log10(satratio) + 1.0                                             !%l
                                                                                          !%l
               else                                                                       !%l
                                                                                          !%l
!  --- ...  use k = kdry                                                                  !%l
                  ake = 0.0                                                               !%l
                                                                                          !%l
               end if   ! end if_satratio_block                                           !%l
                                                                                          !%l
            end if   ! end if_sh2o+0.0005_block                                           !%l
                                                                                          !%l
!  --- ...  thermal conductivity                                                          !%l
            df1 = ake*(thksat - thkdry) + thkdry                                          !%l
            !end call tdfcnd_gpu                                                          !%l

            if (ivegsrc .ge. 1) then
!only igbp type has urban
!urban
               if (vegtyp(i, jj) == 13) then
                  df1 = 3.24
               end if
            end if

!  --- ...  next add subsurface heat flux reduction effect from the
!           overlying green canopy, adapted from section 2.1.2 of
!           peters-lidard et al. (1997, jgr, vol 102(d4))
            df1 = df1*exp(sbeta*shdfac(i, jj))

         end if   ! end if_ice_block

!  --- ...  finally "plane parallel" snowpack effect following
!           v.j. linardini reference cited above. note that dtot is
!           combined depth of snowdepth and thickness of first soil layer
         dsoil = -0.5*zsoil(1)
         if (sneqv(i, jj) == 0.0) then
            ssoil(i, jj) = df1*(t1(i, jj) - stc(i, 1, jj))/dsoil
         else
            dtot = snowh(i, jj) + dsoil
            frcsno = snowh(i, jj)/dtot
            frcsoi = dsoil/dtot

!  --- ...  1. harmonic mean (series flow)

!       df1  = (sncond*df1) / (frcsoi*sncond + frcsno*df1)
!       df1h = (sncond*df1) / (frcsoi*sncond + frcsno*df1)

!  --- ...  2. arithmetic mean (parallel flow)

!       df1  = frcsno*sncond + frcsoi*df1
            df1a = frcsno*sncond + frcsoi*df1

!  --- ...  3. geometric mean (intermediate between harmonic and arithmetic mean)

!       df1 = (sncond**frcsno) * (df1**frcsoi)
!       df1 = df1h*sncovr(i, jj) + df1a*(1.0-sncovr(i, jj))
!       df1 = df1h*sncovr(i, jj) + df1 *(1.0-sncovr(i, jj))
            df1 = df1a*sncovr(i, jj) + df1*(1.0 - sncovr(i, jj))

!  --- ...  calculate subsurface heat flux, ssoil(i, jj), from final thermal
!           diffusivity of surface mediums, df1 above, and skin
!           temperature and top mid-layer soil temperature

            ssoil(i, jj) = df1*(t1(i, jj) - stc(i, 1, jj))/dtot

         end if   ! end if_sneqv_block

!  --- ...  determine surface roughness over snowpack using snow condition
!           from the previous timestep.

!     if (couple(i, jj) == 0) then            ! uncoupled mode
         if (sncovr(i, jj) > 0.0) then

            !call snowz0_gpu &                                                            !%k
!  ---  inputs:                                                         !                 !%k
          !( sncovr(i, jj),   &                                                           !%k
!  ---  input/outputs:                                                  !                 !%k
          !  z0(i, jj))                                                       !           !%k
            z0s = z0(i, jj)                                                               !%k
                                                                                          !%k
            z0(i, jj) = (1.0 - sncovr(i, jj))*z0(i, jj) + sncovr(i, jj)*z0s               !%k
            !end call snowz0_gpu                                                          !%k

         end if
!     endif

!  --- ...  calc virtual temps and virtual potential temps needed by
!           subroutines sfcdif and penman.
         t2v = sfctmp(i, jj)*(1.0 + 0.61*q2(i, jj))

!  --- ...  next call routine sfcdif to calculate the sfc exchange coef (ch(i, jj))
!           for heat and moisture.
!    note - comment out call sfcdif, if sfcdif already called in calling
!           program (such as in coupled atmospheric model).
!         - do not call sfcdif until after above call to redprm, in case
!           alternative values of roughness length (z0(i, jj)) and zilintinkevich
!           coef (czil) are set there via namelist i1/o.
!         - routine sfcdif returns a ch(i, jj) that represents the wind spd times
!           the "original" nondimensional "ch(i, jj)" typical in literature.  hence
!           the ch(i, jj) returned from sfcdif has units of m/s.  the important
!           companion coefficient of ch(i, jj), carried here as "rch", is the 
!           ch(i, jj) from sfcdif times air density and parameter "cp".  "rch" is
!           computed in "call penman". rch rather than ch(i, jj) is the coeff
!           usually invoked later in eqns.
!         - sfcdif also returns the surface exchange coefficient for momentum,
!           cm(i, jj), also known as the surface drage coefficient, but cm(i, jj) is 
!           not used here.

!  --- ...  key required radiation term is the total downward radiation
!           (fdown) = net solar (swnet(i, jj)) + downward longwave (lwdn(i, jj)),
!           for use in penman ep calculation (penman) and other surface
!           energy budget calcuations.  also need downward solar (swdn(i, jj))
!           for canopy resistance routine (canres).
!    note - fdown, swdn(i, jj) are derived differently in the uncoupled and
!           coupled modes.
         if (couple(i, jj) == 0) then                      !......uncoupled mode

!  --- ...  uncoupled mode:
!           compute surface exchange coefficients
            t1v = t1(i, jj)*(1.0 + 0.61*q2(i, jj))
            th2v = th2(i, jj)*(1.0 + 0.61*q2(i, jj))

         !   call sfcdif_gpu &                                                            !%g
!  ---  inputs:                                                         !                 !%g
         ! ( zlvl, z0, t1v, th2v, sfcspd, czil,  &                                        !%g
!  ---  input/outputs:                                                  !                 !%g
         !   cm, ch)                                                   !                  !%g
!  --- ...  this routine sfcdif can handle both over open water (sea, ocean) and          !%g
!           over solid surface (land, sea-ice).                                           !%g
                                                                                          !%g
         ilech = 0                                                                        !%g
                                                                                          !%g
!   --- ...  ztfc: ratio of zoh/zom  less or equal than 1                                 !%g
!            czil: constant c in zilitinkevich, s. s.1995,:note about zt                  !%g
         zilfc = -czil*vkrm*sqvisc                                                        !%g
                                                                                          !%g
         zu = z0(i, jj)                                                                   !%g
                                                                                          !%g
         rdz = 1.0/zlvl(i, jj)                                                            !%g
         cxch = excm*rdz                                                                  !%g
         dthv = th2v - t1v                                                                !%g
         du2 = max(sfcspd(i, jj)*sfcspd(i, jj), epsu2)                                    !%g
                                                                                          !%g
!  --- ...  beljars correction of ustar                                                   !%g
         btgh = btg*hpbl                                                                  !%g
                                                                                          !%g
!  --- ...  if statements to avoid tangent linear problems near zero                      !%g
         if (btgh*ch(i, jj)*dthv /= 0.0) then                                             !%g
            wstar2 = wwst2*abs(btgh*ch(i, jj)*dthv)**(2.0/3.0)                            !%g
         else                                                                             !%g
            wstar2 = 0.0                                                                  !%g
         end if                                                                           !%g
         ustar = max(sqrt(cm(i, jj)*sqrt(du2 + wstar2)), epsust)                          !%g
                                                                                          !%g
!  --- ...  zilitinkevitch approach for zt                                                !%g
                                                                                          !%g
         zt = exp(zilfc*sqrt(ustar*z0(i, jj)))*z0(i, jj)                                  !%g
                                                                                          !%g
         zslu = zlvl(i, jj) + zu                                                          !%g
         zslt = zlvl(i, jj) + zt                                                          !%g
                                                                                          !%g
!     print*,'zslt=',zslt                                                                 !%g
!     print*,'zlvl(i, jj)=',zvll                                                          !%g
!     print*,'zt=',zt                                                                     !%g
                                                                                          !%g
         rlogu = log(zslu/zu)                                                             !%g
         rlogt = log(zslt/zt)                                                             !%g
                                                                                          !%g
         rlmo = elfc*ch(i, jj)*dthv/ustar**3                                              !%g
                                                                                          !%g
!     print*,'rlmo=',rlmo                                                                 !%g
!     print*,'elfc=',elfc                                                                 !%g
!     print*,'ch(i, jj)=',ch(i, jj)                                                       !%g
!     print*,'dthv=',dthv                                                                 !%g
!     print*,'ustar=',ustar                                                               !%g
         !$acc loop seq                                                                   !%g
         do itr = 1, itrmx                                                                !%g
                                                                                          !%g
!  --- ...  1./ monin-obukkhov length-scale                                               !%g
            zetalt = max(zslt*rlmo, ztmin)                                                !%g
            rlmo = zetalt/zslt                                                            !%g
            zetalu = zslu*rlmo                                                            !%g
            zetau = zu*rlmo                                                               !%g
            zetat = zt*rlmo                                                               !%g
            if (ilech == 0) then                                                          !%g
               if (rlmo < 0.0) then                                                       !%g
                  xlu4 = 1.0 - 16.0*zetalu                                                !%g
                  xlt4 = 1.0 - 16.0*zetalt                                                !%g
                  xu4 = 1.0 - 16.0*zetau                                                  !%g
                  xt4 = 1.0 - 16.0*zetat                                                  !%g
                                                                                          !%g
                  xlu = sqrt(sqrt(xlu4))                                                  !%g
                  xlt = sqrt(sqrt(xlt4))                                                  !%g
                  xu_1 = sqrt(sqrt(xu4))                                                  !%g
                  xt = sqrt(sqrt(xt4))                                                    !%g
                                                                                          !%g
                  psmz = -2.0*log((xu_1 + 1.0)*0.5) &                                     !%g
                     - log((xu_1*xu_1 + 1.0)*0.5) + 2.0*atan(xu_1) - pihf                 !%g
                                                                                          !%g
!           print*,'-----------1------------'                                             !%g
!           print*,'psmz=',psmz                                                           !%g
!           print*,'pspmu(zetau)=',pspmu( zetau )                                         !%g
!           print*,'xu_1=',xu_1                                                           !%g
!           print*,'------------------------'                                             !%g
                  simm = -2.0*log((xlu + 1.0)*0.5) &                                      !%g
                     - log((xlu*xlu + 1.0)*0.5) + 2.0*atan(xlu) - pihf &                  !%g
                     - psmz + rlogu                                                       !%g
                  pshz = -2.0*log((xt*xt + 1.0)*0.5)                                      !%g
                  simh = -2.0*log((xlt*xlt + 1.0)*0.5) - pshz + rlogt                     !%g
               else                                                                       !%g
                  zetalu = min(zetalu, ztmax)                                             !%g
                  zetalt = min(zetalt, ztmax)                                             !%g
                  psmz = 5.0*zetau                                                        !%g
                                                                                          !%g
!           print*,'-----------2------------'                                             !%g
!           print*,'psmz=',psmz                                                           !%g
!           print*,'pspms(zetau)=',pspms( zetau )                                         !%g
!           print*,'zetau=',zetau                                                         !%g
!           print*,'------------------------'                                             !%g
                                                                                          !%g
                  simm = 5.0*zetalu - psmz + rlogu                                        !%g
                  pshz = 5.0*zetat                                                        !%g
                  simh = 5.0*zetalt - pshz + rlogt                                        !%g
               end if   ! end if_rlmo_block                                               !%g
                                                                                          !%g
            else                                                                          !%g
                                                                                          !%g
!  --- ...  lech's functions                                                              !%g
               if (rlmo < 0.0) then                                                       !%g
                  psmz = -0.96*log(1.0 - 4.5*zetau)                                       !%g
                                                                                          !%g
!           print*,'-----------3------------'                                             !%g
!           print*,'psmz=',psmz                                                           !%g
!           print*,'pslmu(zetau)=',pslmu( zetau )                                         !%g
!           print*,'zetau=',zetau                                                         !%g
!           print*,'------------------------'                                             !%g
                                                                                          !%g
                  simm = -0.96*log(1.0 - 4.5*zetalu) - psmz + rlogu                       !%g
                  pshz = -0.96*log(1.0 - 4.5*zetat)                                       !%g
                  simh = -0.96*log(1.0 - 4.5*zetalt) - pshz + rlogt                       !%g
               else                                                                       !%g
                  zetalu = min(zetalu, ztmax)                                             !%g
                  zetalt = min(zetalt, ztmax)                                             !%g
                                                                                          !%g
                  psmz = zetau*rric - 2.076*(1.0 - 1.0/(zetau + 1.0))                     !%g
                                                                                          !%g
!           print*,'-----------4------------'                                             !%g
!           print*,'psmz=',psmz                                                           !%g
!           print*,'pslms(zetau)=',pslms( zetau )                                         !%g
!           print*,'zetau=',zetau                                                         !%g
!           print*,'------------------------'                                             !%g
                                                                                          !%g
                  simm = zetalu*rric - 2.076*(1.0 - 1.0/(zetalu + 1.0)) &                 !%g
                         - psmz + rlogu                                                   !%g
                  pshz = zetat*rfac - 2.076*(1.0 - 1.0/(zetat + 1.0))                     !%g
                  simh = zetalt*rfac - 2.076*(1.0 - 1.0/(zetalt + 1.0)) &                 !%g
                         - pshz + rlogt                                                   !%g
               end if   ! end if_rlmo_block                                               !%g
                                                                                          !%g
            end if   ! end if_ilech_block                                                 !%g
                                                                                          !%g
!  --- ...  beljaars correction for ustar                                                 !%g
            ustar = max(sqrt(cm(i, jj)*sqrt(du2 + wstar2)), epsust)                       !%g
                                                                                          !%g
!  --- ...  zilitinkevitch fix for zt                                                     !%g
                                                                                          !%g
            zt = exp(zilfc*sqrt(ustar*z0(i, jj)))*z0(i, jj)                               !%g
                                                                                          !%g
            zslt = zlvl(i, jj) + zt                                                       !%g
            rlogt = log(zslt/zt)                                                          !%g
                                                                                          !%g
            ustark = ustar*vkrm                                                           !%g
            cm(i, jj) = max(ustark/simm, cxch)                                            !%g
            ch(i, jj) = max(ustark/simh, cxch)                                            !%g
                                                                                          !%g
!  --- ...  if statements to avoid tangent linear problems near zero                      !%g
            if (btgh*ch(i, jj)*dthv /= 0.0) then                                          !%g
               wstar2 = wwst2*abs(btgh*ch(i, jj)*dthv)**(2.0/3.0)                         !%g
            else                                                                          !%g
               wstar2 = 0.0                                                               !%g
            end if                                                                        !%g
                                                                                          !%g
            rlmn = elfc*ch(i, jj)*dthv/ustar**3                                           !%g
            rlma = rlmo*wold + rlmn*wnew                                                  !%g
                                                                                          !%g
            rlmo = rlma                                                                   !%g
                                                                                          !%g
         end do   ! end do_itr_loop                                                       !%g
         !end call sfcdir_gpu                                                             !%g

!     swnet(i, jj) = net solar radiation into the ground (w/m2; dn-up) from input
!     fdown  = net solar + downward lw flux at sfc (w/m2)
            fdown = swnet(i, jj) + lwdn(i, jj)

         else                                       !......coupled mode

!  --- ...  coupled mode (couple(i, jj) .ne. 0):
!           surface exchange coefficients computed externally and passed in,
!           hence subroutine sfcdif not called.

!     swnet(i, jj) = net solar radiation into the ground (w/m2; dn-up) from input
!     fdown  = net solar + downward lw flux at sfc (w/m2)
            fdown = swnet(i, jj) + lwdn(i, jj)
         end if   ! end if_couple_block

!  --- ...  call penman subroutine to calculate potential evaporation (etp(i, jj)),
!           and other partial products and sums save in common/rite for later
!           calculations.

         !call penman_gpu &                                                               !%e
!  ---  inputs:                                                         !                 !%e
         ! ( sfctmp, sfcprs, sfcems, ch, t2v, th2, prcp, fdown,  &                        !%e
         !   ssoil, q2, q2sat, dqsdt2, snowng, frzgra,           &                        !%e
!  ---  outputs:                                                        !                 !%e
         !   t24, etp, rch, epsca, rr, flx2 &                                             !%e
         !  )                           !                                                 !%e
         flx2(i, jj) = 0.0                                                                !%e
                                                                                          !%e
!  --- ...  prepare partial quantities for penman equation.                               !%e
         delta = elcp*dqsdt2(i, jj)                                                       !%e
         t24 = sfctmp(i, jj)*sfctmp(i, jj)*sfctmp(i, jj)*sfctmp(i, jj)                    !%e
         rr = t24*6.48e-8/(sfcprs(i, jj)*ch(i, jj)) + 1.0                                 !%e
         rho = sfcprs(i, jj)/(rd1*t2v)                                                    !%e
         rch = rho*cp*ch(i, jj)                                                           !%e
                                                                                          !%e
!  --- ...  adjust the partial sums / products with the latent heat                       !%e
!           effects caused by falling precipitation.                                      !%e
         if (.not. snowng) then                                                           !%e
            if (prcp(i, jj) > 0.0) then                                                   !%e
               rr = rr + cph2o1*prcp(i, jj)/rch                                           !%e
            end if                                                                        !%e
         else                                                                             !%e
               rr = rr + cpice*prcp(i, jj)/rch                                            !%e
         end if                                                                           !%e
                                                                                          !%e
         fnet = fdown - sfcems(i, jj)*sigma1*t24 - ssoil(i, jj)                           !%e
                                                                                          !%e
!  --- ...  include the latent heat effects of frzng rain converting to ice               !%e
!           on impact in the calculation of flx2(i, jj) and fnet.                         !%e
         if (frzgra) then                                                                 !%e
            flx2(i, jj) = -lsubf*prcp(i, jj)                                              !%e
            fnet = fnet - flx2(i, jj)                                                     !%e
         end if                                                                           !%e
                                                                                          !%e
!  --- ...  finish penman equation calculations.                                          !%e
         rad = fnet/rch + th2(i, jj) - sfctmp(i, jj)                                      !%e
         a = elcp*(q2sat(i, jj) - q2(i, jj))                                              !%e
         epsca = (a*rr + rad*delta)/(delta + rr)                                          !%e
         etp(i, jj) = epsca*rch/lsubc                                                     !%e
         !end call penman_gpu                                                             !%e


!  --- ...  call canres to calculate the canopy resistance and convert it
!           into pc(i, jj) if nonzero greenness fraction
         if (shdfac(i, jj) > 0.) then

!  --- ...  frozen ground extension: total soil water "smc" was replaced
!           by unfrozen soil water "sh2o" in call to canres below

          !  call canres_gpu &                                                            !%b
!  ---  inputs:                                                         !                 !%b
          !( nsoil, nroot, swdn, ch, q2, q2sat, dqsdt2, sfctmp,   &                       !%b
          !  sfcprs, sfcems, sh2o, smcwlt, smcref, zsoil, rsmin,  &                       !%b
          !  rsmax, topt, rgl, hs, xlai,                          &                       !%b
!  ---  outputs:                                                        !                 !%b
          !  rc, pc, rcs, rct, rcq, rcsoil,                       &                       !%b
!  ---  dummys:                                                                           !%b
          !  part &                                                                       !%b
          !)                            !                                                 !%b
!  --- ...  initialize canopy resistance multiplier terms.                                !%b
                                                                                          !%b
         rcs(i, jj) = 0.0                                                                 !%b
         rct(i, jj) = 0.0                                                                 !%b
         rcq(i, jj) = 0.0                                                                 !%b
         rcsoil(i, jj) = 0.0                                                              !%b
         rc(i, jj) = 0.0                                                                  !%b
                                                                                          !%b
!  --- ...  contribution due to incoming solar radiation                                  !%b
         ff = 0.55*2.0*swdn(i, jj)/(rgl*xlai(i, jj))                                      !%b
         rcs(i, jj) = (ff + rsmin(i, jj)/rsmax)/(1.0 + ff)                                !%b
         rcs(i, jj) = max(rcs(i, jj), 0.0001)                                             !%b
                                                                                          !%b
!  --- ...  contribution due to air temperature at first model level above ground         !%b
!           rct(i, jj) expression from noilhan and planton (1989, mwr).                   !%b
         rct(i, jj) = 1.0 - 0.0016*(topt - sfctmp(i, jj))**2.0                            !%b
         rct(i, jj) = max(rct(i, jj), 0.0001)                                             !%b
                                                                                          !%b
!  --- ...  contribution due to vapor pressure deficit at first model level.              !%b
!           rcq(i, jj) expression from ssib                                               !%b
         rcq(i, jj) = 1.0/(1.0 + hs*(q2sat(i, jj) - q2(i, jj)))                           !%b
         rcq(i, jj) = max(rcq(i, jj), 0.01)                                               !%b
                                                                                          !%b
!  --- ...  contribution due to soil moisture availability.                               !%b
!           determine contribution from each soil layer, then add them up.                !%b
         gx_1 = (sh2o(i, 1, jj) - smcwlt(i, jj))/(smcref(i, jj) - smcwlt(i, jj))          !%b
         gx_1 = max(0.0, min(1.0, gx_1))                                                  !%b
                                                                                          !%b
!  --- ...  use soil depth as weighting factor                                            !%b
         part(1) = (zsoil(1)/zsoil(nroot(i, jj)))*gx_1                                    !%b
                                                                                          !%b
!  --- ...  use root distribution as weighting factor                                     !%b
!     part(1) = rtdis(1) * gx_1                                                           !%b
         !$acc loop seq                                                                   !%b
         do k = 2, nroot(i, jj)                                                           !%b
            gx_1 = (sh2o(i, k, jj) - smcwlt(i, jj))/(smcref(i, jj) - smcwlt(i, jj))       !%b
            gx_1 = max(0.0, min(1.0, gx_1))                                               !%b
                                                                                          !%b
!  --- ...  use soil depth as weighting factor                                            !%b
            part(k) = ((zsoil(k) - zsoil(k - 1))/zsoil(nroot(i, jj)))*gx_1                !%b
                                                                                          !%b
!  --- ...  use root distribution as weighting factor                                     !%b
!       part(k) = rtdis(k) * gx_1                                                         !%b
                                                                                          !%b
         end do                                                                           !%b
         !$acc loop seq                                                                   !%b
         do k = 1, nroot(i, jj)                                                           !%b
            rcsoil(i, jj) = rcsoil(i, jj) + part(k)                                       !%b
         end do                                                                           !%b
         rcsoil(i, jj) = max(rcsoil(i, jj), 0.0001)                                       !%b
                                                                                          !%b
!  --- ...  determine canopy resistance due to all factors.  convert canopy               !%b
!           resistance (rc) to plant coefficient (pc) to be used with                     !%b
!           potential evap in determining actual evap.  pc is determined by:              !%b
!           pc * linerized penman potential evap = penman-monteith actual                 !%b
!           evaporation (containing rc(i, jj) term).                                      !%b
                                                                                          !%b
         rc(i, jj) = rsmin(i, jj)/ &                                                      !%b
                     (xlai(i, jj)*rcs(i, jj)*rct(i, jj)*rcq(i, jj)*rcsoil(i, jj))         !%b
         rr_1 = (4.0*sfcems(i, jj)*sigma1*rd1/cp1)*(sfctmp(i, jj)**4.0)/ &                !%b
                (sfcprs(i, jj)*ch(i, jj)) + 1.0                                           !%b
         delta_1 = (lsubc/cp1)*dqsdt2(i, jj)                                              !%b
         pc(i, jj) = (rr_1 + delta_1)/(rr_1*(1.0 + rc(i, jj)*ch(i, jj)) + delta_1)        !%b
         !end call canres_gpu                                                             !%b

         end if

!  --- ...  now decide major pathway branch to take depending on whether
!           snowpack exists or not:
         esnow(i, jj) = 0.0
         if (sneqv(i, jj) .eq. 0.0) then

         !   call nopac_gpu &                                                             !%d
!  ---  inputs:                                                         !                 !%d
         ! ( nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref,           &                 !%d
         !   smcdry, cmcmax, dt, shdfac, sbeta, sfctmp, sfcems,         &                 !%d
         !   t24, th2, fdown, epsca, bexp, pc, rch, rr, cfactr,         &                 !%d
         !   slope, kdt, frzx, psisat, zsoil, dksat, dwsat,             &                 !%d
         !   zbot, ice, rtdis, quartz, fxexp, csoil, ivegsrc, vegtyp,   &                 !%d
!  ---  input/outputs:                                                  !                 !%d
         !   cmc, t1, stc, sh2o, tbot, df1,                             &                 !%d
!  ---  outputs:                                                        !                 !%d
         !   eta, smc, ssoil, runoff1, runoff2, runoff3, edir,          &                 !%d
         !   ec, et, ett, beta, drip, dew, flx1, flx3,                  &                 !%d
!  ---  dummys:                                                                           !%d
         !   gx, ciin, rhsttin, dmax, rhstsin, rhstt, sice, sh2oa,      &                 !%d
         !   sh2ofg, ai, bi, ci, rhsts, stcf, stsoil, et1 &                               !%d
         !   )                 !                                                          !%d
!  --- ...  convert etp(i, jj) from kg m-2 s-1 to ms-1 and initialize dew(i, jj).         !%d
         !if (myrank .eq. 0) write(*,*) 'df1 in nopac:', loc(df1)                         !%d
                                                                                          !%d
            prcp1_1 = prcp(i, jj)*0.001                                                   !%d
            etp1 = etp(i, jj)*0.001                                                       !%d
            dew(i, jj) = 0.0                                                              !%d
            edir(i, jj) = 0.0                                                             !%d
            edir1 = 0.0                                                                   !%d
            ec(i, jj) = 0.0                                                               !%d
            ec1 = 0.0                                                                     !%d
            !$acc loop seq                                                                !%d
            do k = 1, nsoil                                                               !%d
               et(i, k, jj) = 0.0                                                         !%d
               et1(k) = 0.0                                                               !%d
            end do                                                                        !%d
                                                                                          !%d
            ett(i, jj) = 0.0                                                              !%d
            ett1 = 0.0                                                                    !%d
                                                                                          !%d
            if (etp(i, jj) > 0.0) then                                                    !%d
                                                                                          !%d
!  --- ...  convert prcp(i, jj) from 'kg m-2 s-1' to 'm s-1'.                             !%d
                                                                                          !%d
               !call evapo_gpu &                                                          !%dm
               !   !  ---  inputs: &                                                      !%dm
               !   (nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil, &                         !%dm
               !    sh2o, smcmax, smcwlt, smcref, smcdry, pc, &                           !%dm
               !    shdfac, cfactr, rtdis, fxexp, &                                       !%dm
               !    !  ---  outputs: &                                                    !%dm
               !    eta1, edir1, ec1, et1, ett1, &                                        !%dm
               !    !  ---  dummys: &                                                     !%dm
               !    gx &                                                                  !%dm
               !    )                                                                     !%dm
   !  --- ...  executable code begins here if the potential evapotranspiration            !%dm
   !           is greater than zero.                                                      !%dm
               edir1 = 0.0                                                                !%dm
               ec1 = 0.0                                                                  !%dm
               !$acc loop seq                                                             !%dm
               do k = 1, nsoil                                                            !%dm
                  et1(k) = 0.0                                                            !%dm
               end do                                                                     !%dm
               ett1 = 0.0                                                                 !%dm
               if (etp1 > 0.0) then                                                       !%dm
                                                                                          !%dm
   !  --- ...  retrieve direct evaporation from soil surface.  call this function         !%dm
   !           only if veg cover not complete.                                            !%dm
   !           frozen ground version:  sh2o states replace smc states.                    !%dm
                  if (shdfac(i, jj) < 1.0) then                                           !%dm
                                                                                          !%dm
                     !call devap_gpu &                                                    !%dmq
                     !   !  ---  inputs: &                                                !%dmq
                     !   (etp1, sh2o(1), shdfac, smcmax, smcdry, fxexp, &                 !%dmq
                     !    !  ---  outputs: &                                              !%dmq
                     !    edir1 &                                                         !%dmq
                     !    )                                                               !%dmq
         !  --- ...  direct evap a function of relative soil moisture availability,       !%dmq
         !           linear when fxexp=1.                                                 !%dmq
         !           fx > 1 represents demand control                                     !%dmq
         !           fx < 1 represents flux control                                       !%dmq
                     sratio = (sh2o(i, 1, jj) - smcdry(i, jj))/ &                         !%dmq
                              (smcmax(i, jj) - smcdry(i, jj))                             !%dmq
                                                                                          !%dmq
                     if (sratio > 0.0) then                                               !%dmq
                        fx = sratio**fxexp                                                !%dmq
                        fx = max(min(fx, 1.0), 0.0)                                       !%dmq
                     else                                                                 !%dmq
                        fx = 0.0                                                          !%dmq
                     end if                                                               !%dmq
                                                                                          !%dmq
         !  --- ...  allow for the direct-evap-reducing effect of shade                   !%dmq
                     edir1 = fx*(1.0 - shdfac(i, jj))*etp1                                !%dmq
                     !end call devap_gpu                                                  !%dmq
                                                                                          !%dm
                  end if                                                                  !%dm
                                                                                          !%dm
   !  --- ...  initialize plant total transpiration, retrieve plant transpiration,        !%dm
   !           and accumulate it for all soil layers.                                     !%dm
                  if (shdfac(i, jj) > 0.0) then                                           !%dm
                                                                                          !%dm
                     !call transp_gpu &                                                   !%dmB
                     !   !  ---  inputs: &                                                !%dmB
                     !   (nsoil, nroot, etp1, sh2o, smcwlt, smcref, &                     !%dmB
                     !    cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis, &                !%dmB
                     !    !  ---  outputs: &                                              !%dmB
                     !    et1, &                                                          !%dmB
                     !    !  ---  dummys: &                                               !%dmB
                     !    gx &                                                            !%dmB
                     !    )                                                               !%dmB
         !  --- ...  initialize plant transp to zero for all soil layers.                 !%dmB
                     !$acc loop seq                                                       !%dmB
                     do k = 1, nsoil                                                      !%dmB
                        et1(k) = 0.0                                                      !%dmB
                     end do                                                               !%dmB
                                                                                          !%dmB
         !  --- ...  calculate an 'adjusted' potential transpiration                      !%dmB
         !           if statement below to avoid tangent linear problems near zero        !%dmB
         !           note: gx and other terms below redistribute transpiration by         !%dmB
         !           layer, et(k), as a function of soil moisture availability,           !%dmB
         !           while preserving total etp1a.                                        !%dmB
                     if (cmc(i, jj) /= 0.0) then                                          !%dmB
                        etp1a = shdfac(i, jj)*pc(i, jj)*etp1* &                           !%dmB
                                (1.0 - (cmc(i, jj)/cmcmax)**cfactr)                       !%dmB
                     else                                                                 !%dmB
                        etp1a = shdfac(i, jj)*pc(i, jj)*etp1                              !%dmB
                     end if                                                               !%dmB
                                                                                          !%dmB
                     sgx = 0.0                                                            !%dmB
                     !$acc loop seq                                                       !%dmB
                     do i1 = 1, nroot(i, jj)                                              !%dmB
                        gx(i1) = (sh2o(i, i1, jj) - smcwlt(i, jj))/ &                     !%dmB
                                 (smcref(i, jj) - smcwlt(i, jj))                          !%dmB
                        gx(i1) = max(min(gx(i1), 1.0), 0.0)                               !%dmB
                        sgx = sgx + gx(i1)                                                !%dmB
                     end do                                                               !%dmB
                     sgx = sgx/nroot(i, jj)                                               !%dmB
                                                                                          !%dmB
                     denom_3 = 0.0                                                        !%dmB
                     !$acc loop seq                                                       !%dmB
                     do i1 = 1, nroot(i, jj)                                              !%dmB
                        rtx = rtdis(i1) + gx(i1) - sgx                                    !%dmB
                        gx(i1) = gx(i1)*max(rtx, 0.0)                                     !%dmB
                        denom_3 = denom_3 + gx(i1)                                        !%dmB
                     end do                                                               !%dmB
                     if (denom_3 <= 0.0) denom_3 = 1.0                                    !%dmB
                                                                                          !%dmB
                     !$acc loop seq                                                       !%dmB
                     do i1 = 1, nroot(i, jj)                                              !%dmB
                        et1(i1) = etp1a*gx(i1)/denom_3                                    !%dmB
                     end do                                                               !%dmB
                     !end call transp_gpu                                                 !%dmB
                     !$acc loop seq                                                       !%dm
                     do k = 1, nsoil                                                      !%dm
                        ett1 = ett1 + et1(k)                                              !%dm
                     end do                                                               !%dm
                                                                                          !%dm
   !  --- ...  calculate canopy evaporation.                                              !%dm
   !           if statements to avoid tangent linear problems near cmc(i, jj)=0.0.        !%dm
                     if (cmc(i, jj) > 0.0) then                                           !%dm
                        ec1 = shdfac(i, jj)*((cmc(i, jj)/cmcmax)**cfactr)*etp1            !%dm
                     else                                                                 !%dm
                        ec1 = 0.0                                                         !%dm
                     end if                                                               !%dm
                                                                                          !%dm
   !  --- ...  ec(i, jj) should be limited by the total amount of available water         !%dm
   !           on the canopy.  -f.chen, 18-oct-1994                                       !%dm
                     cmc2ms = cmc(i, jj)/dt                                               !%dm
                     ec1 = min(cmc2ms, ec1)                                               !%dm
                  end if                                                                  !%dm
                                                                                          !%dm
               end if   ! end if_etp1_block                                               !%dm
                                                                                          !%dm
   !  --- ...  total up evap and transp types to obtain actual evapotransp                !%dm
               eta1 = edir1 + ett1 + ec1                                                  !%dm
               !end call evapo_gpu                                                        !%dm
                                                                                          !%d
               !call smflx_gpu &                                                          !%do
               !   !  ---  inputs: &                                                      !%do
               !   (nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1_1, &                    !%do
               !    zsoil, slope, frzx, bexp, dksat, dwsat, shdfac, &                     !%do
               !    edir1, ec1, et1, &                                                    !%do
               !    !  ---  input/outputs: &                                              !%do
               !    cmc, sh2o, &                                                          !%do
               !    !  ---  outputs: &                                                    !%do
               !    smc, runoff1, runoff2, runoff3, drip, &                               !%do
               !    !  ---  dummys: &                                                     !%do
               !    ciin, rhsttin, dmax, rhstt, sice, sh2oa, sh2ofg, ai, bi, ci &         !%do
               !    )                                                                     !%do
   !  --- ...  executable code begins here.                                               !%do
               dummy = 0.0                                                                !%do
                                                                                          !%do
   !  --- ...  compute the right hand side of the canopy eqn term ( rhsct )               !%do
                                                                                          !%do
               rhsct = shdfac(i, jj)*prcp1_1 - ec1                                        !%do
                                                                                          !%do
   !  --- ...  convert rhsct (a rate) to trhsct (an amount) and add it to                 !%do
   !           existing cmc(i, jj).  if resulting amt exceeds max capacity, it            !%do
   !           becomes drip(i, jj) and will fall to the grnd.                             !%do
                                                                                          !%do
               drip(i, jj) = 0.0                                                          !%do
               trhsct = dt*rhsct                                                          !%do
               excess = cmc(i, jj) + trhsct                                               !%do
               if (excess > cmcmax) then                                                  !%do
                  drip(i, jj) = excess - cmcmax                                           !%do
               end if                                                                     !%do
                                                                                          !%do
   !  --- ...  pcpdrp is the combined prcp1_1 and drip(i, jj) (from cmc(i, jj)) that      !%do
   !           goes into the soil                                                         !%do
               pcpdrp = (1.0 - shdfac(i, jj))*prcp1_1 + drip(i, jj)/dt                    !%do
                                                                                          !%do
   !  --- ...  store ice content at each soil layer before calling srt & sstep            !%do
               !$acc loop seq                                                             !%do
               do i1 = 1, nsoil                                                           !%do
                  sice(i1) = smc(i, i1, jj) - sh2o(i, i1, jj)                             !%do
               end do                                                                     !%do
                                                                                          !%do
   !  --- ...  call subroutines srt and sstep to solve the soil moisture                  !%do
   !           tendency equations.                                                        !%do
                                                                                          !%do
   !  ---  if the infiltrating precip rate is nontrivial,                                 !%do
   !         (we consider nontrivial to be a precip total over the time step              !%do
   !         exceeding one one-thousandth of the water holding capacity of                !%do
   !         the first soil layer)                                                        !%do
   !       then call the srt/sstep subroutine pair twice in the manner of                 !%do
   !         time scheme "f" (implicit state, averaged coefficient)                       !%do
   !         of section 2 of kalnay and kanamitsu (1988, mwr, vol 116,                    !%do
   !         pages 1945-1958)to minimize 2-delta-t oscillations in the                    !%do
   !         soil moisture value of the top soil layer that can arise because             !%do
   !         of the extreme nonlinear dependence of the soil hydraulic                    !%do
   !         diffusivity coefficient and the hydraulic conductivity on the                !%do
   !         soil moisture state                                                          !%do
   !       otherwise call the srt/sstep subroutine pair once in the manner of             !%do
   !         time scheme "d" (implicit state, explicit coefficient)                       !%do
   !         of section 2 of kalnay and kanamitsu                                         !%do
   !       pcpdrp is units of kg/m**2/s or mm/s, zsoil is negative depth in m             !%do
                                                                                          !%do
   !     if ( pcpdrp .gt. 0.0 ) then                                                      !%do
               if ((pcpdrp*dt) > (0.001*1000.0*(-zsoil(1))*smcmax(i, jj))) then           !%do
                                                                                          !%do
   !  --- ...  frozen ground version:                                                     !%do
   !           smc states replaced by sh2o states in srt subr.  sh2o & sice states        !%do
   !           included in sstep subr.  frozen ground correction factor, frzx             !%do
   !           added.  all water balance calculations using unfrozen water                !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%dox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) - &               !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) - &        !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%dox
                     mxsmc = sh2o(i, 1, jj)                                               !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
         !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the            !%doxC
         !           vertical gradient of unfrozen water. the latter gradient can         !%doxC
         !           become very extreme in freezing/thawing situations, and given        !%doxC
         !           the relatively few and thick soil layers, this gradient sufferes     !%doxC
         !           serious trunction errors yielding erroneously high vertical          !%doxC
         !           transports ofunfrozen water in both directions from huge             !%doxC
         !           hydraulic diffusivity. therefore, we found we had to arbitrarily     !%doxC
         !           constrain wdf                                                        !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%dox
                                                                                          !%dox
                  mxsmc = sh2o(i, 1, jj)                                                  !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%dox
                        mxsmc2 = sh2o(i, k, jj)                                           !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    dummy, rhstt, ai, bi, ci, &                                        !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2ofg, runoff3, smc, &                                            !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
                  !  --- ...  create 'amount' values of variables to be input to the      !%doy
   !           tri-diagonal matrix routine.                                               !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhstt(k) = rhstt(k)*dt                                                  !%doy
                  ai(k) = ai(k)*dt                                                        !%doy
                  bi(k) = 1.+bi(k)*dt                                                     !%doy
                  ci(k) = ci(k)*dt                                                        !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  copy values for input variables before call to rosr12                      !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhsttin(k) = rhstt(k)                                                   !%doy
               end do                                                                     !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsold                                                            !%doy
                  ciin(k) = ci(k)                                                         !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%doy
                                                                                          !%doy
               !call rosr12_gpu &                                                         !%doyv
               !   !  ---  inputs: &                                                      !%doyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%doyv
               !    !  ---  input/outputs: &                                              !%doyv
               !    ciin, &                                                               !%doyv
               !    !  ---  outputs: &                                                    !%doyv
               !    ci, rhstt &                                                           !%doyv
               !    )                                                                     !%doyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%doyv
                                                                                          !%doyv
               ciin(nsoil) = 0.0                                                          !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%doyv
               ci(1) = -ciin(1)/bi(1)                                                     !%doyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%doyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%doyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%doyv
               end do                                                                     !%doyv
                                                                                          !%doyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%doyv
               ci(nsoil) = rhstt(nsoil)                                                   !%doyv
                                                                                          !%doyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  kk = nsoil - k + 1                                                      !%doyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%doyv
               end do                                                                     !%doyv
               !end call rosr12_gpu                                                       !%doyv
                                                                                          !%doy
                                                                                          !%doy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%doy
   !           a new value.  min allowable value of smc will be 0.02.                     !%doy
   !      runoff3(i, jj): runoff within soil layers                                       !%doy
                                                                                          !%doy
               wplus = 0.0                                                                !%doy
               runoff3(i, jj) = 0.0                                                       !%doy
               ddz = -zsoil(1)                                                            !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  if (k /= 1) then                                                        !%doy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%doy
                  end if                                                                  !%doy
                  sh2ofg(k) = sh2o(i, k, jj) + ci(k) + wplus/ddz                          !%doy
                                                                                          !%doy
                  stot = sh2ofg(k) + sice(k)                                              !%doy
                  if (stot > smcmax(i, jj)) then                                          !%doy
                     if (k == 1) then                                                     !%doy
                        ddz = -zsoil(1)                                                   !%doy
                     else                                                                 !%doy
                        kk11 = k - 1                                                      !%doy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%doy
                     end if                                                               !%doy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%doy
                  else                                                                    !%doy
                     wplus = 0.0                                                          !%doy
                  end if                                                                  !%doy
                                                                                          !%doy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%doy
                  sh2ofg(k) = max(smc(i, k, jj) - sice(k), 0.0)                           !%doy
               end do                                                                     !%doy
               runoff3(i, jj) = wplus                                                     !%doy
                                                                                          !%doy
   !  --- ...  update canopy water content/interception (dummy).  convert rhsct to        !%doy
   !           an 'amount' value and add to previous dummy value to get new dummy.        !%doy
                                                                                          !%doy
               dummy = dummy + dt*rhsct                                                   !%doy
               if (dummy < 1.e-20) dummy = 0.0                                            !%doy
               dummy = min(dummy, cmcmax)                                                 !%doy
               !end call sstep_gpu                                                        !%doy
                  !$acc loop seq                                                          !%do
                  do k = 1, nsoil                                                         !%do
                     sh2oa(k) = (sh2o(i, k, jj) + sh2ofg(k))*0.5                          !%do
                  end do                                                                  !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2oa, pcpdrp, zsoil, dwsat, &            !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}). current    !%dox
      !           logic doesn't allow cvfrz be bigger than 3                              !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2oa(1) + sice(1) - &                     !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2oa(ks) + sice(ks) - &              !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2oa(1), sh2oa(2) )                                          !%dox
                     mxsmc = sh2oa(1)                                                     !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf                                         !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2oa(1), sh2oa(2))'                                        !%dox
                                                                                          !%dox
                  mxsmc = sh2oa(1)                                                        !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2oa(k), sh2oa(k+1))'                                   !%dox
                        mxsmc2 = sh2oa(k)                                                 !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2oa(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &         !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2oa(nsoil)/smcmax(i, jj)                               !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2o, runoff3, smc, &                                              !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
   !  --- ...  create 'amount' values of variables to be input to the                     !%doy
   !           tri-diagonal matrix routine.                                               !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhstt(k) = rhstt(k)*dt                                                  !%doy
                  ai(k) = ai(k)*dt                                                        !%doy
                  bi(k) = 1.+bi(k)*dt                                                     !%doy
                  ci(k) = ci(k)*dt                                                        !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  copy values for input variables before call to rosr12                      !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhsttin(k) = rhstt(k)                                                   !%doy
               end do                                                                     !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsold                                                            !%doy
                  ciin(k) = ci(k)                                                         !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%doy
                                                                                          !%doy
               !call rosr12_gpu &                                                         !%doyv
               !   !  ---  inputs: &                                                      !%doyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%doyv
               !    !  ---  input/outputs: &                                              !%doyv
               !    ciin, &                                                               !%doyv
               !    !  ---  outputs: &                                                    !%doyv
               !    ci, rhstt &                                                           !%doyv
               !    )                                                                     !%doyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%doyv
                                                                                          !%doyv
               ciin(nsoil) = 0.0                                                          !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%doyv
               ci(1) = -ciin(1)/bi(1)                                                     !%doyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%doyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%doyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%doyv
               end do                                                                     !%doyv
                                                                                          !%doyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%doyv
               ci(nsoil) = rhstt(nsoil)                                                   !%doyv
                                                                                          !%doyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  kk = nsoil - k + 1                                                      !%doyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%doyv
               end do                                                                     !%doyv
               !end call rosr12_gpu                                                       !%doyv
                                                                                          !%doy
                                                                                          !%doy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%doy
   !           a new value.  min allowable value of smc will be 0.02.                     !%doy
   !      runoff3(i, jj): runoff within soil layers                                       !%doy
                                                                                          !%doy
               wplus = 0.0                                                                !%doy
               runoff3(i, jj) = 0.0                                                       !%doy
               ddz = -zsoil(1)                                                            !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  if (k /= 1) then                                                        !%doy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%doy
                  end if                                                                  !%doy
                  sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                     !%doy
                                                                                          !%doy
                  stot = sh2o(i, k, jj) + sice(k)                                         !%doy
                  if (stot > smcmax(i, jj)) then                                          !%doy
                     if (k == 1) then                                                     !%doy
                        ddz = -zsoil(1)                                                   !%doy
                     else                                                                 !%doy
                        kk11 = k - 1                                                      !%doy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%doy
                     end if                                                               !%doy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%doy
                  else                                                                    !%doy
                     wplus = 0.0                                                          !%doy
                  end if                                                                  !%doy
                                                                                          !%doy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%doy
                  sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                      !%doy
               end do                                                                     !%doy
               runoff3(i, jj) = wplus                                                     !%doy
                                                                                          !%doy
   !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert rhsct      !%doy
   !           to an 'amount' value and add to previous cmc(i, jj) value to get new       !%doy
   !           cmc(i, jj).                                                                !%doy
                                                                                          !%doy
               cmc(i, jj) = cmc(i, jj) + dt*rhsct                                         !%doy
               if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                                  !%doy
               cmc(i, jj) = min(cmc(i, jj), cmcmax)                                       !%doy
               !end call sstep_gpu                                                        !%doy
                                                                                          !%do
               else                                                                       !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%dox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) - &               !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) - &        !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%dox
                     mxsmc = sh2o(i, 1, jj)                                               !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf                                         !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%dox
                                                                                          !%dox
                  mxsmc = sh2o(i, 1, jj)                                                  !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%dox
                        mxsmc2 = sh2o(i, k, jj)                                           !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2o, runoff3, smc, &                                              !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
      !  --- ...  create 'amount' values of variables to be input to the                  !%doy
      !           tri-diagonal matrix routine.                                            !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     rhstt(k) = rhstt(k)*dt                                               !%doy
                     ai(k) = ai(k)*dt                                                     !%doy
                     bi(k) = 1.+bi(k)*dt                                                  !%doy
                     ci(k) = ci(k)*dt                                                     !%doy
                  end do                                                                  !%doy
                                                                                          !%doy
      !  --- ...  copy values for input variables before call to rosr12                   !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     rhsttin(k) = rhstt(k)                                                !%doy
                  end do                                                                  !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsold                                                         !%doy
                     ciin(k) = ci(k)                                                      !%doy
                  end do                                                                  !%doy
                                                                                          !%doy
      !  --- ...  call rosr12 to solve the tri-diagonal matrix                            !%doy
                                                                                          !%doy
                  !call rosr12_gpu &                                                      !%doyv
                  !   !  ---  inputs: &                                                   !%doyv
                  !   (nsoil, ai, bi, rhsttin, &                                          !%doyv
                  !    !  ---  input/outputs: &                                           !%doyv
                  !    ciin, &                                                            !%doyv
                  !    !  ---  outputs: &                                                 !%doyv
                  !    ci, rhstt &                                                        !%doyv
                  !    )                                                                  !%doyv
                  !  --- ...  initialize eqn coef ciin for the lowest soil layer          !%doyv
                                                                                          !%doyv
                  ciin(nsoil) = 0.0                                                       !%doyv
                                                                                          !%doyv
      !  --- ...  solve the coefs for the 1st soil layer                                  !%doyv
                  ci(1) = -ciin(1)/bi(1)                                                  !%doyv
                  rhstt(1) = rhsttin(1)/bi(1)                                             !%doyv
                                                                                          !%doyv
      !  --- ...  solve the coefs for soil layers 2 thru nsoil                            !%doyv
                  !$acc loop seq                                                          !%doyv
                  do k = 2, nsoil                                                         !%doyv
                     ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                     !%doyv
                     rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                       !%doyv
                                *(1.0/(bi(k) + ai(k)*ci(k - 1)))                          !%doyv
                  end do                                                                  !%doyv
                                                                                          !%doyv
      !  --- ...  set ci to rhstt for lowest soil layer                                   !%doyv
                  ci(nsoil) = rhstt(nsoil)                                                !%doyv
                                                                                          !%doyv
      !  --- ...  adjust ci for soil layers 2 thru nsoil                                  !%doyv
                  !$acc loop seq                                                          !%doyv
                  do k = 2, nsoil                                                         !%doyv
                     kk = nsoil - k + 1                                                   !%doyv
                     ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                               !%doyv
                  end do                                                                  !%doyv
                  !end call rosr12_gpu                                                    !%doyv
                                                                                          !%doy
                                                                                          !%doy
      !  --- ...  sum the previous smc value and the matrix solution to get               !%doy
      !           a new value.  min allowable value of smc will be 0.02.                  !%doy
      !      runoff3(i, jj): runoff within soil layers                                    !%doy
                                                                                          !%doy
                  wplus = 0.0                                                             !%doy
                  runoff3(i, jj) = 0.0                                                    !%doy
                  ddz = -zsoil(1)                                                         !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     if (k /= 1) then                                                     !%doy
                        ddz = zsoil(k - 1) - zsoil(k)                                     !%doy
                     end if                                                               !%doy
                     sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                  !%doy
                                                                                          !%doy
                     stot = sh2o(i, k, jj) + sice(k)                                      !%doy
                     if (stot > smcmax(i, jj)) then                                       !%doy
                        if (k == 1) then                                                  !%doy
                           ddz = -zsoil(1)                                                !%doy
                        else                                                              !%doy
                           kk11 = k - 1                                                   !%doy
                           ddz = -zsoil(k) + zsoil(kk11)                                  !%doy
                        end if                                                            !%doy
                        wplus = (stot - smcmax(i, jj))*ddz                                !%doy
                     else                                                                 !%doy
                        wplus = 0.0                                                       !%doy
                     end if                                                               !%doy
                                                                                          !%doy
                     smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                  !%doy
                     sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                   !%doy
                  end do                                                                  !%doy
                  runoff3(i, jj) = wplus                                                  !%doy
                                                                                          !%doy
      !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert         !%doy
      !           rhsct to an 'amount' value and add to previous cmc(i, jj) value to      !%doy
      !           get new cmc(i, jj).                                                     !%doy
                                                                                          !%doy
                  cmc(i, jj) = cmc(i, jj) + dt*rhsct                                      !%doy
                  if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                               !%doy
                  cmc(i, jj) = min(cmc(i, jj), cmcmax)                                    !%doy
                  !end call sstep_gpu                                                     !%doy
                                                                                          !%do
               end if                                                                     !%do
               !end call smflx_gpu                                                        !%do
                                                                                          !%d
            else                                                                          !%d
                                                                                          !%d
!  --- ...  if etp(i, jj) < 0, assume dew(i, jj) forms (transform etp1 into               !%d
!           dew(i, jj) and reinitialize etp1 to zero).                                    !%d
                                                                                          !%d
               eta1 = 0.0                                                                 !%d
               dew(i, jj) = -etp1                                                         !%d
                                                                                          !%d
!  --- ...  convert prcp(i, jj) from 'kg m-2 s-1' to 'm s-1' and add dew(i, jj)           !%d
!           amount.                                                                       !%d
                                                                                          !%d
               prcp1_1 = prcp1_1 + dew(i, jj)                                             !%d
                                                                                          !%d
               !call smflx_gpu &                                                          !%do
               !   !  ---  inputs: &                                                      !%do
               !   (nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1_1, &                    !%do
               !    zsoil, slope, frzx, bexp, dksat, dwsat, shdfac, &                     !%do
               !    edir1, ec1, et1, &                                                    !%do
               !    !  ---  input/outputs: &                                              !%do
               !    cmc, sh2o, &                                                          !%do
               !    !  ---  outputs: &                                                    !%do
               !    smc, runoff1, runoff2, runoff3, drip, &                               !%do
               !    !  ---  dummys: &                                                     !%do
               !    ciin, rhsttin, dmax, rhstt, sice, sh2oa, sh2ofg, ai, bi, ci &         !%do
               !    )                                                                     !%do
   !  --- ...  executable code begins here.                                               !%do
               dummy = 0.0                                                                !%do
                                                                                          !%do
   !  --- ...  compute the right hand side of the canopy eqn term ( rhsct )               !%do
                                                                                          !%do
               rhsct = shdfac(i, jj)*prcp1_1 - ec1                                        !%do
                                                                                          !%do
   !  --- ...  convert rhsct (a rate) to trhsct (an amount) and add it to                 !%do
   !           existing cmc(i, jj).  if resulting amt exceeds max capacity, it            !%do
   !           becomes drip(i, jj) and will fall to the grnd.                             !%do
                                                                                          !%do
               drip(i, jj) = 0.0                                                          !%do
               trhsct = dt*rhsct                                                          !%do
               excess = cmc(i, jj) + trhsct                                               !%do
               if (excess > cmcmax) then                                                  !%do
                  drip(i, jj) = excess - cmcmax                                           !%do
               end if                                                                     !%do
                                                                                          !%do
   !  --- ...  pcpdrp is the combined prcp1_1 and drip(i, jj) (from cmc(i, jj)) that      !%do
   !           goes into the soil                                                         !%do
               pcpdrp = (1.0 - shdfac(i, jj))*prcp1_1 + drip(i, jj)/dt                    !%do
                                                                                          !%do
   !  --- ...  store ice content at each soil layer before calling srt & sstep            !%do
               !$acc loop seq                                                             !%do
               do i1 = 1, nsoil                                                           !%do
                  sice(i1) = smc(i, i1, jj) - sh2o(i, i1, jj)                             !%do
               end do                                                                     !%do
                                                                                          !%do
   !  --- ...  call subroutines srt and sstep to solve the soil moisture                  !%do
   !           tendency equations.                                                        !%do
                                                                                          !%do
   !  ---  if the infiltrating precip rate is nontrivial,                                 !%do
   !         (we consider nontrivial to be a precip total over the time step              !%do
   !         exceeding one one-thousandth of the water holding capacity of                !%do
   !         the first soil layer)                                                        !%do
   !       then call the srt/sstep subroutine pair twice in the manner of                 !%do
   !         time scheme "f" (implicit state, averaged coefficient)                       !%do
   !         of section 2 of kalnay and kanamitsu (1988, mwr, vol 116,                    !%do
   !         pages 1945-1958)to minimize 2-delta-t oscillations in the                    !%do
   !         soil moisture value of the top soil layer that can arise because             !%do
   !         of the extreme nonlinear dependence of the soil hydraulic                    !%do
   !         diffusivity coefficient and the hydraulic conductivity on the                !%do
   !         soil moisture state                                                          !%do
   !       otherwise call the srt/sstep subroutine pair once in the manner of             !%do
   !         time scheme "d" (implicit state, explicit coefficient)                       !%do
   !         of section 2 of kalnay and kanamitsu                                         !%do
   !       pcpdrp is units of kg/m**2/s or mm/s, zsoil is negative depth in m             !%do
                                                                                          !%do
   !     if ( pcpdrp .gt. 0.0 ) then                                                      !%do
               if ((pcpdrp*dt) > (0.001*1000.0*(-zsoil(1))*smcmax(i, jj))) then           !%do
                                                                                          !%do
   !  --- ...  frozen ground version:                                                     !%do
   !           smc states replaced by sh2o states in srt subr.  sh2o & sice states        !%do
   !           included in sstep subr.  frozen ground correction factor, frzx             !%do
   !           added.  all water balance calculations using unfrozen water                !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%dox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) - &               !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) - &        !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%dox
                     mxsmc = sh2o(i, 1, jj)                                               !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf                                         !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%dox
                                                                                          !%dox
                  mxsmc = sh2o(i, 1, jj)                                                  !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%dox
                        mxsmc2 = sh2o(i, k, jj)                                           !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    dummy, rhstt, ai, bi, ci, &                                        !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2ofg, runoff3, smc, &                                            !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
                  !  --- ...  create 'amount' values of variables to be input to the      !%doy
   !           tri-diagonal matrix routine.                                               !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhstt(k) = rhstt(k)*dt                                                  !%doy
                  ai(k) = ai(k)*dt                                                        !%doy
                  bi(k) = 1.+bi(k)*dt                                                     !%doy
                  ci(k) = ci(k)*dt                                                        !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  copy values for input variables before call to rosr12                      !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhsttin(k) = rhstt(k)                                                   !%doy
               end do                                                                     !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsold                                                            !%doy
                  ciin(k) = ci(k)                                                         !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%doy
                                                                                          !%doy
               !call rosr12_gpu &                                                         !%doyv
               !   !  ---  inputs: &                                                      !%doyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%doyv
               !    !  ---  input/outputs: &                                              !%doyv
               !    ciin, &                                                               !%doyv
               !    !  ---  outputs: &                                                    !%doyv
               !    ci, rhstt &                                                           !%doyv
               !    )                                                                     !%doyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%doyv
                                                                                          !%doyv
               ciin(nsoil) = 0.0                                                          !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%doyv
               ci(1) = -ciin(1)/bi(1)                                                     !%doyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%doyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%doyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%doyv
               end do                                                                     !%doyv
                                                                                          !%doyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%doyv
               ci(nsoil) = rhstt(nsoil)                                                   !%doyv
                                                                                          !%doyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%doyv
                                                                                          !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  kk = nsoil - k + 1                                                      !%doyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%doyv
               end do                                                                     !%doyv
               !end call rosr12_gpu                                                       !%doyv
                                                                                          !%doy
                                                                                          !%doy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%doy
   !           a new value.  min allowable value of smc will be 0.02.                     !%doy
   !      runoff3(i, jj): runoff within soil layers                                       !%doy
                                                                                          !%doy
               wplus = 0.0                                                                !%doy
               runoff3(i, jj) = 0.0                                                       !%doy
               ddz = -zsoil(1)                                                            !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  if (k /= 1) then                                                        !%doy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%doy
                  end if                                                                  !%doy
                  sh2ofg(k) = sh2o(i, k, jj) + ci(k) + wplus/ddz                          !%doy
                                                                                          !%doy
                  stot = sh2ofg(k) + sice(k)                                              !%doy
                  if (stot > smcmax(i, jj)) then                                          !%doy
                     if (k == 1) then                                                     !%doy
                        ddz = -zsoil(1)                                                   !%doy
                     else                                                                 !%doy
                        kk11 = k - 1                                                      !%doy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%doy
                     end if                                                               !%doy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%doy
                  else                                                                    !%doy
                     wplus = 0.0                                                          !%doy
                  end if                                                                  !%doy
                                                                                          !%doy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%doy
                  sh2ofg(k) = max(smc(i, k, jj) - sice(k), 0.0)                           !%doy
               end do                                                                     !%doy
               runoff3(i, jj) = wplus                                                     !%doy
                                                                                          !%doy
   !  --- ...  update canopy water content/interception (dummy).  convert rhsct to        !%doy
   !           an 'amount' value and add to previous dummy value to get new dummy.        !%doy
                                                                                          !%doy
               dummy = dummy + dt*rhsct                                                   !%doy
               if (dummy < 1.e-20) dummy = 0.0                                            !%doy
               dummy = min(dummy, cmcmax)                                                 !%doy
               !end call sstep_gpu                                                        !%doy
                                                                                          !%do
                  !$acc loop seq                                                          !%do
                  do k = 1, nsoil                                                         !%do
                     sh2oa(k) = (sh2o(i, k, jj) + sh2ofg(k))*0.5                          !%do
                  end do                                                                  !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2oa, pcpdrp, zsoil, dwsat, &            !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%dox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2oa(1) + sice(1) - &                     !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                                                                                          !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2oa(ks) + sice(ks) - &              !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                                                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2oa(1), sh2oa(2) )                                          !%dox
                     mxsmc = sh2oa(1)                                                     !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf                                         !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2oa(1), sh2oa(2))'                                        !%dox
                                                                                          !%dox
                  mxsmc = sh2oa(1)                                                        !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2oa(k), sh2oa(k+1))'                                   !%dox
                        mxsmc2 = sh2oa(k)                                                 !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2oa(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &         !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2oa(nsoil)/smcmax(i, jj)                               !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2o, runoff3, smc, &                                              !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
   !  --- ...  create 'amount' values of variables to be input to the                     !%doy
   !           tri-diagonal matrix routine.                                               !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhstt(k) = rhstt(k)*dt                                                  !%doy
                  ai(k) = ai(k)*dt                                                        !%doy
                  bi(k) = 1.+bi(k)*dt                                                     !%doy
                  ci(k) = ci(k)*dt                                                        !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  copy values for input variables before call to rosr12                      !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  rhsttin(k) = rhstt(k)                                                   !%doy
               end do                                                                     !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsold                                                            !%doy
                  ciin(k) = ci(k)                                                         !%doy
               end do                                                                     !%doy
                                                                                          !%doy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%doy
                                                                                          !%doy
               !call rosr12_gpu &                                                         !%doyv
               !   !  ---  inputs: &                                                      !%doyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%doyv
               !    !  ---  input/outputs: &                                              !%doyv
               !    ciin, &                                                               !%doyv
               !    !  ---  outputs: &                                                    !%doyv
               !    ci, rhstt &                                                           !%doyv
               !    )                                                                     !%doyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%doyv
                                                                                          !%doyv
               ciin(nsoil) = 0.0                                                          !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%doyv
               ci(1) = -ciin(1)/bi(1)                                                     !%doyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%doyv
                                                                                          !%doyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%doyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%doyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%doyv
               end do                                                                     !%doyv
                                                                                          !%doyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%doyv
               ci(nsoil) = rhstt(nsoil)                                                   !%doyv
                                                                                          !%doyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%doyv
                                                                                          !%doyv
               !$acc loop seq                                                             !%doyv
               do k = 2, nsoil                                                            !%doyv
                  kk = nsoil - k + 1                                                      !%doyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%doyv
               end do                                                                     !%doyv
               !end call rosr12_gpu                                                       !%doyv
                                                                                          !%doy
                                                                                          !%doy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%doy
   !           a new value.  min allowable value of smc will be 0.02.                     !%doy
   !      runoff3(i, jj): runoff within soil layers                                       !%doy
                                                                                          !%doy
               wplus = 0.0                                                                !%doy
               runoff3(i, jj) = 0.0                                                       !%doy
               ddz = -zsoil(1)                                                            !%doy
                                                                                          !%doy
               !$acc loop seq                                                             !%doy
               do k = 1, nsoil                                                            !%doy
                  if (k /= 1) then                                                        !%doy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%doy
                  end if                                                                  !%doy
                  sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                     !%doy
                                                                                          !%doy
                  stot = sh2o(i, k, jj) + sice(k)                                         !%doy
                  if (stot > smcmax(i, jj)) then                                          !%doy
                     if (k == 1) then                                                     !%doy
                        ddz = -zsoil(1)                                                   !%doy
                     else                                                                 !%doy
                        kk11 = k - 1                                                      !%doy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%doy
                     end if                                                               !%doy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%doy
                  else                                                                    !%doy
                     wplus = 0.0                                                          !%doy
                  end if                                                                  !%doy
                                                                                          !%doy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%doy
                  sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                      !%doy
               end do                                                                     !%doy
               runoff3(i, jj) = wplus                                                     !%doy
                                                                                          !%doy
   !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert rhsct      !%doy
   !           to an 'amount' value and add to previous cmc(i, jj) value to get new       !%doy
   !           cmc(i, jj).                                                                !%doy
                                                                                          !%doy
               cmc(i, jj) = cmc(i, jj) + dt*rhsct                                         !%doy
               if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                                  !%doy
               cmc(i, jj) = min(cmc(i, jj), cmcmax)                                       !%doy
               !end call sstep_gpu                                                        !%doy
                                                                                          !%do
               else                                                                       !%do
                                                                                          !%do
                  !call srt_gpu &                                                         !%dox
                  !   !  ---  inputs: &                                                   !%dox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%dox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%dox
                  !    !  ---  outputs: &                                                 !%dox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%dox
                  !    !  ---  dummys: &                                                  !%dox
                  !    dmax &                                                             !%dox
                  !    )                                                                  !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%dox
      !           of areal distribution function of soil ice content which equals         !%dox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%dox
      !           based on field data cv depends on areal mean of frozen depth, and       !%dox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%dox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%dox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
                                                                                          !%dox
      ! ----------------------------------------------------------------------            !%dox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%dox
      !           the infiltration formule from schaake and koren model.                  !%dox
      !           modified by q duan                                                      !%dox
                                                                                          !%dox
                  iohinf = 1                                                              !%dox
                                                                                          !%dox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%dox
      !          soil layers.                                                             !%dox
                                                                                          !%dox
                  sicemax = 0.0                                                           !%dox
                  !$acc loop seq                                                          !%dox
                  do ks = 1, nsoil                                                        !%dox
                     if (sice(ks) > sicemax) then                                         !%dox
                        sicemax = sice(ks)                                                !%dox
                     end if                                                               !%dox
                  end do                                                                  !%dox
                                                                                          !%dox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%dox
                  pddum = pcpdrp                                                          !%dox
                  runoff1(i, jj) = 0.0                                                    !%dox
                                                                                          !%dox
                  if (pcpdrp /= 0.0) then                                                 !%dox
                                                                                          !%dox
      !  --- ...  modified by q. duan, 5/16/94                                            !%dox
                     dt1 = dt/86400.                                                      !%dox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%dox
                     dmax(1) = -zsoil(1)*smcav                                            !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                                                                                          !%dox
                     dice = -zsoil(1)*sice(1)                                             !%dox
                                                                                          !%dox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) - &               !%dox
                               smcwlt(i, jj))/smcav)                                      !%dox
                     dd = dmax(1)                                                         !%dox
                                                                                          !%dox
                     !$acc loop seq                                                       !%dox
                     do ks = 2, nsoil                                                     !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%dox
                                                                                          !%dox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%dox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) - &        !%dox
                                   smcwlt(i, jj))/smcav)                                  !%dox
                        dd = dd + dmax(ks)                                                !%dox
                     end do                                                               !%dox
                                                                                          !%dox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%dox
      !           in below, remove the sqrt in above                                      !%dox
                     val = 1.0 - exp(-kdt*dt1)                                            !%dox
                     ddt = dd*val                                                         !%dox
                                                                                          !%dox
                     px = pcpdrp*dt                                                       !%dox
                     if (px < 0.0) px = 0.0                                               !%dox
                                                                                          !%dox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%dox
                                                                                          !%dox
      !  --- ...  frozen ground version:                                                  !%dox
      !           reduction of infiltration based on frozen ground parameters             !%dox
                                                                                          !%dox
                     fcr = 1.                                                             !%dox
                     if (dice > 1.e-2) then                                               !%dox
                        acrt = cvfrz*frzx/dice                                            !%dox
                        sum = 1.                                                          !%dox
                                                                                          !%dox
                        ialp1 = cvfrz - 1                                                 !%dox
                        !$acc loop seq                                                    !%dox
                        do j = 1, ialp1                                                   !%dox
                           k = 1                                                          !%dox
                                                                                          !%dox
                           !$acc loop seq                                                 !%dox
                           do j1 = j + 1, ialp1                                           !%dox
                              k = k*j1                                                    !%dox
                           end do                                                         !%dox
                                                                                          !%dox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%dox
                        end do                                                            !%dox
                                                                                          !%dox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%dox
                     end if                                                               !%dox
                     infmax = infmax*fcr                                                  !%dox
                                                                                          !%dox
      !  --- ...  correction of infiltration limitation:                                  !%dox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%dox
      !           of hydrolic conductivity                                                !%dox
                                                                                          !%dox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%dox
                     mxsmc = sh2o(i, 1, jj)                                               !%dox
                                                                                          !%dox
                     !call wdfcnd_gpu &                                                   !%doxC
                     !   !  ---  inputs: &                                                !%doxC
                     !   (mxsmc, smcmax(i, jj), bexp, dksat, dwsat, sicemax, &            !%doxC
                     !    !  ---  outputs: &                                              !%doxC
                     !    wdf, wcnd &                                                     !%doxC
                     !    )                                                               !%doxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%doxC
                                                                                          !%doxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%doxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%doxC
                                                                                          !%doxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%doxC
                                                                                          !%doxC
                     expon = bexp + 2.0                                                   !%doxC
                     wdf = dwsat*factr2**expon                                            !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf                                         !%doxC
         !                                                                                !%doxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%doxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%doxC
                     if (sicemax > 0.0) then                                              !%doxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%doxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%doxC
                     end if                                                               !%doxC
                                                                                          !%doxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%doxC
                     expon = (2.0*bexp) + 3.0                                             !%doxC
                     wcnd = dksat*factr2**expon                                           !%doxC
                     !end call wdfcnd_gpu                                                 !%doxC
                                                                                          !%dox
                     infmax = max(infmax, wcnd)                                           !%dox
                     infmax = min(infmax, px)                                             !%dox
                                                                                          !%dox
                     if (pcpdrp > infmax) then                                            !%dox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%dox
                        pddum = infmax                                                    !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                  end if   ! end if_pcpdrp_block                                          !%dox
                                                                                          !%dox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%dox
      !          in line below replaced with new approach in 2nd line:                    !%dox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%dox
                                                                                          !%dox
                  mxsmc = sh2o(i, 1, jj)                                                  !%dox
                                                                                          !%dox
                  !call wdfcnd_gpu &                                                      !%doxC
                  !   !  ---  inputs: &                                                   !%doxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%doxC
                  !    !  ---  outputs: &                                                 !%doxC
                  !    wdf, wcnd &                                                        !%doxC
                  !    )                                                                  !%doxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%doxC
                                                                                          !%doxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%doxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%doxC
                                                                                          !%doxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%doxC
                                                                                          !%doxC
                  expon = bexp + 2.0                                                      !%doxC
                  wdf = dwsat*factr2**expon                                               !%doxC
                                                                                          !%doxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%doxC
      !           gradient of unfrozen water. the latter gradient can become very         !%doxC
      !           extreme in freezing/thawing situations, and given the relatively        !%doxC
      !           few and thick soil layers, this gradient sufferes serious               !%doxC
      !           trunction errors yielding erroneously high vertical transports of       !%doxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%doxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%doxC
      !                                                                                   !%doxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%doxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%doxC
                  if (sicemax > 0.0) then                                                 !%doxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%doxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%doxC
                  end if                                                                  !%doxC
                                                                                          !%doxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%doxC
                  expon = (2.0*bexp) + 3.0                                                !%doxC
                  wcnd = dksat*factr2**expon                                              !%doxC
                  !end call wdfcnd_gpu                                                    !%doxC
                                                                                          !%dox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%dox
                                                                                          !%dox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%dox
                  ai(1) = 0.0                                                             !%dox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%dox
                  ci(1) = -bi(1)                                                          !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%dox
      !           moisture gradient btwn the top and next to top layers.                  !%dox
                                                                                          !%dox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%dox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%dox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%dox
                                                                                          !%dox
      !  --- ...  initialize ddz2                                                         !%dox
                                                                                          !%dox
                  ddz2 = 0.0                                                              !%dox
                                                                                          !%dox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%dox
                  !$acc loop seq                                                          !%dox
                  do k = 2, nsoil                                                         !%dox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%dox
                     if (k /= nsoil) then                                                 !%dox
                        slopx = 1.0                                                       !%dox
                                                                                          !%dox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%dox
      !           in line below replaced with new approach in 2nd line:                   !%dox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%dox
                        mxsmc2 = sh2o(i, k, jj)                                           !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%dox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%dox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%dox
                                                                                          !%dox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%dox
                                                                                          !%dox
                        ddz2 = 2.0/denom_4                                                !%dox
                        ci(k) = -wdf2*ddz2/denom2                                         !%dox
                                                                                          !%dox
                     else   ! if_k_block                                                  !%dox
                                                                                          !%dox
      !  --- ...  slope of bottom layer is introduced                                     !%dox
                        slopx = slope                                                     !%dox
                                                                                          !%dox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%dox
      !           for this layer                                                          !%dox
                                                                                          !%dox
                        !call wdfcnd_gpu &                                                !%doxC
                        !   !  ---  inputs: &                                             !%doxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%doxC
                        !    !  ---  outputs: &                                           !%doxC
                        !    wdf2, wcnd2 &                                                !%doxC
                        !    )                                                            !%doxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%doxC
                                                                                          !%doxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%doxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%doxC
                                                                                          !%doxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%doxC
                                                                                          !%doxC
                        expon = bexp + 2.0                                                !%doxC
                        wdf2 = dwsat*factr2**expon                                        !%doxC
                                                                                          !%doxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%doxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%doxC
            !           become very extreme in freezing/thawing situations, and given     !%doxC
            !           the relatively few and thick soil layers, this gradient           !%doxC
            !           sufferes serious trunction errors yielding erroneously high       !%doxC
            !           vertical transports of unfrozen water in both directions from     !%doxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%doxC
            !           arbitrarily constrain wdf2                                        !%doxC
            !                                                                             !%doxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%doxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%doxC
                        if (sicemax > 0.0) then                                           !%doxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%doxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%doxC
                        end if                                                            !%doxC
                                                                                          !%doxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%doxC
                        expon = (2.0*bexp) + 3.0                                          !%doxC
                        wcnd2 = dksat*factr2**expon                                       !%doxC
                        !end call wdfcnd_gpu                                              !%doxC
                                                                                          !%dox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%dox
                        dsmdz2 = 0.0                                                      !%dox
                                                                                          !%dox
      !  --- ...  set matrix coef ci to zero                                              !%dox
                                                                                          !%dox
                        ci(k) = 0.0                                                       !%dox
                                                                                          !%dox
                     end if   ! end if_k_block                                            !%dox
                                                                                          !%dox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%dox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%dox
                     rhstt(k) = numer/(-denom2)                                           !%dox
                                                                                          !%dox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%dox
                                                                                          !%dox
                     ai(k) = -wdf*ddz_1/denom2                                            !%dox
                     bi(k) = -(ai(k) + ci(k))                                             !%dox
                                                                                          !%dox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%dox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%dox
                     if (k == nsoil) then                                                 !%dox
                        runoff2(i, jj) = slopx*wcnd2                                      !%dox
                     end if                                                               !%dox
                                                                                          !%dox
                     if (k /= nsoil) then                                                 !%dox
                        wdf = wdf2                                                        !%dox
                        wcnd = wcnd2                                                      !%dox
                        dsmdz = dsmdz2                                                    !%dox
                        ddz_1 = ddz2                                                      !%dox
                     end if                                                               !%dox
                  end do   ! end do_k_loop                                                !%dox
                  !end call srt_gpu                                                       !%dox
                                                                                          !%do
                  !call sstep_gpu &                                                       !%doy
                  !   !  ---  inputs: &                                                   !%doy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%doy
                  !    !  ---  input/outputs: &                                           !%doy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%doy
                  !    !  ---  outputs: &                                                 !%doy
                  !    sh2o, runoff3, smc, &                                              !%doy
                  !    !  ---  dummys: &                                                  !%doy
                  !    ciin, rhsttin &                                                    !%doy
                  !    )                                                                  !%doy
      !  --- ...  create 'amount' values of variables to be input to the                  !%doy
      !           tri-diagonal matrix routine.                                            !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     rhstt(k) = rhstt(k)*dt                                               !%doy
                     ai(k) = ai(k)*dt                                                     !%doy
                     bi(k) = 1.+bi(k)*dt                                                  !%doy
                     ci(k) = ci(k)*dt                                                     !%doy
                  end do                                                                  !%doy
                                                                                          !%doy
      !  --- ...  copy values for input variables before call to rosr12                   !%doy
                                                                                          !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     rhsttin(k) = rhstt(k)                                                !%doy
                  end do                                                                  !%doy
                                                                                          !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsold                                                         !%doy
                     ciin(k) = ci(k)                                                      !%doy
                  end do                                                                  !%doy
                                                                                          !%doy
      !  --- ...  call rosr12 to solve the tri-diagonal matrix                            !%doy
                                                                                          !%doy
                  !call rosr12_gpu &                                                      !%doyv
                  !   !  ---  inputs: &                                                   !%doyv
                  !   (nsoil, ai, bi, rhsttin, &                                          !%doyv
                  !    !  ---  input/outputs: &                                           !%doyv
                  !    ciin, &                                                            !%doyv
                  !    !  ---  outputs: &                                                 !%doyv
                  !    ci, rhstt &                                                        !%doyv
                  !    )                                                                  !%doyv
                  !  --- ...  initialize eqn coef ciin for the lowest soil layer          !%doyv
                                                                                          !%doyv
                  ciin(nsoil) = 0.0                                                       !%doyv
                                                                                          !%doyv
      !  --- ...  solve the coefs for the 1st soil layer                                  !%doyv
                  ci(1) = -ciin(1)/bi(1)                                                  !%doyv
                  rhstt(1) = rhsttin(1)/bi(1)                                             !%doyv
                                                                                          !%doyv
      !  --- ...  solve the coefs for soil layers 2 thru nsoil                            !%doyv
                  !$acc loop seq                                                          !%doyv
                  do k = 2, nsoil                                                         !%doyv
                     ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                     !%doyv
                     rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                       !%doyv
                                *(1.0/(bi(k) + ai(k)*ci(k - 1)))                          !%doyv
                  end do                                                                  !%doyv
                                                                                          !%doyv
      !  --- ...  set ci to rhstt for lowest soil layer                                   !%doyv
                  ci(nsoil) = rhstt(nsoil)                                                !%doyv
                                                                                          !%doyv
      !  --- ...  adjust ci for soil layers 2 thru nsoil                                  !%doyv
                                                                                          !%doyv
                  !$acc loop seq                                                          !%doyv
                  do k = 2, nsoil                                                         !%doyv
                     kk = nsoil - k + 1                                                   !%doyv
                     ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                               !%doyv
                  end do                                                                  !%doyv
                  !end call rosr12_gpu                                                    !%doyv
                                                                                          !%doy
                                                                                          !%doy
      !  --- ...  sum the previous smc value and the matrix solution to get               !%doy
      !           a new value.  min allowable value of smc will be 0.02.                  !%doy
      !      runoff3(i, jj): runoff within soil layers                                    !%doy
                                                                                          !%doy
                  wplus = 0.0                                                             !%doy
                  runoff3(i, jj) = 0.0                                                    !%doy
                  ddz = -zsoil(1)                                                         !%doy
                                                                                          !%doy
                  !$acc loop seq                                                          !%doy
                  do k = 1, nsoil                                                         !%doy
                     if (k /= 1) then                                                     !%doy
                        ddz = zsoil(k - 1) - zsoil(k)                                     !%doy
                     end if                                                               !%doy
                     sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                  !%doy
                                                                                          !%doy
                     stot = sh2o(i, k, jj) + sice(k)                                      !%doy
                     if (stot > smcmax(i, jj)) then                                       !%doy
                        if (k == 1) then                                                  !%doy
                           ddz = -zsoil(1)                                                !%doy
                        else                                                              !%doy
                           kk11 = k - 1                                                   !%doy
                           ddz = -zsoil(k) + zsoil(kk11)                                  !%doy
                        end if                                                            !%doy
                        wplus = (stot - smcmax(i, jj))*ddz                                !%doy
                     else                                                                 !%doy
                        wplus = 0.0                                                       !%doy
                     end if                                                               !%doy
                                                                                          !%doy
                     smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                  !%doy
                     sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                   !%doy
                  end do                                                                  !%doy
                  runoff3(i, jj) = wplus                                                  !%doy
                                                                                          !%doy
      !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert         !%doy
      !           rhsct to an 'amount' value and add to previous cmc(i, jj) value to      !%doy
      !           get new cmc(i, jj).                                                     !%doy
                                                                                          !%doy
                  cmc(i, jj) = cmc(i, jj) + dt*rhsct                                      !%doy
                  if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                               !%doy
                  cmc(i, jj) = min(cmc(i, jj), cmcmax)                                    !%doy
                  !end call sstep_gpu                                                     !%doy
                                                                                          !%do
               end if                                                                     !%do
               !end call smflx_gpu                                                        !%do
                                                                                          !%d
            end if   ! end if_etp_block                                                   !%d
                                                                                          !%d
!  --- ...  convert modeled evapotranspiration fm  m s-1  to  kg m-2 s-1                  !%d
                                                                                          !%d
            eta(i, jj) = eta1*1000.0                                                      !%d
            edir(i, jj) = edir1*1000.0                                                    !%d
            ec(i, jj) = ec1*1000.0                                                        !%d
            !$acc loop seq                                                                !%d
            do k = 1, nsoil                                                               !%d
               et(i, k, jj) = et1(k)*1000.0                                               !%d
            end do                                                                        !%d
            ett(i, jj) = ett1*1000.0                                                      !%d
                                                                                          !%d
!  --- ...  based on etp(i, jj) and e values, determine beta(i, jj)                       !%d
            if (etp(i, jj) <= 0.0) then                                                   !%d
               beta(i, jj) = 0.0                                                          !%d
               if (etp(i, jj) < 0.0) then                                                 !%d
                  beta(i, jj) = 1.0                                                       !%d
               end if                                                                     !%d
            else                                                                          !%d
               beta(i, jj) = eta(i, jj)/etp(i, jj)                                        !%d
            end if                                                                        !%d
                                                                                          !%d
!  --- ...  get soil thermal diffuxivity/conductivity for top soil lyr,                   !%d
!           calc. adjusted top lyr soil temp and adjusted soil flux, then                 !%d
!           call shflx to compute/update soil heat flux and soil temps.                   !%d
                                                                                          !%d
            !call tdfcnd_gpu &                                                            !%dl
            !   !  ---  inputs: &                                                         !%dl
            !   (smc(1), quartz, smcmax(i, jj), sh2o(1), &                                !%dl
            !    !  ---  outputs: &                                                       !%dl
            !    df2 &                                                                    !%dl
            !    )                                                                        !%dl
            k = 1                                                                         !%dl
!  --- ...  if the soil has any moisture content compute a partial sum/product            !%dl
!           otherwise use a constant value which works well with most soils               !%dl
                                                                                          !%dl
!  --- ...  saturation ratio:                                                             !%dl
            satratio = smc(i, k, jj)/smcmax(i, jj)                                        !%dl
                                                                                          !%dl
!  --- ...  parameters  w/(m.k)                                                           !%dl
            thkice = 2.2                                                                  !%dl
            thkw = 0.57                                                                   !%dl
            thko = 2.0                                                                    !%dl
!     if (quartz <= 0.2) thko = 3.0                                                       !%dl
            thkqtz = 7.7                                                                  !%dl
                                                                                          !%dl
!  --- ...  solids' conductivity                                                          !%dl
                                                                                          !%dl
            thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                                !%dl
                                                                                          !%dl
!  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0. (100% frozen))           !%dl
                                                                                          !%dl
            xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)                    !%dl
                                                                                          !%dl
!  --- ...  unfrozen volume for saturation (porosity*xunfroz)                             !%dl
                                                                                          !%dl
            xu = xunfroz*smcmax(i, jj)                                                    !%dl
                                                                                          !%dl
!  --- ...  saturated thermal conductivity                                                !%dl
                                                                                          !%dl
            thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu)*thkw**(xu)     !%dl
                                                                                          !%dl
!  --- ...  dry density in kg/m3                                                          !%dl
                                                                                          !%dl
            gammd = (1.0 - smcmax(i, jj))*2700.0                                          !%dl
                                                                                          !%dl
!  --- ...  dry thermal conductivity in w.m-1.k-1                                         !%dl
                                                                                          !%dl
            thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                          !%dl
            if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then         ! frozen            !%dl
               ake = satratio                                                             !%dl
                                                                                          !%dl
            else                                  ! unfrozen                              !%dl
                                                                                          !%dl
!  --- ...  range of validity for the kersten number (ake)                                !%dl
               if (satratio > 0.1) then                                                   !%dl
                                                                                          !%dl
!  --- ...  kersten number (using "fine" formula, valid for soils containing              !%dl
!           at least 5% of particles with diameter less than 2.e-6 meters.)               !%dl
!           (for "coarse" formula, see peters-lidard et al., 1998).                       !%dl
                                                                                          !%dl
                  ake = log10(satratio) + 1.0                                             !%dl
                                                                                          !%dl
               else                                                                       !%dl
                                                                                          !%dl
!  --- ...  use k = kdry                                                                  !%dl
                  ake = 0.0                                                               !%dl
                                                                                          !%dl
               end if   ! end if_satratio_block                                           !%dl
                                                                                          !%dl
            end if   ! end if_sh2o+0.0005_block                                           !%dl
                                                                                          !%dl
!  --- ...  thermal conductivity                                                          !%dl
            df2 = ake*(thksat - thkdry) + thkdry                                          !%dl
            !end call tdfcnd_gpu                                                          !%dl
                                                                                          !%d
            if (ivegsrc .ge. 1) then                                                      !%d
!urban                                                                                    !%d
               if (vegtyp(i, jj) == 13) then                                              !%d
                  df2 = 3.24                                                              !%d
               end if                                                                     !%d
            end if                                                                        !%d
                                                                                          !%d
!  --- ... vegetation greenness fraction reduction in subsurface heat                     !%d
!          flux via reduction factor, which is convenient to apply here                   !%d
!          to thermal diffusivity that is later used in hrt to compute                    !%d
!          sub sfc heat flux (see additional comments on veg effect                       !%d
!          sub-sfc heat flx in routine sflx)                                              !%d
                                                                                          !%d
            df2 = df2*exp(sbeta*shdfac(i, jj))                                            !%d
                                                                                          !%d
!  --- ...  compute intermediate terms passed to routine hrt (via routine                 !%d
!           shflx below) for use in computing subsurface heat flux in hrt                 !%d
                                                                                          !%d
            yynum = fdown - sfcems(i, jj)*sigma1*t24                                      !%d
            yy = sfctmp(i, jj) + (yynum/rch + th2(i, jj) - sfctmp(i, jj) - &              !%d
                 beta(i, jj)*epsca)/rr                                                    !%d
            zz1 = df2/(-0.5*zsoil(1)*rch*rr) + 1.0                                        !%d
                                                                                          !%d
            !call shflx_gpu &                                                             !%dn
            !   !  ---  inputs: &                                                         !%dn
            !   (nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot, &                          !%dn
            !    psisat, bexp, df2, ice, quartz, csoil, vegtyp, ivegsrc,&                 !%dn
            !    !  ---  input/outputs: &                                                 !%dn
            !    stc, t1, tbot, sh2o, df1, &                                              !%dn
            !    !  ---  outputs: &                                                       !%dn
            !    ssoil, &                                                                 !%dn
            !    !  --- dummys: &                                                         !%dn
            !    ciin, rhstsin, ai, bi, ci, rhsts, stcf, stsoil &                         !%dn
            !    )                                                                        !%dn
            oldt1 = t1(i, jj)                                                             !%dn
            !$acc loop seq                                                                !%dn
            do i1 = 1, nsoil                                                              !%dn
               stsoil(i1) = stc(i, i1, jj)                                                !%dn
            end do                                                                        !%dn
                                                                                          !%dn
!  --- ...  hrt routine calcs the right hand side of the soil temp dif eqn                !%dn
            if (ice /= 0) then                                                            !%dn
                                                                                          !%dn
!  --- ...  sea-ice case, glacial-ice case                                                !%dn
                                                                                          !%dn
               !call hrtice_gpu &                                                         !%dnt
               !   !  ---  inputs: &                                                      !%dnt
               !   (nsoil, stc, zsoil, yy, zz1, df2, ice, &                               !%dnt
               !    !  ---  input/outputs: &                                              !%dnt
               !    tbot, &                                                               !%dnt
               !    !  ---  outputs: &                                                    !%dnt
               !    rhsts, ai, bi, ci &                                                   !%dnt
               !    )                                                                     !%dnt
   !  --- ...  set a nominal universal value of the sea-ice specific heat capacity,       !%dnt
   !           hcpct = 1880.0*917.0 = 1.72396e+6 (source:  fei chen, 1995)                !%dnt
   !           set bottom of sea-ice pack temperature: tbot(i, jj) = 271.16               !%dnt
   !           set a nominal universal value of glacial-ice specific heat capacity,       !%dnt
   !           hcpct = 2100.0*900.0 = 1.89000e+6 (source:  bob grumbine, 2005)            !%dnt
   !           tbot(i, jj) passed in as argument, value from global data set              !%dnt
               if (ice == 1) then                                                         !%dnt
   !  --- ...  sea-ice                                                                    !%dnt
                  hcpct = 1.72396e+6                                                      !%dnt
                  tbot(i, jj) = 271.16                                                    !%dnt
               else                                                                       !%dnt
   !  --- ...  glacial-ice                                                                !%dnt
                  hcpct = 1.89000e+6                                                      !%dnt
               end if                                                                     !%dnt
                                                                                          !%dnt
   !  --- ...  the input argument df2 is a universally constant value of sea-ice          !%dnt
   !           and glacial-ice thermal diffusivity, set in sflx as df2 = 2.2.             !%dnt
                                                                                          !%dnt
   !  --- ...  set ice pack depth.  use tbot(i, jj) as ice pack lower boundary            !%dnt
   !           temperature (that of unfrozen sea water at bottom of sea ice pack).        !%dnt
   !           assume ice pack is of n=nsoil layers spanning a uniform constant ice       !%dnt
   !           pack thickness as defined by zsoil(nsoil) in routine sflx.                 !%dnt
   !           if glacial-ice, set zbot_1 = -25 meters                                    !%dnt
               if (ice == 1) then                                                         !%dnt
   !  --- ...  sea-ice                                                                    !%dnt
                  zbot_1 = zsoil(nsoil)                                                   !%dnt
               else                                                                       !%dnt
   !  --- ...  glacial-ice                                                                !%dnt
                  zbot_1 = -25.0                                                          !%dnt
               end if                                                                     !%dnt
                                                                                          !%dnt
   !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer              !%dnt
               ddz_3 = 1.0/(-0.5*zsoil(2))                                                !%dnt
               ai(1) = 0.0                                                                !%dnt
               ci(1) = (df2*ddz_3)/(zsoil(1)*hcpct)                                       !%dnt
               bi(1) = -ci(1) + df2/(0.5*zsoil(1)*zsoil(1)*hcpct*zz1)                     !%dnt
                                                                                          !%dnt
   !  --- ...  calc the vertical soil temp gradient btwn the top and 2nd soil             !%dnt
   !           layers. recalc/adjust the soil heat flux.  use the gradient and            !%dnt
   !           flux to calc rhsts for the top soil layer.                                 !%dnt
                                                                                          !%dnt
               dtsdz = (stc(i, 1, jj) - stc(i, 2, jj))/(-0.5*zsoil(2))                    !%dnt
               ssoil_1 = df2*(stc(i, 1, jj) - yy)/(0.5*zsoil(1)*zz1)                      !%dnt
               rhsts(1) = (df2*dtsdz - ssoil_1)/(zsoil(1)*hcpct)                          !%dnt
                                                                                          !%dnt
   !  --- ...  initialize ddz2_2                                                          !%dnt
                                                                                          !%dnt
               ddz2_2 = 0.0                                                               !%dnt
                                                                                          !%dnt
   !  --- ...  loop thru the remaining soil layers, repeating the above process           !%dnt
               !$acc loop seq                                                             !%dnt
               do k = 2, nsoil                                                            !%dnt
                                                                                          !%dnt
                  if (k /= nsoil) then                                                    !%dnt
                                                                                          !%dnt
   !  --- ...  calc the vertical soil temp gradient thru this layer.                      !%dnt
                     denom_5 = 0.5*(zsoil(k - 1) - zsoil(k + 1))                          !%dnt
                     dtsdz2 = (stc(i, k, jj) - stc(i, k + 1, jj))/denom_5                 !%dnt
                                                                                          !%dnt
   !  --- ...  calc the matrix coef, ci, after calc'ng its partial product.               !%dnt
                                                                                          !%dnt
                     ddz2_2 = 2.0/(zsoil(k - 1) - zsoil(k + 1))                           !%dnt
                     ci(k) = -df2*ddz2_2/((zsoil(k - 1) - zsoil(k))*hcpct)                !%dnt
                                                                                          !%dnt
                  else                                                                    !%dnt
                                                                                          !%dnt
   !  --- ...  calc the vertical soil temp gradient thru the lowest layer.                !%dnt
                     dtsdz2 = (stc(i, k, jj) - tbot(i, jj)) &                             !%dnt
                              /(0.5*(zsoil(k - 1) + zsoil(k)) - zbot_1)                   !%dnt
                                                                                          !%dnt
   !  --- ...  set matrix coef, ci to zero.                                               !%dnt
                                                                                          !%dnt
                     ci(k) = 0.0                                                          !%dnt
                                                                                          !%dnt
                  end if   ! end if_k_block                                               !%dnt
                                                                                          !%dnt
   !  --- ...  calc rhsts for this layer after calc'ng a partial product.                 !%dnt
                  denom_5 = (zsoil(k) - zsoil(k - 1))*hcpct                               !%dnt
                  rhsts(k) = (df2*dtsdz2 - df2*dtsdz)/denom_5                             !%dnt
                                                                                          !%dnt
   !  --- ...  calc matrix coefs, ai, and bi for this layer.                              !%dnt
                                                                                          !%dnt
                  ai(k) = -df2*ddz_3/((zsoil(k - 1) - zsoil(k))*hcpct)                    !%dnt
                  bi(k) = -(ai(k) + ci(k))                                                !%dnt
                                                                                          !%dnt
   !  --- ...  reset values of dtsdz and ddz_3 for loop to next soil lyr.                 !%dnt
                                                                                          !%dnt
                  dtsdz = dtsdz2                                                          !%dnt
                  ddz_3 = ddz2_2                                                          !%dnt
                                                                                          !%dnt
               end do   ! end do_k_loop                                                   !%dnt
               !end call hrtice_gpu                                                       !%dnt
                                                                                          !%dn
               !call hstep_gpu &                                                          !%dnu
               !   !  ---  inputs: &                                                      !%dnu
               !   (nsoil, stc, dt, &                                                     !%dnu
               !    !  ---  input/outputs: &                                              !%dnu
               !    rhsts, ai, bi, ci, &                                                  !%dnu
               !    !  ---  outputs: &                                                    !%dnu
               !    stcf, &                                                               !%dnu
               !    !  ---  dummys: &                                                     !%dnu
               !    ciin, rhstsin &                                                       !%dnu
               !    )                                                                     !%dnu
   !  --- ...  create finite difference values for use in rosr12 routine                  !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  rhsts(k) = rhsts(k)*dt                                                  !%dnu
                  ai(k) = ai(k)*dt                                                        !%dnu
                  bi(k) = 1.0 + bi(k)*dt                                                  !%dnu
                  ci(k) = ci(k)*dt                                                        !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
   !  --- ...  copy values for input variables before call to rosr12                      !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  rhstsin(k) = rhsts(k)                                                   !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsold                                                            !%dnu
                  ciin(k) = ci(k)                                                         !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
   !  --- ...  solve the tri-diagonal matrix equation                                     !%dnu
                                                                                          !%dnu
               !call rosr12_gpu &                                                         !%dnuv
               !   !  ---  inputs: &                                                      !%dnuv
               !   (nsoil, ai, bi, rhstsin, &                                             !%dnuv
               !    !  ---  input/outputs: &                                              !%dnuv
               !    ciin, &                                                               !%dnuv
               !    !  ---  outputs: &                                                    !%dnuv
               !    ci, rhsts &                                                           !%dnuv
               !    )                                                                     !%dnuv
   !  --- ...  initialize eqn coef ciin for the lowest soil layer                         !%dnuv
                                                                                          !%dnuv
               ciin(nsoil) = 0.0                                                          !%dnuv
                                                                                          !%dnuv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%dnuv
               ci(1) = -ciin(1)/bi(1)                                                     !%dnuv
               rhsts(1) = rhstsin(1)/bi(1)                                                !%dnuv
                                                                                          !%dnuv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%dnuv
               !$acc loop seq                                                             !%dnuv
               do k = 2, nsoil                                                            !%dnuv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%dnuv
                  rhsts(k) = (rhstsin(k) - ai(k)*rhsts(k - 1)) &                          !%dnuv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%dnuv
               end do                                                                     !%dnuv
                                                                                          !%dnuv
   !  --- ...  set ci to rhsts for lowest soil layer                                      !%dnuv
               ci(nsoil) = rhsts(nsoil)                                                   !%dnuv
                                                                                          !%dnuv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%dnuv
                                                                                          !%dnuv
               !$acc loop seq                                                             !%dnuv
               do k = 2, nsoil                                                            !%dnuv
                  kk = nsoil - k + 1                                                      !%dnuv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhsts(kk)                                  !%dnuv
               end do                                                                     !%dnuv
               !end call rosr12_gpu                                                       !%dnuv
                                                                                          !%dnu
   !  --- ...  calc/update the soil temps using matrix solution                           !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  stcf(k) = stc(i, k, jj) + ci(k)                                         !%dnu
               end do                                                                     !%dnu
               !end call hstep_gpu                                                        !%dnu
                                                                                          !%dn
            else                                                                          !%dn
                                                                                          !%dn
!  --- ...  land-mass case                                                                !%dn
                                                                                          !%dn
               !call hrt_gpu &                                                            !%dns
               !   !  ---  inputs: &                                                      !%dns
               !   (nsoil, stc, smc, smcmax, zsoil, yy, zz1, tbot, &                      !%dns
               !    zbot, psisat, dt, bexp, df2, quartz, csoil, vegtyp, ivegsrc,&         !%dns
               !    !  ---  input/outputs: &                                              !%dns
               !    sh2o, df1, &                                                          !%dns
               !    !  ---  outputs: &                                                    !%dns
               !    rhsts, ai, bi, ci &                                                   !%dns
               !    )                                                                     !%dns
               csoil_loc = csoil                                                          !%dns
               if (ivegsrc .ge. 1) then                                                   !%dns
   !urban                                                                                 !%dns
                  if (vegtyp(i, jj) == 13) then                                           !%dns
                     csoil_loc = 3.0e6                                                    !%dns
                  end if                                                                  !%dns
               end if                                                                     !%dns
                                                                                          !%dns
   !  --- ...  initialize logical for soil layer temperature averaging.                   !%dns
                                                                                          !%dns
               itavg = .true.                                                             !%dns
   !     itavg = .false.                                                                  !%dns
                                                                                          !%dns
   !  ===  begin section for top soil layer                                               !%dns
                                                                                          !%dns
   !  --- ...  calc the heat capacity of the top soil layer                               !%dns
               hcpct_1 = sh2o(i, 1, jj)*cph2o2 + (1.0 - smcmax(i, jj))*csoil_loc &        !%dns
                       + (smcmax(i, jj) - smc(i, 1, jj))*cp2 + (smc(i, 1, jj) - &         !%dns
                       sh2o(i, 1, jj))*cpice1                                             !%dns
                                                                                          !%dns
   !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer              !%dns
                                                                                          !%dns
               ddz_2 = 1.0/(-0.5*zsoil(2))                                                !%dns
               ai(1) = 0.0                                                                !%dns
               ci(1) = (df2*ddz_2)/(zsoil(1)*hcpct_1)                                     !%dns
               bi(1) = -ci(1) + df2/(0.5*zsoil(1)*zsoil(1)*hcpct_1*zz1)                   !%dns
                                                                                          !%dns
   !  --- ...  calculate the vertical soil temp gradient btwn the 1st and 2nd soil        !%dns
   !           layers.  then calculate the subsurface heat flux. use the temp             !%dns
   !           gradient and subsfc heat flux to calc "right-hand side tendency            !%dns
   !           terms", or "rhsts", for top soil layer.                                    !%dns
                                                                                          !%dns
               dtsdz_1 = (stc(i, 1, jj) - stc(i, 2, jj))/(-0.5*zsoil(2))                  !%dns
               ssoil_2 = df2*(stc(i, 1, jj) - yy)/(0.5*zsoil(1)*zz1)                      !%dns
               rhsts(1) = (df2*dtsdz_1 - ssoil_2)/(zsoil(1)*hcpct_1)                      !%dns
                                                                                          !%dns
   !  --- ...  next capture the vertical difference of the heat flux at top and           !%dns
   !           bottom of first soil layer for use in heat flux constraint applied to      !%dns
   !           potential soil freezing/thawing in routine snksrc.                         !%dns
                                                                                          !%dns
               qtot = ssoil_2 - df2*dtsdz_1                                               !%dns
                                                                                          !%dns
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%dns
   !           set temp "tsurf" at top of soil column (for use in freezing soil           !%dns
   !           physics later in subroutine snksrc).  if snowpack content is               !%dns
   !           zero, then tsurf expression below gives tsurf = skin temp.  if             !%dns
   !           snowpack is nonzero (hence argument zz1=1), then tsurf expression          !%dns
   !           below yields soil column top temperature under snowpack.  then             !%dns
   !           calculate temperature at bottom interface of 1st soil layer for use        !%dns
   !           later in subroutine snksrc                                                 !%dns
               if (itavg) then                                                            !%dns
                  tsurf = (yy + (zz1 - 1)*stc(i, 1, jj))/zz1                              !%dns
                                                                                          !%dns
                  !call tbnd_gpu &                                                        !%dnsz
                  !   !  ---  inputs: &                                                   !%dnsz
                  !   (stc(1), stc(2), zsoil, zbot, 1, nsoil, &                           !%dnsz
                  !    !  ---  outputs: &                                                 !%dnsz
                  !    tbk &                                                              !%dnsz
                  !    )                                                                  !%dnsz
      !  --- ...  use surface temperature on the top of the first layer                   !%dnsz
                  k = 1                                                                   !%dnsz
                  if (k == 1) then                                                        !%dnsz
                     zup = 0.0                                                            !%dnsz
                  else                                                                    !%dnsz
                     zup = zsoil(k - 1)                                                   !%dnsz
                  end if                                                                  !%dnsz
                                                                                          !%dnsz
      !  --- ...  use depth of the constant bottom temperature when interpolate           !%dnsz
      !           temperature into the last layer boundary                                !%dnsz
                                                                                          !%dnsz
                  if (k == nsoil) then                                                    !%dnsz
                     zb = 2.0*zbot - zsoil(k)                                             !%dnsz
                  else                                                                    !%dnsz
                     zb = zsoil(k + 1)                                                    !%dnsz
                  end if                                                                  !%dnsz
                                                                                          !%dnsz
      !  --- ...  linear interpolation between the average layer temperatures             !%dnsz
                  tbk = stc(i, k, jj) + (stc(i, k + 1, jj) - stc(i, k, jj))*(zup - &      !%dnsz
                        zsoil(k))/(zup - zb)                                              !%dnsz
                  !end call tbnd_gpu                                                      !%dnsz
                                                                                          !%dns
               end if                                                                     !%dns
                                                                                          !%dns
   !  --- ...  calculate frozen water content in 1st soil layer.                          !%dns
               sice_1 = smc(i, 1, jj) - sh2o(i, 1, jj)                                    !%dns
                                                                                          !%dns
   !  --- ...  if frozen water present or any of layer-1 mid-point or bounding            !%dns
   !           interface temperatures below freezing, then call snksrc to                 !%dns
   !           compute heat source/sink (and change in frozen water content)              !%dns
   !           due to possible soil water phase change                                    !%dns
                                                                                          !%dns
               if ((sice_1 > 0.0) .or. (tsurf < tfreez) .or. &                            !%dns
                   (stc(i, 1, jj) < tfreez) .or. (tbk < tfreez)) then                     !%dns
                  if (itavg) then                                                         !%dns
                                                                                          !%dns
                     !call tmpavg_gpu &                                                   !%dnsA
                     !   !  ---  inputs: &                                                !%dnsA
                     !   (tsurf, stc(1), tbk, zsoil, nsoil, 1, &                          !%dnsA
                     !    !  ---  outputs: &                                              !%dnsA
                     !    tavg &                                                          !%dnsA
                     !    )                                                               !%dnsA
                     k = 1                                                                !%dnsA
                     if (k == 1) then                                                     !%dnsA
                        dz = -zsoil(1)                                                    !%dnsA
                     else                                                                 !%dnsA
                        dz = zsoil(k - 1) - zsoil(k)                                      !%dnsA
                     end if                                                               !%dnsA
                     dzh = dz*0.5                                                         !%dnsA
                                                                                          !%dnsA
                     if (tsurf < tfreez) then                                             !%dnsA
                        if (stc(i, k, jj) < tfreez) then                                  !%dnsA
                           if (tbk < tfreez) then ! tsurf, stc(k), tbk < t0               !%dnsA
                              tavg = (tsurf + 2.0*stc(i, k, jj) + tbk)/4.0                !%dnsA
                           else ! tsurf & stc(k) < t0,  tbk >= t0                         !%dnsA
                              x0 = (tfreez - stc(i, k, jj))*dzh/(tbk - stc(i, k, jj))     !%dnsA
                              tavg = 0.5*(tsurf*dzh + stc(i, k, jj)*(dzh + x0) + &        !%dnsA
                                     tfreez*(2.*dzh - x0))/dz                             !%dnsA
                           end if                                                         !%dnsA
                        else                                                              !%dnsA
                           if (tbk < tfreez) then ! tsurf < t0, stc(k) >= t0, tbk < t0    !%dnsA
                              xup = (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)          !%dnsA
                              xdn = dzh - (tfreez - stc(i, k, jj))*dzh/(tbk - &           !%dnsA
                                    stc(i, k, jj))                                        !%dnsA
                              tavg = 0.5*(tsurf*xup + tfreez*(2.*dz - xup - xdn) + &      !%dnsA
                                     tbk*xdn)/dz                                          !%dnsA
                           else ! tsurf < t0, stc(k) >= t0, tbk >= t0                     !%dnsA
                              xup = (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)          !%dnsA
                              tavg = 0.5*(tsurf*xup + tfreez*(2.*dz - xup))/dz            !%dnsA
                           end if                                                         !%dnsA
                        end if                                                            !%dnsA
                                                                                          !%dnsA
                     else    ! if_tup_block                                               !%dnsA
                        if (stc(i, k, jj) < tfreez) then                                  !%dnsA
                           if (tbk < tfreez) then ! tsurf >= t0, stc(k) < t0, tbk < t0    !%dnsA
                              xup = dzh - (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)    !%dnsA
                              tavg = 0.5*(tfreez*(dz - xup) + stc(i, k, jj)* &            !%dnsA
                                     (dzh + xup) + tbk*dzh)/dz                            !%dnsA
                                                                                          !%dnsA
                           else ! tsurf >= t0, stc(k) < t0, tbk >= t0                     !%dnsA
                              xup = dzh - (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)    !%dnsA
                              xdn = (tfreez - stc(i, k, jj))*dzh/(tbk - stc(i, k, jj))    !%dnsA
                              tavg = 0.5*(tfreez*(2.*dz - xup - xdn) + stc(i, k, jj)* &   !%dnsA
                                     (xup + xdn))/dz                                      !%dnsA
                           end if                                                         !%dnsA
                        else                                                              !%dnsA
                           if (tbk < tfreez) then ! tsurf >= t0, stc(k) >= t0, tbk < t0   !%dnsA
                              xdn = dzh - (tfreez - stc(i, k, jj))*dzh/ &                 !%dnsA
                                    (tbk - stc(i, k, jj))                                 !%dnsA
                              tavg = (tfreez*(dz - xdn) + 0.5*(tfreez + tbk)*xdn)/dz      !%dnsA
                           else ! tsurf >= t0, stc(k) >= t0, tbk >= t0                    !%dnsA
                              tavg = (tsurf + 2.0*stc(i, k, jj) + tbk)/4.0                !%dnsA
                           end if                                                         !%dnsA
                        end if                                                            !%dnsA
                                                                                          !%dnsA
                     end if   ! end if_tup_block                                          !%dnsA
                     !end call tmpavg_gpu                                                 !%dnsA
                                                                                          !%dns
                  else                                                                    !%dns
                     tavg = stc(i, 1, jj)                                                 !%dns
                                                                                          !%dns
                  end if   ! end if_itavg_block                                           !%dns
                                                                                          !%dns
                  !call snksrc_gpu &                                                      !%dnsw
                  !   !  ---  inputs: &                                                   !%dnsw
                  !   (nsoil, 1, tavg, smc(1), smcmax, psisat, bexp, dt, &                !%dnsw
                  !    qtot, zsoil, ivegsrc, vegtyp, &                                    !%dnsw
                  !    !  ---  input/outputs: &                                           !%dnsw
                  !    sh2o(1), df1, &                                                    !%dnsw
                  !    !  ---  outputs: &                                                 !%dnsw
                  !    tsnsr &                                                            !%dnsw
                  !    )                                                                  !%dnsw
                  k = 1                                                                   !%dnsw
                  if (ivegsrc .ge. 1) then                                                !%dnsw
                     if (vegtyp(i, jj) == 13) then                                        !%dnsw
                        df1 = 3.24                                                        !%dnsw
                     end if                                                               !%dnsw
                  end if                                                                  !%dnsw
      !                                                                                   !%dnsw
      !===> ...  begin here                                                               !%dnsw
      !                                                                                   !%dnsw
                  if (k == 1) then                                                        !%dnsw
                     dz_1 = -zsoil(1)                                                     !%dnsw
                  else                                                                    !%dnsw
                     dz_1 = zsoil(k - 1) - zsoil(k)                                       !%dnsw
                  end if                                                                  !%dnsw
                                                                                          !%dnsw
      !  --- ...  via function frh2o, compute potential or 'equilibrium' unfrozen         !%dnsw
      !           supercooled free water for given soil type and soil layer               !%dnsw
      !           temperature. function frh20 invokes eqn (17) from v. koren et al        !%dnsw
      !           (1999, jgr, vol. 104, pg 19573).  (aside:  latter eqn in journal in     !%dnsw
      !           centigrade units. routine frh2o use form of eqn in kelvin units.)       !%dnsw
                                                                                          !%dnsw
      !     free = frh2o( tavg,smc(k),sh2o(k),smcmax(i, jj),bexp,psisat )                 !%dnsw
                                                                                          !%dnsw
                  !call frh2o_gpu &                                                       !%dnswr
                  !   !  ---  inputs: &                                                   !%dnswr
                  !   (tavg, smc(k), sh2o(k), smcmax, bexp, psisat, &                     !%dnswr
                  !    !  ---  outputs: &                                                 !%dnswr
                  !    free &                                                             !%dnswr
                  !    )                                                                  !%dnswr
                                                                                          !%dnswr
      !                                                                                   !%dnswr
      !===> ...  begin here                                                               !%dnswr
      !                                                                                   !%dnswr
      !  --- ...  limits on parameter b: b < 5.5  (use parameter blim)                    !%dnswr
      !           simulations showed if b > 5.5 unfrozen water content is                 !%dnswr
      !           non-realistically high at very low temperatures.                        !%dnswr
                  bx = bexp                                                               !%dnswr
                  if (bexp > blim) bx = blim                                              !%dnswr
                                                                                          !%dnswr
      !  --- ...  initializing iterations counter and iterative solution flag.            !%dnswr
                                                                                          !%dnswr
                  nlog = 0                                                                !%dnswr
                  kcount = 0                                                              !%dnswr
                                                                                          !%dnswr
      !  --- ...  if temperature not significantly below freezing (t0),                   !%dnswr
      !           sh2o(k) = smc(k)                                                        !%dnswr
                  if (tavg > (tfreez - 1.e-3)) then                                       !%dnswr
                     free = smc(i, k, jj)                                                 !%dnswr
                                                                                          !%dnswr
                  else                                                                    !%dnswr
                     if (ck /= 0.0) then                                                  !%dnswr
                                                                                          !%dnswr
      !  --- ...  option 1: iterated solution for nonzero ck                              !%dnswr
      !                     in koren et al, jgr, 1999, eqn 17                             !%dnswr
                                                                                          !%dnswr
      !  --- ...  initial guess for swl (frozen content)                                  !%dnswr
                        swl = smc(i, k, jj) - sh2o(i, k, jj)                              !%dnswr
                                                                                          !%dnswr
      !  --- ...  keep within bounds.                                                     !%dnswr
                                                                                          !%dnswr
                        swl = max(min(swl, smc(i, k, jj) - 0.02), 0.0)                    !%dnswr
                                                                                          !%dnswr
      !  --- ...  start of iterations                                                     !%dnswr
                        do while ((nlog < 10) .and. (kcount == 0))                        !%dnswr
                           nlog = nlog + 1                                                !%dnswr
                                                                                          !%dnswr
                           df = dlog((psisat*gs2/lsubf)*((1.0 + ck*swl)**2.0) &           !%dnswr
                                     *(smcmax(i, jj)/(smc(i, k, jj) - swl))**bx) &        !%dnswr
                                     - dlog(-(tavg - tfreez)/tavg)                        !%dnswr
                                                                                          !%dnswr
                           denom_1 = 2.0*ck/(1.0 + ck*swl) + bx/(smc(i, k, jj) - swl)     !%dnswr
                           swlk = swl - df/denom_1                                        !%dnswr
                                                                                          !%dnswr
      !  --- ...  bounds useful for mathematical solution.                                !%dnswr
                                                                                          !%dnswr
                           swlk = max(min(swlk, smc(i, k, jj) - 0.02), 0.0)               !%dnswr
                                                                                          !%dnswr
      !  --- ...  mathematical solution bounds applied.                                   !%dnswr
                                                                                          !%dnswr
                           dswl = abs(swlk - swl)                                         !%dnswr
                           swl = swlk                                                     !%dnswr
                                                                                          !%dnswr
      !  --- ...  if more than 10 iterations, use explicit method (ck=0 approx.)          !%dnswr
      !           when dswl less or eq. error, no more iterations required.               !%dnswr
                           if (dswl <= error) then                                        !%dnswr
                              kcount = kcount + 1                                         !%dnswr
                           end if                                                         !%dnswr
                        end do   !  end do_while_loop                                     !%dnswr
                                                                                          !%dnswr
      !  --- ...  bounds applied within do-block are valid for physical solution.         !%dnswr
                        free = smc(i, k, jj) - swl                                        !%dnswr
                                                                                          !%dnswr
                     end if   ! end if_ck_block                                           !%dnswr
                                                                                          !%dnswr
      !  --- ...  option 2: explicit solution for flerchinger eq. i1.e. ck=0              !%dnswr
      !                     in koren et al., jgr, 1999, eqn 17                            !%dnswr
      !           apply physical bounds to flerchinger solution                           !%dnswr
                     if (kcount == 0) then                                                !%dnswr
                        fk = (((lsubf/(gs2*(-psisat))) &                                  !%dnswr
                               *((tavg - tfreez)/tavg))**(-1/bx))*smcmax(i, jj)           !%dnswr
                                                                                          !%dnswr
                        fk = max(fk, 0.02)                                                !%dnswr
                                                                                          !%dnswr
                        free = min(fk, smc(i, k, jj))                                     !%dnswr
                     end if                                                               !%dnswr
                                                                                          !%dnswr
                  end if   ! end if_tkelv_block                                           !%dnswr
                  !end call frh2o_gpu                                                     !%dnswr
                                                                                          !%dnsw
                                                                                          !%dnsw
      !  --- ...  in next block of code, invoke eqn 18 of v. koren et al (1999, jgr,      !%dnsw
      !           vol. 104, pg 19573.)  that is, first estimate the new amountof          !%dnsw
      !           liquid water, 'xh2o', implied by the sum of (1) the liquid water at     !%dnsw
      !           the begin of current time step, and (2) the freeze of thaw change       !%dnsw
      !           in liquid water implied by the heat flux 'qtot' passed in from          !%dnsw
      !           routine hrt. second, determine if xh2o needs to be bounded by           !%dnsw
      !           'free' (equil amt) or if 'free' needs to be bounded by xh2o.            !%dnsw
                                                                                          !%dnsw
                  xh2o = sh2o(i, k, jj) + qtot*dt/(dh2o*lsubf*dz_1)                       !%dnsw
                                                                                          !%dnsw
      !  --- ...  first, if freezing and remaining liquid less than lower bound, then     !%dnsw
      !           reduce extent of freezing, thereby letting some or all of heat flux     !%dnsw
      !           qtot cool the soil temp later in routine hrt.                           !%dnsw
                  if (xh2o < sh2o(i, k, jj) .and. xh2o < free) then                       !%dnsw
                     if (free > sh2o(i, k, jj)) then                                      !%dnsw
                        xh2o = sh2o(i, k, jj)                                             !%dnsw
                     else                                                                 !%dnsw
                        xh2o = free                                                       !%dnsw
                     end if                                                               !%dnsw
                  end if                                                                  !%dnsw
                                                                                          !%dnsw
      !  --- ...  second, if thawing and the increase in liquid water greater than        !%dnsw
      !           upper bound, then reduce extent of thaw, thereby letting some or        !%dnsw
      !           all of heat flux qtot warm the soil temp later in routine hrt.          !%dnsw
                                                                                          !%dnsw
                  if (xh2o > sh2o(i, k, jj) .and. xh2o > free) then                       !%dnsw
                     if (free < sh2o(i, k, jj)) then                                      !%dnsw
                        xh2o = sh2o(i, k, jj)                                             !%dnsw
                     else                                                                 !%dnsw
                        xh2o = free                                                       !%dnsw
                     end if                                                               !%dnsw
                  end if                                                                  !%dnsw
                                                                                          !%dnsw
                  xh2o = max(min(xh2o, smc(i, k, jj)), 0.0)                               !%dnsw
                                                                                          !%dnsw
      !  --- ...  calculate phase-change heat source/sink term for use in routine hrt     !%dnsw
      !           and update liquid water to reflcet final freeze/thaw increment.         !%dnsw
                                                                                          !%dnsw
                  tsnsr = -dh2o*lsubf*dz_1*(xh2o - sh2o(i, k, jj))/dt                     !%dnsw
                  sh2o(i, k, jj) = xh2o                                                   !%dnsw
                  !end call snksrc_gpu                                                    !%dnsw
                  rhsts(1) = rhsts(1) - tsnsr/(zsoil(1)*hcpct_1)                          !%dns
                                                                                          !%dns
               end if   ! end if_sice_block                                               !%dns
                                                                                          !%dns
   !  ===  this ends section for top soil layer.                                          !%dns
                                                                                          !%dns
   !  --- ...  initialize ddz2_1                                                          !%dns
                                                                                          !%dns
               ddz2_1 = 0.0                                                               !%dns
                                                                                          !%dns
   !  --- ...  loop thru the remaining soil layers, repeating the above process           !%dns
   !           (except subsfc or "ground" heat flux not repeated in lower layers)         !%dns
               df1k = df2                                                                 !%dns
                                                                                          !%dns
               !$acc loop seq                                                             !%dns
               do k = 2, nsoil                                                            !%dns
                                                                                          !%dns
   !  --- ...  calculate heat capacity for this soil layer.                               !%dns
                  hcpct_1 = sh2o(i, k, jj)*cph2o2 + (1.0 - smcmax(i, jj))*csoil_loc &     !%dns
                          + (smcmax(i, jj) - smc(i, k, jj))*cp2 + (smc(i, k, jj) &        !%dns
                          - sh2o(i, k, jj))*cpice1                                        !%dns
                                                                                          !%dns
                  if (k /= nsoil) then                                                    !%dns
                                                                                          !%dns
   !  --- ...  this section for layer 2 or greater, but not last layer.                   !%dns
   !           calculate thermal diffusivity for this layer.                              !%dns
                                                                                          !%dns
                     !call tdfcnd_gpu &                                                   !%dnsl
                     !   !  ---  inputs: &                                                !%dnsl
                     !   (smc(k), quartz, smcmax, sh2o(k), &                              !%dnsl
                     !    !  ---  outputs: &                                              !%dnsl
                     !    df1n &                                                          !%dnsl
                     !    )                                                               !%dnsl
         !  --- ...  if the soil has any moisture content compute a partial               !%dnsl
         !           sum/product otherwise use a constant value which works well with     !%dnsl
         !           most soils                                                           !%dnsl
                                                                                          !%dnsl
         !  --- ...  saturation ratio:                                                    !%dnsl
                     satratio = smc(i, k, jj)/smcmax(i, jj)                               !%dnsl
                                                                                          !%dnsl
         !  --- ...  parameters  w/(m.k)                                                  !%dnsl
                     thkice = 2.2                                                         !%dnsl
                     thkw = 0.57                                                          !%dnsl
                     thko = 2.0                                                           !%dnsl
         !     if (quartz <= 0.2) thko = 3.0                                              !%dnsl
                     thkqtz = 7.7                                                         !%dnsl
                                                                                          !%dnsl
         !  --- ...  solids' conductivity                                                 !%dnsl
                                                                                          !%dnsl
                     thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                       !%dnsl
                                                                                          !%dnsl
         !  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0.                 !%dnsl
         !           (100% frozen))                                                       !%dnsl
                                                                                          !%dnsl
                     xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)           !%dnsl
                                                                                          !%dnsl
         !  --- ...  unfrozen volume for saturation (porosity*xunfroz)                    !%dnsl
                                                                                          !%dnsl
                     xu = xunfroz*smcmax(i, jj)                                           !%dnsl
                                                                                          !%dnsl
         !  --- ...  saturated thermal conductivity                                       !%dnsl
                                                                                          !%dnsl
                     thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu) &     !%dnsl
                              *thkw**(xu)                                                 !%dnsl
                                                                                          !%dnsl
         !  --- ...  dry density in kg/m3                                                 !%dnsl
                                                                                          !%dnsl
                     gammd = (1.0 - smcmax(i, jj))*2700.0                                 !%dnsl
                                                                                          !%dnsl
         !  --- ...  dry thermal conductivity in w.m-1.k-1                                !%dnsl
                                                                                          !%dnsl
                     thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                 !%dnsl
                     if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then     ! frozen       !%dnsl
                        ake = satratio                                                    !%dnsl
                                                                                          !%dnsl
                     else                                  ! unfrozen                     !%dnsl
                                                                                          !%dnsl
         !  --- ...  range of validity for the kersten number (ake)                       !%dnsl
                        if (satratio > 0.1) then                                          !%dnsl
                                                                                          !%dnsl
         !  --- ...  kersten number (using "fine" formula, valid for soils containing     !%dnsl
         !           at least 5% of particles with diameter less than 2.e-6 meters.)      !%dnsl
         !           (for "coarse" formula, see peters-lidard et al., 1998).              !%dnsl
                                                                                          !%dnsl
                           ake = log10(satratio) + 1.0                                    !%dnsl
                                                                                          !%dnsl
                        else                                                              !%dnsl
                                                                                          !%dnsl
         !  --- ...  use k = kdry                                                         !%dnsl
                           ake = 0.0                                                      !%dnsl
                                                                                          !%dnsl
                        end if   ! end if_satratio_block                                  !%dnsl
                                                                                          !%dnsl
                     end if   ! end if_sh2o+0.0005_block                                  !%dnsl
                                                                                          !%dnsl
         !  --- ...  thermal conductivity                                                 !%dnsl
                     df1n = ake*(thksat - thkdry) + thkdry                                !%dnsl
                     !end call tdfcnd_gpu                                                 !%dnsl
   !urban                                                                                 !%dns
                     if (ivegsrc .ge. 1) then                                             !%dns
                        if (vegtyp(i, jj) == 13) then                                     !%dns
                           df1n = 3.24                                                    !%dns
                        end if                                                            !%dns
                     end if                                                               !%dns
                                                                                          !%dns
   !  --- ...  calc the vertical soil temp gradient thru this layer                       !%dns
                     denom_2 = 0.5*(zsoil(k - 1) - zsoil(k + 1))                          !%dns
                     dtsdz2_1 = (stc(i, k, jj) - stc(i, k + 1, jj))/denom_2               !%dns
                                                                                          !%dns
   !  --- ...  calc the matrix coef, ci, after calc'ng its partial product                !%dns
                                                                                          !%dns
                     ddz2_1 = 2.0/(zsoil(k - 1) - zsoil(k + 1))                           !%dns
                     ci(k) = -df1n*ddz2_1/((zsoil(k - 1) - zsoil(k))*hcpct_1)             !%dns
                                                                                          !%dns
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%dns
   !           calculate temp at bottom of layer.                                         !%dns
                     if (itavg) then                                                      !%dns
                                                                                          !%dns
                        !call tbnd_gpu &                                                  !%dnsz
                        !   !  ---  inputs: &                                             !%dnsz
                        !   (stc(k), stc(k + 1), zsoil, zbot, k, nsoil, &                 !%dnsz
                        !    !  ---  outputs: &                                           !%dnsz
                        !    tbk1 &                                                       !%dnsz
                        !    )                                                            !%dnsz
            !  --- ...  use surface temperature on the top of the first layer             !%dnsz
                        if (k == 1) then                                                  !%dnsz
                           zup = 0.0                                                      !%dnsz
                        else                                                              !%dnsz
                           zup = zsoil(k - 1)                                             !%dnsz
                        end if                                                            !%dnsz
                                                                                          !%dnsz
            !  --- ...  use depth of the constant bottom temperature when interpolate     !%dnsz
            !           temperature into the last layer boundary                          !%dnsz
                                                                                          !%dnsz
                        if (k == nsoil) then                                              !%dnsz
                           zb = 2.0*zbot - zsoil(k)                                       !%dnsz
                        else                                                              !%dnsz
                           zb = zsoil(k + 1)                                              !%dnsz
                        end if                                                            !%dnsz
                                                                                          !%dnsz
            !  --- ...  linear interpolation between the average layer temperatures       !%dnsz
                        tbk1 = stc(i, k, jj) + (stc(i, k + 1, jj) - stc(i, k, jj)) &      !%dnsz
                               *(zup - zsoil(k))/(zup - zb)                               !%dnsz
                        !end call tbnd_gpu                                                !%dnsz
                                                                                          !%dns
                     end if                                                               !%dns
                                                                                          !%dns
                  else                                                                    !%dns
                                                                                          !%dns
   !  --- ...  special case of bottom soil layer:  calculate thermal diffusivity          !%dns
   !           for bottom layer.                                                          !%dns
                                                                                          !%dns
                     !call tdfcnd_gpu &                                                   !%dnsl
                     !   !  ---  inputs: &                                                !%dnsl
                     !   (smc(k), quartz, smcmax, sh2o(k), &                              !%dnsl
                     !    !  ---  outputs: &                                              !%dnsl
                     !    df1n &                                                          !%dnsl
                     !    )                                                               !%dnsl
         !  --- ...  if the soil has any moisture content compute a partial               !%dnsl
         !           sum/product                                                          !%dnsl
         !           otherwise use a constant value which works well with most soils      !%dnsl
                                                                                          !%dnsl
         !  --- ...  saturation ratio:                                                    !%dnsl
                     satratio = smc(i, k, jj)/smcmax(i, jj)                               !%dnsl
                                                                                          !%dnsl
         !  --- ...  parameters  w/(m.k)                                                  !%dnsl
                     thkice = 2.2                                                         !%dnsl
                     thkw = 0.57                                                          !%dnsl
                     thko = 2.0                                                           !%dnsl
         !     if (quartz <= 0.2) thko = 3.0                                              !%dnsl
                     thkqtz = 7.7                                                         !%dnsl
                                                                                          !%dnsl
         !  --- ...  solids' conductivity                                                 !%dnsl
                                                                                          !%dnsl
                     thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                       !%dnsl
                                                                                          !%dnsl
         !  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0.                 !%dnsl
         !           (100% frozen))                                                       !%dnsl
                                                                                          !%dnsl
                     xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)           !%dnsl
                                                                                          !%dnsl
         !  --- ...  unfrozen volume for saturation (porosity*xunfroz)                    !%dnsl
                                                                                          !%dnsl
                     xu = xunfroz*smcmax(i, jj)                                           !%dnsl
                                                                                          !%dnsl
         !  --- ...  saturated thermal conductivity                                       !%dnsl
                                                                                          !%dnsl
                     thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu)&      !%dnsl
                              *thkw**(xu)                                                 !%dnsl
                                                                                          !%dnsl
         !  --- ...  dry density in kg/m3                                                 !%dnsl
                                                                                          !%dnsl
                     gammd = (1.0 - smcmax(i, jj))*2700.0                                 !%dnsl
                                                                                          !%dnsl
         !  --- ...  dry thermal conductivity in w.m-1.k-1                                !%dnsl
                                                                                          !%dnsl
                     thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                 !%dnsl
                     if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then      ! frozen      !%dnsl
                        ake = satratio                                                    !%dnsl
                                                                                          !%dnsl
                     else                                  ! unfrozen                     !%dnsl
                                                                                          !%dnsl
         !  --- ...  range of validity for the kersten number (ake)                       !%dnsl
                        if (satratio > 0.1) then                                          !%dnsl
                                                                                          !%dnsl
         !  --- ...  kersten number (using "fine" formula, valid for soils containing     !%dnsl
         !           at least 5% of particles with diameter less than 2.e-6 meters.)      !%dnsl
         !           (for "coarse" formula, see peters-lidard et al., 1998).              !%dnsl
                                                                                          !%dnsl
                           ake = log10(satratio) + 1.0                                    !%dnsl
                                                                                          !%dnsl
                        else                                                              !%dnsl
                                                                                          !%dnsl
         !  --- ...  use k = kdry                                                         !%dnsl
                           ake = 0.0                                                      !%dnsl
                                                                                          !%dnsl
                        end if   ! end if_satratio_block                                  !%dnsl
                                                                                          !%dnsl
                     end if   ! end if_sh2o+0.0005_block                                  !%dnsl
                                                                                          !%dnsl
         !  --- ...  thermal conductivity                                                 !%dnsl
                     df1n = ake*(thksat - thkdry) + thkdry                                !%dnsl
                     !end call tdfcnd_gpu                                                 !%dnsl
   !urban                                                                                 !%dns
                     if (ivegsrc .ge. 1) then                                             !%dns
                        if (vegtyp(i, jj) == 13) then                                     !%dns
                           df1n = 3.24                                                    !%dns
                        end if                                                            !%dns
                     end if                                                               !%dns
                                                                                          !%dns
   !  --- ...  calc the vertical soil temp gradient thru bottom layer.                    !%dns
                     denom_2 = 0.5*(zsoil(k - 1) + zsoil(k)) - zbot                       !%dns
                     dtsdz2_1 = (stc(i, k, jj) - tbot(i, jj))/denom_2                     !%dns
                                                                                          !%dns
   !  --- ...  set matrix coef, ci to zero if bottom layer.                               !%dns
                                                                                          !%dns
                     ci(k) = 0.0                                                          !%dns
                                                                                          !%dns
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%dns
   !           calculate temp at bottom of last layer.                                    !%dns
                     if (itavg) then                                                      !%dns
                                                                                          !%dns
                        !call tbnd_gpu &                                                  !%dnsz
                        !   !  ---  inputs: &                                             !%dnsz
                        !   (stc(k), tbot, zsoil, zbot, k, nsoil, &                       !%dnsz
                        !    !  ---  outputs:                                             !%dnsz
                        !    tbk1 &                                                       !%dnsz
                        !    )                                                            !%dnsz
            !  --- ...  use surface temperature on the top of the first layer             !%dnsz
                        if (k == 1) then                                                  !%dnsz
                           zup = 0.0                                                      !%dnsz
                        else                                                              !%dnsz
                           zup = zsoil(k - 1)                                             !%dnsz
                        end if                                                            !%dnsz
                                                                                          !%dnsz
            !  --- ...  use depth of the constant bottom temperature when interpolate     !%dnsz
            !           temperature into the last layer boundary                          !%dnsz
                                                                                          !%dnsz
                        if (k == nsoil) then                                              !%dnsz
                           zb = 2.0*zbot - zsoil(k)                                       !%dnsz
                        else                                                              !%dnsz
                           zb = zsoil(k + 1)                                              !%dnsz
                        end if                                                            !%dnsz
                                                                                          !%dnsz
            !  --- ...  linear interpolation between the average layer temperatures       !%dnsz
                        tbk1 = stc(i, k, jj) + (tbot(i, jj) - stc(i, k, jj)) &            !%dnsz
                               *(zup - zsoil(k))/(zup - zb)                               !%dnsz
                        !end call tbnd_gpu                                                !%dnsz
                                                                                          !%dns
                     end if                                                               !%dns
                                                                                          !%dns
                  end if   ! end if_k_block                                               !%dns
                                                                                          !%dns
   !  --- ...  calculate rhsts for this layer after calc'ng a partial product.            !%dns
                  denom_2 = (zsoil(k) - zsoil(k - 1))*hcpct_1                             !%dns
                  rhsts(k) = (df1n*dtsdz2_1 - df1k*dtsdz_1)/denom_2                       !%dns
                                                                                          !%dns
                  qtot = -1.0*denom_2*rhsts(k)                                            !%dns
                  sice_1 = smc(i, k, jj) - sh2o(i, k, jj)                                 !%dns
                  if ((sice_1 > 0.0) .or. (tbk < tfreez) .or. &                           !%dns
                      (stc(i, k, jj) < tfreez) .or. (tbk1 < tfreez)) then                 !%dns
                     if (itavg) then                                                      !%dns
                                                                                          !%dns
                        !call tmpavg_gpu &                                                !%dnsA
                        !   !  ---  inputs: &                                             !%dnsA
                        !   (tbk, stc(k), tbk1, zsoil, nsoil, k, &                        !%dnsA
                        !    !  ---  outputs: &                                           !%dnsA
                        !    tavg &                                                       !%dnsA
                        !    )                                                            !%dnsA
                        if (k == 1) then                                                  !%dnsA
                           dz = -zsoil(1)                                                 !%dnsA
                        else                                                              !%dnsA
                           dz = zsoil(k - 1) - zsoil(k)                                   !%dnsA
                        end if                                                            !%dnsA
                        dzh = dz*0.5                                                      !%dnsA
                                                                                          !%dnsA
                        if (tbk < tfreez) then                                            !%dnsA
                           if (stc(i, k, jj) < tfreez) then                               !%dnsA
                              if (tbk1 < tfreez) then ! tbk, stc(k), tbk1 < t0            !%dnsA
                                 tavg = (tbk + 2.0*stc(i, k, jj) + tbk1)/4.0              !%dnsA
                              else ! tbk & stc(k) < t0,  tbk1 >= t0                       !%dnsA
                                 x0 = (tfreez - stc(i, k, jj))*dzh &                      !%dnsA
                                      /(tbk1 - stc(i, k, jj))                             !%dnsA
                                 tavg = 0.5*(tbk*dzh + stc(i, k, jj)*(dzh + x0) &         !%dnsA
                                        + tfreez*(2.*dzh - x0))/dz                        !%dnsA
                              end if                                                      !%dnsA
                           else                                                           !%dnsA
                              if (tbk1 < tfreez) then  ! tbk < t0, stc(k)                 !%dnsA
                                                       ! >= t0, tbk1 < t0                 !%dnsA
                                 xup = (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)           !%dnsA
                                 xdn = dzh - (tfreez - stc(i, k, jj))*dzh &               !%dnsA
                                       /(tbk1 - stc(i, k, jj))                            !%dnsA
                                 tavg = 0.5*(tbk*xup + tfreez &                           !%dnsA
                                        *(2.*dz - xup - xdn) + tbk1*xdn)/dz               !%dnsA
                              else ! tbk < t0, stc(k) >= t0, tbk1 >= t0                   !%dnsA
                                 xup = (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)           !%dnsA
                                 tavg = 0.5*(tbk*xup + tfreez*(2.*dz - xup))/dz           !%dnsA
                              end if                                                      !%dnsA
                           end if                                                         !%dnsA
                                                                                          !%dnsA
                        else    ! if_tup_block                                            !%dnsA
                           if (stc(i, k, jj) < tfreez) then                               !%dnsA
                              if (tbk1 < tfreez) then ! tbk >= t0,                        !%dnsA
                                                      ! stc(k) < t0, tbk1 < t0            !%dnsA
                                 xup = dzh - (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)     !%dnsA
                                 tavg = 0.5*(tfreez*(dz - xup) + stc(i, k, jj) &          !%dnsA
                                        *(dzh + xup) + tbk1*dzh)/dz                       !%dnsA
                                                                                          !%dnsA
                              else ! tbk >= t0, stc(k) < t0, tbk1 >= t0                   !%dnsA
                                 xup = dzh - (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)     !%dnsA
                                 xdn = (tfreez - stc(i, k, jj)) &                         !%dnsA
                                       *dzh/(tbk1 - stc(i, k, jj))                        !%dnsA
                                 tavg = 0.5*(tfreez*(2.*dz - xup - xdn) &                 !%dnsA
                                        + stc(i, k, jj)*(xup + xdn))/dz                   !%dnsA
                              end if                                                      !%dnsA
                           else                                                           !%dnsA
                              if (tbk1 < tfreez) then  ! tbk >= t0, stc(k) >= t0,         !%dnsA
                                                       ! tbk1 < t0                        !%dnsA
                                 xdn = dzh - (tfreez - stc(i, k, jj))*dzh &               !%dnsA
                                       /(tbk1 - stc(i, k, jj))                            !%dnsA
                                 tavg = (tfreez*(dz - xdn) &                              !%dnsA
                                        + 0.5*(tfreez + tbk1)*xdn)/dz                     !%dnsA
                              else ! tbk >= t0, stc(k) >= t0, tbk1 >= t0                  !%dnsA
                                 tavg = (tbk + 2.0*stc(i, k, jj) + tbk1)/4.0              !%dnsA
                              end if                                                      !%dnsA
                           end if                                                         !%dnsA
                                                                                          !%dnsA
                        end if   ! end if_tup_block                                       !%dnsA
                        !end call tmpavg_gpu                                              !%dnsA
                                                                                          !%dns
                     else                                                                 !%dns
                        tavg = stc(i, k, jj)                                              !%dns
                     end if                                                               !%dns
                                                                                          !%dns
                     !call snksrc_gpu &                                                   !%dnsw
                     !   !  ---  inputs: &                                                !%dnsw
                     !   (nsoil, k, tavg, smc(k), smcmax, psisat, bexp, dt, &             !%dnsw
                     !    qtot, zsoil, ivegsrc, vegtyp, &                                 !%dnsw
                     !    !  ---  input/outputs: &                                        !%dnsw
                     !    sh2o(k), df1, &                                                 !%dnsw
                     !    !  ---  outputs: &                                              !%dnsw
                     !    tsnsr &                                                         !%dnsw
                     !    )                                                               !%dnsw
                     if (ivegsrc .ge. 1) then                                             !%dnsw
                        if (vegtyp(i, jj) == 13) then                                     !%dnsw
                           df1 = 3.24                                                     !%dnsw
                        end if                                                            !%dnsw
                     end if                                                               !%dnsw
         !                                                                                !%dnsw
         !===> ...  begin here                                                            !%dnsw
         !                                                                                !%dnsw
                     if (k == 1) then                                                     !%dnsw
                        dz_1 = -zsoil(1)                                                  !%dnsw
                     else                                                                 !%dnsw
                        dz_1 = zsoil(k - 1) - zsoil(k)                                    !%dnsw
                     end if                                                               !%dnsw
                                                                                          !%dnsw
         !  --- ...  via function frh2o, compute potential or 'equilibrium' unfrozen      !%dnsw
         !           supercooled free water for given soil type and soil layer            !%dnsw
         !           temperature. function frh20 invokes eqn (17) from v. koren et al     !%dnsw
         !           (1999, jgr, vol. 104, pg 19573).  (aside:  latter eqn in journal     !%dnsw
         !           in centigrade units. routine frh2o use form of eqn in kelvin         !%dnsw
         !           units.)                                                              !%dnsw
                                                                                          !%dnsw
         !     free = frh2o( tavg,smc(k),sh2o(k),smcmax(i, jj),bexp,psisat )              !%dnsw
                                                                                          !%dnsw
                     !call frh2o_gpu &                                                    !%dnswr
                     !   !  ---  inputs: &                                                !%dnswr
                     !   (tavg, smc(k), sh2o(k), smcmax, bexp, psisat, &                  !%dnswr
                     !    !  ---  outputs: &                                              !%dnswr
                     !    free &                                                          !%dnswr
                     !    )                                                               !%dnswr
                                                                                          !%dnswr
         !                                                                                !%dnswr
         !===> ...  begin here                                                            !%dnswr
         !                                                                                !%dnswr
         !  --- ...  limits on parameter b: b < 5.5  (use parameter blim)                 !%dnswr
         !           simulations showed if b > 5.5 unfrozen water content is              !%dnswr
         !           non-realistically high at very low temperatures.                     !%dnswr
                     bx = bexp                                                            !%dnswr
                     if (bexp > blim) bx = blim                                           !%dnswr
                                                                                          !%dnswr
         !  --- ...  initializing iterations counter and iterative solution flag.         !%dnswr
                                                                                          !%dnswr
                     nlog = 0                                                             !%dnswr
                     kcount = 0                                                           !%dnswr
                                                                                          !%dnswr
         !  --- ...  if temperature not significantly below freezing (t0),                !%dnswr
         !           sh2o(k) = smc(k)                                                     !%dnswr
                     if (tavg > (tfreez - 1.e-3)) then                                    !%dnswr
                        free = smc(i, k, jj)                                              !%dnswr
                                                                                          !%dnswr
                     else                                                                 !%dnswr
                        if (ck /= 0.0) then                                               !%dnswr
                                                                                          !%dnswr
         !  --- ...  option 1: iterated solution for nonzero ck                           !%dnswr
         !                     in koren et al, jgr, 1999, eqn 17                          !%dnswr
                                                                                          !%dnswr
         !  --- ...  initial guess for swl (frozen content)                               !%dnswr
                           swl = smc(i, k, jj) - sh2o(i, k, jj)                           !%dnswr
                                                                                          !%dnswr
         !  --- ...  keep within bounds.                                                  !%dnswr
                                                                                          !%dnswr
                           swl = max(min(swl, smc(i, k, jj) - 0.02), 0.0)                 !%dnswr
                                                                                          !%dnswr
         !  --- ...  start of iterations                                                  !%dnswr
                           do while ((nlog < 10) .and. (kcount == 0))                     !%dnswr
                              nlog = nlog + 1                                             !%dnswr
                                                                                          !%dnswr
                              df = dlog((psisat*gs2/lsubf)*((1.0 + ck*swl)**2.0) &        !%dnswr
                                        *(smcmax(i, jj)/(smc(i, k, jj) - swl))**bx) &     !%dnswr
                                        - dlog(-(tavg - tfreez)/tavg)                     !%dnswr
                                                                                          !%dnswr
                              denom_1 = 2.0*ck/(1.0 + ck*swl) + bx/(smc(i, k, jj) &       !%dnswr
                                        - swl)                                            !%dnswr
                              swlk = swl - df/denom_1                                     !%dnswr
                                                                                          !%dnswr
         !  --- ...  bounds useful for mathematical solution.                             !%dnswr
                                                                                          !%dnswr
                              swlk = max(min(swlk, smc(i, k, jj) - 0.02), 0.0)            !%dnswr
                                                                                          !%dnswr
         !  --- ...  mathematical solution bounds applied.                                !%dnswr
                                                                                          !%dnswr
                              dswl = abs(swlk - swl)                                      !%dnswr
                              swl = swlk                                                  !%dnswr
                                                                                          !%dnswr
         !  --- ...  if more than 10 iterations, use explicit method (ck=0 approx.)       !%dnswr
         !           when dswl less or eq. error, no more iterations required.            !%dnswr
                              if (dswl <= error) then                                     !%dnswr
                                 kcount = kcount + 1                                      !%dnswr
                              end if                                                      !%dnswr
                           end do   !  end do_while_loop                                  !%dnswr
                                                                                          !%dnswr
         !  --- ...  bounds applied within do-block are valid for physical solution.      !%dnswr
                           free = smc(i, k, jj) - swl                                     !%dnswr
                                                                                          !%dnswr
                        end if   ! end if_ck_block                                        !%dnswr
                                                                                          !%dnswr
         !  --- ...  option 2: explicit solution for flerchinger eq. i1.e. ck=0           !%dnswr
         !                     in koren et al., jgr, 1999, eqn 17                         !%dnswr
         !           apply physical bounds to flerchinger solution                        !%dnswr
                        if (kcount == 0) then                                             !%dnswr
                           fk = (((lsubf/(gs2*(-psisat))) &                               !%dnswr
                                  *((tavg - tfreez)/tavg))**(-1/bx))*smcmax(i, jj)        !%dnswr
                                                                                          !%dnswr
                           fk = max(fk, 0.02)                                             !%dnswr
                                                                                          !%dnswr
                           free = min(fk, smc(i, k, jj))                                  !%dnswr
                        end if                                                            !%dnswr
                                                                                          !%dnswr
                     end if   ! end if_tkelv_block                                        !%dnswr
                     !end call frh2o_gpu                                                  !%dnswr
                                                                                          !%dnsw
                                                                                          !%dnsw
         !  --- ...  in next block of code, invoke eqn 18 of v. koren et al (1999,        !%dnsw
         !           jgr, vol. 104, pg 19573.)  that is, first estimate the new           !%dnsw
         !           amountof liquid water, 'xh2o', implied by the sum of (1) the         !%dnsw
         !           liquid water at the begin of current time step, and (2) the          !%dnsw
         !           freeze of thaw change in liquidwater implied by the heat flux        !%dnsw
         !           'qtot' passed in from routine hrt.second, determine if xh2o          !%dnsw
         !           needs to be bounded by 'free' (equil amt) orif 'free' needs to       !%dnsw
         !           be bounded by xh2o.                                                  !%dnsw
                                                                                          !%dnsw
                     xh2o = sh2o(i, k, jj) + qtot*dt/(dh2o*lsubf*dz_1)                    !%dnsw
                                                                                          !%dnsw
         !  --- ...  first, if freezing and remaining liquid less than lower bound,       !%dnsw
         !           then reduce extent of freezing, thereby letting some or all of       !%dnsw
         !           heat flux qtot cool the soil temp later in routine hrt.              !%dnsw
                     if (xh2o < sh2o(i, k, jj) .and. xh2o < free) then                    !%dnsw
                        if (free > sh2o(i, k, jj)) then                                   !%dnsw
                           xh2o = sh2o(i, k, jj)                                          !%dnsw
                        else                                                              !%dnsw
                           xh2o = free                                                    !%dnsw
                        end if                                                            !%dnsw
                     end if                                                               !%dnsw
                                                                                          !%dnsw
         !  --- ...  second, if thawing and the increase in liquid water greater than     !%dnsw
         !           upper bound, then reduce extent of thaw, thereby letting some or     !%dnsw
         !           all of heat flux qtot warm the soil temp later in routine hrt.       !%dnsw
                                                                                          !%dnsw
                     if (xh2o > sh2o(i, k, jj) .and. xh2o > free) then                    !%dnsw
                        if (free < sh2o(i, k, jj)) then                                   !%dnsw
                           xh2o = sh2o(i, k, jj)                                          !%dnsw
                        else                                                              !%dnsw
                           xh2o = free                                                    !%dnsw
                        end if                                                            !%dnsw
                     end if                                                               !%dnsw
                                                                                          !%dnsw
                     xh2o = max(min(xh2o, smc(i, k, jj)), 0.0)                            !%dnsw
                                                                                          !%dnsw
         !  --- ...  calculate phase-change heat source/sink term for use in routine      !%dnsw
         !           hrt and update liquid water to reflcet final freeze/thaw             !%dnsw
         !           increment.                                                           !%dnsw
                                                                                          !%dnsw
                     tsnsr = -dh2o*lsubf*dz_1*(xh2o - sh2o(i, k, jj))/dt                  !%dnsw
                     sh2o(i, k, jj) = xh2o                                                !%dnsw
                     !end call snksrc_gpu                                                 !%dnsw
                     rhsts(k) = rhsts(k) - tsnsr/denom_2                                  !%dns
                  end if                                                                  !%dns
                                                                                          !%dns
   !  --- ...  calc matrix coefs, ai, and bi for this layer.                              !%dns
                  ai(k) = -df2*ddz_2/((zsoil(k - 1) - zsoil(k))*hcpct_1)                  !%dns
                  bi(k) = -(ai(k) + ci(k))                                                !%dns
                                                                                          !%dns
   !  --- ...  reset values of df1, dtsdz_1, ddz_2, and tbk for loop to next soil         !%dns
   !           layer.                                                                     !%dns
                                                                                          !%dns
                  tbk = tbk1                                                              !%dns
                  df1k = df1n                                                             !%dns
                  dtsdz_1 = dtsdz2_1                                                      !%dns
                  ddz_2 = ddz2_1                                                          !%dns
                                                                                          !%dns
               end do   ! end do_k_loop                                                   !%dns
               !end call hrt_gpu                                                          !%dns
                                                                                          !%dn
               !call hstep_gpu &                                                          !%dnu
               !   !  ---  inputs: &                                                      !%dnu
               !   (nsoil, stc, dt, &                                                     !%dnu
               !    !  ---  input/outputs: &                                              !%dnu
               !    rhsts, ai, bi, ci, &                                                  !%dnu
               !    !  ---  outputs: &                                                    !%dnu
               !    stcf, &                                                               !%dnu
               !    !  ---  dummys: &                                                     !%dnu
               !    ciin, rhstsin &                                                       !%dnu
               !    )                                                                     !%dnu
   !  --- ...  create finite difference values for use in rosr12 routine                  !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  rhsts(k) = rhsts(k)*dt                                                  !%dnu
                  ai(k) = ai(k)*dt                                                        !%dnu
                  bi(k) = 1.0 + bi(k)*dt                                                  !%dnu
                  ci(k) = ci(k)*dt                                                        !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
   !  --- ...  copy values for input variables before call to rosr12                      !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  rhstsin(k) = rhsts(k)                                                   !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsold                                                            !%dnu
                  ciin(k) = ci(k)                                                         !%dnu
               end do                                                                     !%dnu
                                                                                          !%dnu
   !  --- ...  solve the tri-diagonal matrix equation                                     !%dnu
                                                                                          !%dnu
               !call rosr12_gpu &                                                         !%dnuv
               !   !  ---  inputs: &                                                      !%dnuv
               !   (nsoil, ai, bi, rhstsin, &                                             !%dnuv
               !    !  ---  input/outputs: &                                              !%dnuv
               !    ciin, &                                                               !%dnuv
               !    !  ---  outputs: &                                                    !%dnuv
               !    ci, rhsts &                                                           !%dnuv
               !    )                                                                     !%dnuv
   !  --- ...  initialize eqn coef ciin for the lowest soil layer                         !%dnuv
                                                                                          !%dnuv
               ciin(nsoil) = 0.0                                                          !%dnuv
                                                                                          !%dnuv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%dnuv
               ci(1) = -ciin(1)/bi(1)                                                     !%dnuv
               rhsts(1) = rhstsin(1)/bi(1)                                                !%dnuv
                                                                                          !%dnuv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%dnuv
               !$acc loop seq                                                             !%dnuv
               do k = 2, nsoil                                                            !%dnuv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%dnuv
                  rhsts(k) = (rhstsin(k) - ai(k)*rhsts(k - 1)) &                          !%dnuv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%dnuv
               end do                                                                     !%dnuv
                                                                                          !%dnuv
   !  --- ...  set ci to rhsts for lowest soil layer                                      !%dnuv
               ci(nsoil) = rhsts(nsoil)                                                   !%dnuv
                                                                                          !%dnuv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%dnuv
                                                                                          !%dnuv
               !$acc loop seq                                                             !%dnuv
               do k = 2, nsoil                                                            !%dnuv
                  kk = nsoil - k + 1                                                      !%dnuv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhsts(kk)                                  !%dnuv
               end do                                                                     !%dnuv
               !end call rosr12_gpu                                                       !%dnuv
                                                                                          !%dnu
   !  --- ...  calc/update the soil temps using matrix solution                           !%dnu
                                                                                          !%dnu
               !$acc loop seq                                                             !%dnu
               do k = 1, nsoil                                                            !%dnu
                  stcf(k) = stc(i, k, jj) + ci(k)                                         !%dnu
               end do                                                                     !%dnu
               !end call hstep_gpu                                                        !%dnu
                                                                                          !%dn
            end if                                                                        !%dn
            !$acc loop seq                                                                !%dn
            do i1 = 1, nsoil                                                              !%dn
               stc(i, i1, jj) = stcf(i1)                                                  !%dn
            end do                                                                        !%dn
                                                                                          !%dn
!  --- ...  in the no snowpack case (via routine nopac branch,) update the grnd           !%dn
!           (skin) temperature here in response to the updated soil temperature           !%dn
!           profile above.  (note: inspection of routine snopac shows that t1(i, jj)      !%dn
!           below is a dummy variable only, as skin temperature is updated                !%dn
!           differently in routine snopac)                                                !%dn
            t1(i, jj) = (yy + (zz1 - 1.0)*stc(i, 1, jj))/zz1                              !%dn
            t1(i, jj) = ctfil1*t1(i, jj) + ctfil2*oldt1                                   !%dn
            !$acc loop seq                                                                !%dn
            do i1 = 1, nsoil                                                              !%dn
               stc(i, i1, jj) = ctfil1*stc(i, i1, jj) + ctfil2*stsoil(i1)                 !%dn
            end do                                                                        !%dn
                                                                                          !%dn
!  --- ...  calculate surface soil heat flux                                              !%dn
            ssoil(i, jj) = df2*(stc(i, 1, jj) - t1(i, jj))/(0.5*zsoil(1))                 !%dn
            !end call shflx_gpu                                                           !%dn
                                                                                          !%d
!  --- ...  set flx1(i, jj) and flx3(i, jj) (snopack phase change heat fluxes) to         !%d
!           zero since they are not used here in snopac.  flx2(i, jj) (freezing rain      !%d
!           heat flux) was similarly initialized in the penman routine.                   !%d
                                                                                          !%d
            flx1(i, jj) = 0.0                                                             !%d
            flx3(i, jj) = 0.0                                                             !%d
            !end call nopac_gpu                                                           !%d


         else

         !   call snopac_gpu &                                                            !%i
!  ---  inputs:                                                         !                 !%i
         ! ( nsoil, nroot, etp, prcp, smcmax, smcwlt, smcref, smcdry,   &                 !%i
         !   cmcmax, dt, df1, sfcems, sfctmp, t24, th2, fdown, epsca,   &                 !%i
         !   bexp, pc, rch, rr, cfactr, slope, kdt, frzx, psisat,       &                 !%i
         !   zsoil, dwsat, dksat, zbot, shdfac, ice, rtdis, quartz,     &                 !%i
         !   fxexp, csoil, flx2, snowng, vegtyp, ivegsrc,               &                 !%i
!  ---  input/outputs:                                                  !                 !%i
         !   prcp1, cmc, t1, stc, sncovr, sneqv, sndens, snowh,         &                 !%i
         !   sh2o, tbot, beta,                                          &                 !%i
!  ---  outputs:                                                        !                 !%i
         !   smc, ssoil, runoff1, runoff2, runoff3, edir, ec, et,       &                 !%i
         !   ett, snomlt, drip, dew, flx1, flx3, esnow,                 &                 !%i
!  ---  dummys:                                                                           !%i
         !   gx, ciin, rhsttin, dmax, rhstsin, rhstt, sice, sh2oa,      &                 !%i
         !   sh2ofg, ai, bi, ci, rhsts, stcf, stsoil, et1 &                               !%i
         !   )                !                                                           !%i
!  --- ...  convert potential evap (etp(i, jj)) from kg m-2 s-1 to m s-1 and then to      !%i
!           an amount (m) given timestep (dt) and call it an effective snowpack           !%i
!           reduction amount, esnow2 (m) for a snowcover fraction = 1.0.  this is         !%i
!           the amount the snowpack would be reduced due to sublimation from the          !%i
!           snow sfc during the timestep.  sublimation will proceed at the                !%i
!           potential rate unless the snow depth is less than the expected                !%i
!           snowpack reduction.  for snowcover fraction = 1.0,                            !%i
!           0=edir(i, jj)=et=ec(i, jj), and                                               !%i
!           hence total evap = esnow(i, jj) = sublimation (potential evap rate)           !%i
                                                                                          !%i
!  --- ...  if sea-ice (ice=1) or glacial-ice (ice=-1), snowcover fraction = 1.0,         !%i
!           and sublimation is at the potential rate.                                     !%i
!           for non-glacial land (ice=0), if snowcover fraction < 1.0, total              !%i
!           evaporation < potential due to non-potential contribution from                !%i
!           non-snow covered fraction.                                                    !%i
         !if (myrank .eq. 0) write(*,*) 'df1 in snopac:', loc(df1)                        !%i
                                                                                          !%i
                                                                                          !%i
                                                                                          !%i
            prcp1 = prcp1*0.001                                                           !%i
                                                                                          !%i
            edir(i, jj) = 0.0                                                             !%i
            edir1 = 0.0                                                                   !%i
                                                                                          !%i
            ec(i, jj) = 0.0                                                               !%i
            ec1 = 0.0                                                                     !%i
            !$acc loop seq                                                                !%i
            do k = 1, nsoil                                                               !%i
               et(i, k, jj) = 0.0                                                         !%i
               et1(k) = 0.0                                                               !%i
            end do                                                                        !%i
                                                                                          !%i
            ett(i, jj) = 0.0                                                              !%i
            ett1 = 0.0                                                                    !%i
            etns = 0.0                                                                    !%i
            etns1 = 0.0                                                                   !%i
            esnow(i, jj) = 0.0                                                            !%i
            esnow1 = 0.0                                                                  !%i
            esnow2 = 0.0                                                                  !%i
            dew(i, jj) = 0.0                                                              !%i
            etp1 = etp(i, jj)*0.001                                                       !%i
                                                                                          !%i
            if (etp(i, jj) < 0.0) then                                                    !%i
                                                                                          !%i
!  --- ...  if etp(i, jj)<0 (downward) then dewfall (=frostfall in this case).            !%i
               dew(i, jj) = -etp1                                                         !%i
               esnow2 = etp1*dt                                                           !%i
               etanrg = etp(i, jj)*((1.0 - sncovr(i, jj))*lsubc + sncovr(i, jj)*lsubs)    !%i
                                                                                          !%i
            else                                                                          !%i
                                                                                          !%i
!  --- ...  etp(i, jj) >= 0, upward moisture flux                                         !%i
               if (ice /= 0) then           ! for sea-ice and glacial-ice case            !%i
                  esnow(i, jj) = etp(i, jj)                                               !%i
                  esnow1 = esnow(i, jj)*0.001                                             !%i
                  esnow2 = esnow1*dt                                                      !%i
                  etanrg = esnow(i, jj)*lsubs                                             !%i
                                                                                          !%i
               else                         ! for non-glacial land case                   !%i
                  if (sncovr(i, jj) < 1.0) then                                           !%i
                                                                                          !%i
                     !call evapo_gpu &                                                    !%im
                     !   !  ---  inputs: &                                                !%im
                     !   (nsoil, nroot, cmc, cmcmax, etp1, dt, zsoil, &                   !%im
                     !    sh2o, smcmax, smcwlt, smcref, smcdry, pc, &                     !%im
                     !    shdfac, cfactr, rtdis, fxexp, &                                 !%im
                     !    !  ---  outputs: &                                              !%im
                     !    etns1, edir1, ec1, et1, ett1, &                                 !%im
                     !    !  ---  dummys:                                                 !%im
                     !    gx &                                                            !%im
                     !    )                                                               !%im
         !  --- ...  executable code begins here if the potential evapotranspiration      !%im
         !           is greater than zero.                                                !%im
                     edir1 = 0.0                                                          !%im
                     ec1 = 0.0                                                            !%im
                     !$acc loop seq                                                       !%im
                     do k = 1, nsoil                                                      !%im
                        et1(k) = 0.0                                                      !%im
                     end do                                                               !%im
                     ett1 = 0.0                                                           !%im
                     if (etp1 > 0.0) then                                                 !%im
                                                                                          !%im
         !  --- ...  retrieve direct evaporation from soil surface.  call this            !%im
         !           function only if veg cover not complete.                             !%im
         !           frozen ground version:  sh2o states replace smc states.              !%im
                        if (shdfac(i, jj) < 1.0) then                                     !%im
                                                                                          !%im
                           !call devap_gpu &                                              !%imq
                           !   !  ---  inputs: &                                          !%imq
                           !   (etp1, sh2o(1), shdfac, smcmax, smcdry, fxexp, &           !%imq
                           !    !  ---  outputs: &                                        !%imq
                           !    edir1 &                                                   !%imq
                           !    )                                                         !%imq
               !  --- ...  direct evap a function of relative soil moisture               !%imq
               !           availability, linear when fxexp=1.                             !%imq
               !           fx > 1 represents demand control                               !%imq
               !           fx < 1 represents flux control                                 !%imq
                           sratio = (sh2o(i, 1, jj) - smcdry(i, jj)) &                    !%imq
                                    /(smcmax(i, jj) - smcdry(i, jj))                      !%imq
                                                                                          !%imq
                           if (sratio > 0.0) then                                         !%imq
                              fx = sratio**fxexp                                          !%imq
                              fx = max(min(fx, 1.0), 0.0)                                 !%imq
                           else                                                           !%imq
                              fx = 0.0                                                    !%imq
                           end if                                                         !%imq
                                                                                          !%imq
               !  --- ...  allow for the direct-evap-reducing effect of shade             !%imq
                           edir1 = fx*(1.0 - shdfac(i, jj))*etp1                          !%imq
                           !end call devap_gpu                                            !%imq
                                                                                          !%im
                        end if                                                            !%im
                                                                                          !%im
         !  --- ...  initialize plant total transpiration, retrieve plant                 !%im
         !           transpiration, and accumulate it for all soil layers.                !%im
                        if (shdfac(i, jj) > 0.0) then                                     !%im
                                                                                          !%im
                           !call transp_gpu &                                             !%imB
                           !   !  ---  inputs: &                                          !%imB
                           !   (nsoil, nroot, etp1, sh2o, smcwlt, smcref, &               !%imB
                           !    cmc, cmcmax, zsoil, shdfac, pc, cfactr, rtdis, &          !%imB
                           !    !  ---  outputs: &                                        !%imB
                           !    et1, &                                                    !%imB
                           !    !  ---  dummys: &                                         !%imB
                           !    gx &                                                      !%imB
                           !    )                                                         !%imB
               !  --- ...  initialize plant transp to zero for all soil layers.           !%imB
                           !$acc loop seq                                                 !%imB
                           do k = 1, nsoil                                                !%imB
                              et1(k) = 0.0                                                !%imB
                           end do                                                         !%imB
                                                                                          !%imB
               !  --- ...  calculate an 'adjusted' potential transpiration                !%imB
               !           if statement below to avoid tangent linear problems near       !%imB
               !           zero note: gx and other terms below redistribute               !%imB
               !           transpiration by layer, et(k), as a function of soil           !%imB
               !           moisture availability, while preserving total etp1a.           !%imB
                           if (cmc(i, jj) /= 0.0) then                                    !%imB
                              etp1a = shdfac(i, jj)*pc(i, jj)*etp1*(1.0 &                 !%imB
                                      - (cmc(i, jj)/cmcmax)**cfactr)                      !%imB
                           else                                                           !%imB
                              etp1a = shdfac(i, jj)*pc(i, jj)*etp1                        !%imB
                           end if                                                         !%imB
                                                                                          !%imB
                           sgx = 0.0                                                      !%imB
                           !$acc loop seq                                                 !%imB
                           do i1 = 1, nroot(i, jj)                                        !%imB
                              gx(i1) = (sh2o(i, i1, jj) - smcwlt(i, jj)) &                !%imB
                                       /(smcref(i, jj) - smcwlt(i, jj))                   !%imB
                              gx(i1) = max(min(gx(i1), 1.0), 0.0)                         !%imB
                              sgx = sgx + gx(i1)                                          !%imB
                           end do                                                         !%imB
                           sgx = sgx/nroot(i, jj)                                         !%imB
                                                                                          !%imB
                           denom_3 = 0.0                                                  !%imB
                           !$acc loop seq                                                 !%imB
                           do i1 = 1, nroot(i, jj)                                        !%imB
                              rtx = rtdis(i1) + gx(i1) - sgx                              !%imB
                              gx(i1) = gx(i1)*max(rtx, 0.0)                               !%imB
                              denom_3 = denom_3 + gx(i1)                                  !%imB
                           end do                                                         !%imB
                           if (denom_3 <= 0.0) denom_3 = 1.0                              !%imB
                                                                                          !%imB
                           !$acc loop seq                                                 !%imB
                           do i1 = 1, nroot(i, jj)                                        !%imB
                              et1(i1) = etp1a*gx(i1)/denom_3                              !%imB
                           end do                                                         !%imB
                           !end call transp_gpu                                           !%imB
                           !$acc loop seq                                                 !%im
                           do k = 1, nsoil                                                !%im
                              ett1 = ett1 + et1(k)                                        !%im
                           end do                                                         !%im
                                                                                          !%im
         !  --- ...  calculate canopy evaporation.                                        !%im
         !           if statements to avoid tangent linear problems near                  !%im
         !           cmc(i, jj)=0.0.                                                      !%im
                           if (cmc(i, jj) > 0.0) then                                     !%im
                              ec1 = shdfac(i, jj)*((cmc(i, jj)/cmcmax)**cfactr)*etp1      !%im
                           else                                                           !%im
                              ec1 = 0.0                                                   !%im
                           end if                                                         !%im
                                                                                          !%im
         !  --- ...  ec(i, jj) should be limited by the total amount of available         !%im
         !           water on the canopy.  -f.chen, 18-oct-1994                           !%im
                           cmc2ms = cmc(i, jj)/dt                                         !%im
                           ec1 = min(cmc2ms, ec1)                                         !%im
                        end if                                                            !%im
                                                                                          !%im
                     end if   ! end if_etp1_block                                         !%im
                                                                                          !%im
         !  --- ...  total up evap and transp types to obtain actual evapotransp          !%im
                     etns1 = edir1 + ett1 + ec1                                           !%im
                     !end call evapo_gpu                                                  !%im
                     edir1 = edir1*(1.0 - sncovr(i, jj))                                  !%i
                     ec1 = ec1*(1.0 - sncovr(i, jj))                                      !%i
                     !$acc loop seq                                                       !%i
                     do k = 1, nsoil                                                      !%i
                        et1(k) = et1(k)*(1.0 - sncovr(i, jj))                             !%i
                     end do                                                               !%i
                                                                                          !%i
                     ett1 = ett1*(1.0 - sncovr(i, jj))                                    !%i
                     etns1 = etns1*(1.0 - sncovr(i, jj))                                  !%i
                                                                                          !%i
                     edir(i, jj) = edir1*1000.0                                           !%i
                     ec(i, jj) = ec1*1000.0                                               !%i
                                                                                          !%i
                     !$acc loop seq                                                       !%i
                     do k = 1, nsoil                                                      !%i
                        et(i, k, jj) = et1(k)*1000.0                                      !%i
                     end do                                                               !%i
                     ett(i, jj) = ett1*1000.0                                             !%i
                     etns = etns1*1000.0                                                  !%i
                                                                                          !%i
                  end if   ! end if_sncovr_block                                          !%i
                  esnow(i, jj) = etp(i, jj)*sncovr(i, jj)                                 !%i
!         esnow1 = etp(i, jj) * 0.001                                                     !%i
                  esnow1 = esnow(i, jj)*0.001                                             !%i
                  esnow2 = esnow1*dt                                                      !%i
                  etanrg = esnow(i, jj)*lsubs + etns*lsubc                                !%i
                                                                                          !%i
               end if   ! end if_ice_block                                                !%i
                                                                                          !%i
            end if   ! end if_etp_block                                                   !%i
                                                                                          !%i
!  --- ...  if precip is falling, calculate heat flux from snow sfc to newly              !%i
!           accumulating precip.  note that this reflects the flux appropriate for        !%i
!           the not-yet-updated skin temperature (t1(i, jj)).  assumes temperature of     !%i
!           the snowfall striking the gound is =sfctmp(i, jj) (lowest model level air     !%i
!          temp).                                                                         !%i
            flx1(i, jj) = 0.0                                                             !%i
            if (snowng) then                                                              !%i
               flx1(i, jj) = cpice*prcp(i, jj)*(t1(i, jj) - sfctmp(i, jj))                !%i
            else                                                                          !%i
               if (prcp(i, jj) > 0.0) then                                                !%i
                  flx1(i, jj) = cph2o1*prcp(i, jj)*(t1(i, jj) - sfctmp(i, jj))            !%i
               end if                                                                     !%i
            end if                                                                        !%i
                                                                                          !%i
!  --- ...  calculate an 'effective snow-grnd sfc temp' (t12) based on heat               !%i
!           fluxes between the snow pack and the soil and on net radiation.               !%i
!           include flx1(i, jj) (precip-snow sfc) and flx2(i, jj) (freezing rain          !%i
!           latent heat) fluxes.                                                          !%i
!           flx2(i, jj) reflects freezing rain latent heat flux using t1(i, jj)           !%i
!           calculated in penman.                                                         !%i
            dsoil_1 = -0.5*zsoil(1)                                                       !%i
            dtot_1 = snowh(i, jj) + dsoil_1                                               !%i
            denom = 1.0 + df1/(dtot_1*rr*rch)                                             !%i
                                                                                          !%i
!     t12a = ( (fdown - flx1(i, jj) - flx2(i, jj) - sigma1*t24) / rch                 &   !%i
!    &     + th2(i, jj) - sfctmp(i, jj) - beta(i, jj)*epsca ) / rr                        !%i
            t12a = ((fdown - flx1(i, jj) - flx2(i, jj) - sfcems(i, jj)*sigma1*t24)/rch &  !%i
                    + th2(i, jj) - sfctmp(i, jj) - etanrg/rch)/rr                         !%i
                                                                                          !%i
            t12b = df1*stc(i, 1, jj)/(dtot_1*rr*rch)                                      !%i
            t12 = (sfctmp(i, jj) + t12a + t12b)/denom                                     !%i
                                                                                          !%i
!  --- ...  if the 'effective snow-grnd sfc temp' is at or below freezing, no snow        !%i
!           melt will occur.  set the skin temp to this effective temp.  reduce           !%i
!           (by sublimination ) or increase (by frost) the depth of the snowpack,         !%i
!           depending on sign of etp(i, jj).                                              !%i
!           update soil heat flux (ssoil(i, jj)) using new skin temperature (t1(i, jj))   !%i
!           since no snowmelt, set accumulated snowmelt to zero, set 'effective'          !%i
!           precip from snowmelt to zero, set phase-change heat flux from snowmelt        !%i
!           to zero.                                                                      !%i
            if (t12 <= tfreez) then                                                       !%i
                                                                                          !%i
               t1(i, jj) = t12                                                            !%i
!old                                                                                      !%i
!       ssoil(i, jj) = df1 * (t1(i, jj) - stc(1)) / dtot_1                                !%i
!new_ssoil                                                                                !%i
               ssoil(i, jj) = (t1(i, jj) - stc(i, 1, jj))*max(7.0, df1/dtot_1)            !%i
               sneqv(i, jj) = max(0.0, sneqv(i, jj) - esnow2)                             !%i
               flx3(i, jj) = 0.0                                                          !%i
               ex = 0.0                                                                   !%i
               snomlt(i, jj) = 0.0                                                        !%i
                                                                                          !%i
            else                                                                          !%i
                                                                                          !%i
!  --- ...  if the 'effective snow-grnd sfc temp' is above freezing, snow melt            !%i
!           will occur.  call the snow melt rate,ex and amt, snomlt(i, jj).  revise       !%i
!           the effective snow depth.  revise the skin temp because it would have chgd    !%i
!           due to the latent heat released by the melting. calc the latent heat          !%i
!           released, flx3(i, jj). set the effective precip, prcp1 to the snow melt       !%i
!           rate, ex for use in smflx.  adjustment to t1(i, jj) to account for snow       !%i
!           patches. calculate qsat valid at freezing point.  note that esat (saturation  !%i
!           vapor pressure) value of 6.11e+2 used here is that valid at frzzing           !%i
!           point.  note that etp(i, jj) from call penman in sflx is ignored here in      !%i
!           favor of bulk etp(i, jj) over 'open water' at freezing temp.                  !%i
!           update soil heat flux (s) using new skin temperature (t1(i, jj))              !%i
                                                                                          !%i
!  --- ...  noah v2.7.1   mek feb2004                                                     !%i
!           non-linear weighting of snow vs non-snow covered portions of gridbox          !%i
!           so with snoexp = 2.0 (>1), surface skin temperature is higher than            !%i
!           for the linear case (snoexp = 1).                                             !%i
                                                                                          !%i
               t1(i, jj) = tfreez*sncovr(i, jj)**snoexp + t12*(1.0 &                      !%i
                           - sncovr(i, jj)**snoexp)                                       !%i
                                                                                          !%i
               beta(i, jj) = 1.0                                                          !%i
               ssoil(i, jj) = df1*(t1(i, jj) - stc(i, 1, jj))/dtot_1                      !%i
                                                                                          !%i
!  --- ...  if potential evap (sublimation) greater than depth of snowpack.               !%i
!           beta(i, jj)<1                                                                 !%i
!           snowpack has sublimated away, set depth to zero.                              !%i
                                                                                          !%i
               if (sneqv(i, jj) - esnow2 <= esdmin) then                                  !%i
                                                                                          !%i
                  sneqv(i, jj) = 0.0                                                      !%i
                  ex = 0.0                                                                !%i
                  snomlt(i, jj) = 0.0                                                     !%i
                  flx3(i, jj) = 0.0                                                       !%i
                                                                                          !%i
               else                                                                       !%i
                                                                                          !%i
!  --- ...  potential evap (sublimation) less than depth of snowpack, retain              !%i
!           beta(i, jj)=1.                                                                !%i
                                                                                          !%i
                  sneqv(i, jj) = sneqv(i, jj) - esnow2                                    !%i
                  seh = rch*(t1(i, jj) - th2(i, jj))                                      !%i
                                                                                          !%i
                  t14 = t1(i, jj)*t1(i, jj)                                               !%i
                  t14 = t14*t14                                                           !%i
                                                                                          !%i
                  flx3(i, jj) = fdown - flx1(i, jj) - flx2(i, jj) - sfcems(i, jj) &       !%i
                                *sigma1*t14 - ssoil(i, jj) - seh - etanrg                 !%i
                  if (flx3(i, jj) <= 0.0) flx3(i, jj) = 0.0                               !%i
                                                                                          !%i
                  ex = flx3(i, jj)*0.001/lsubf                                            !%i
                                                                                          !%i
!  --- ...  snowmelt reduction depending on snow cover                                    !%i
!           if snow cover less than 5% no snowmelt reduction                              !%i
!     note: does 'if' below fail to match the melt water with the melt                    !%i
!           energy?                                                                       !%i
                                                                                          !%i
!         if (sncovr(i, jj) > 0.05) ex = ex * sncovr(i, jj)                               !%i
                  snomlt(i, jj) = ex*dt                                                   !%i
                                                                                          !%i
!  --- ...  esdmin represents a snowpack depth threshold value below which we             !%i
!           choose not to retain any snowpack, and instead include it in snowmelt.        !%i
                                                                                          !%i
                  if (sneqv(i, jj) - snomlt(i, jj) >= esdmin) then                        !%i
                                                                                          !%i
                     sneqv(i, jj) = sneqv(i, jj) - snomlt(i, jj)                          !%i
                                                                                          !%i
                  else                                                                    !%i
                                                                                          !%i
!  --- ...  snowmelt exceeds snow depth                                                   !%i
                     ex = sneqv(i, jj)/dt                                                 !%i
                     flx3(i, jj) = ex*1000.0*lsubf                                        !%i
                     snomlt(i, jj) = sneqv(i, jj)                                         !%i
                     sneqv(i, jj) = 0.0                                                   !%i
                                                                                          !%i
                  end if   ! end if_sneqv-snomlt_block                                    !%i
                                                                                          !%i
               end if   ! end if_sneqv-esnow2_block                                       !%i
                                                                                          !%i
!       prcp1 = prcp1 + ex                                                                !%i
                                                                                          !%i
!  --- ...  if non-glacial land, add snowmelt rate (ex) to precip rate to be used         !%i
!           in subroutine smflx (soil moisture evolution) via infiltration.               !%i
                                                                                          !%i
!  --- ...  for sea-ice and glacial-ice, the snowmelt will be added to subsurface         !%i
!           runoff/baseflow later near the end of sflx (after return from call to         !%i
!           subroutine snopac)                                                            !%i
                                                                                          !%i
               if (ice == 0) then                                                         !%i
                  prcp1 = prcp1 + ex                                                      !%i
               end if                                                                     !%i
                                                                                          !%i
            end if   ! end if_t12<=tfreez_block                                           !%i
                                                                                          !%i
!  --- ...  final beta(i, jj) now in hand, so compute evaporation.  evap equals           !%i
!           etp(i, jj) unless beta(i, jj)<1.                                              !%i
                                                                                          !%i
!      eta(i, jj) = beta(i, jj) * etp(i, jj)                                              !%i
                                                                                          !%i
!  --- ...  smflx returns updated soil moisture values for non-glacial land.              !%i
!           if sea-ice (ice=1) or glacial-ice (ice=-1), skip call to smflx, since         !%i
!           no soil medium for sea-ice or glacial-ice                                     !%i
            if (ice == 0) then                                                            !%i
                                                                                          !%i
               !call smflx_gpu &                                                          !%io
               !   !  ---  inputs: &                                                      !%io
               !   (nsoil, dt, kdt, smcmax, smcwlt, cmcmax, prcp1, &                      !%io
               !    zsoil, slope, frzx, bexp, dksat, dwsat, shdfac, &                     !%io
               !    edir1, ec1, et1, &                                                    !%io
               !    !  ---  input/outputs: &                                              !%io
               !    cmc, sh2o, &                                                          !%io
               !    !  ---  outputs: &                                                    !%io
               !    smc, runoff1, runoff2, runoff3, drip, &                               !%io
               !    !  ---  dummys: &                                                     !%io
               !    ciin, rhsttin, dmax, rhstt, sice, sh2oa, sh2ofg, ai, bi, ci &         !%io
               !    )                                                                     !%io
   !  --- ...  executable code begins here.                                               !%io
               dummy = 0.0                                                                !%io
                                                                                          !%io
   !  --- ...  compute the right hand side of the canopy eqn term ( rhsct )               !%io
                                                                                          !%io
               rhsct = shdfac(i, jj)*prcp1 - ec1                                          !%io
                                                                                          !%io
   !  --- ...  convert rhsct (a rate) to trhsct (an amount) and add it to                 !%io
   !           existing cmc(i, jj).  if resulting amt exceeds max capacity, it becomes    !%io
   !           drip(i, jj) and will fall to the grnd.                                     !%io
                                                                                          !%io
               drip(i, jj) = 0.0                                                          !%io
               trhsct = dt*rhsct                                                          !%io
               excess = cmc(i, jj) + trhsct                                               !%io
               if (excess > cmcmax) then                                                  !%io
                  drip(i, jj) = excess - cmcmax                                           !%io
               end if                                                                     !%io
                                                                                          !%io
   !  --- ...  pcpdrp is the combined prcp1 and drip(i, jj) (from cmc(i, jj)) that        !%io
   !           goes into the soil                                                         !%io
               pcpdrp = (1.0 - shdfac(i, jj))*prcp1 + drip(i, jj)/dt                      !%io
                                                                                          !%io
   !  --- ...  store ice content at each soil layer before calling srt & sstep            !%io
                                                                                          !%io
               !$acc loop seq                                                             !%io
               do i1 = 1, nsoil                                                           !%io
                  sice(i1) = smc(i, i1, jj) - sh2o(i, i1, jj)                             !%io
               end do                                                                     !%io
                                                                                          !%io
   !  --- ...  call subroutines srt and sstep to solve the soil moisture                  !%io
   !           tendency equations.                                                        !%io
                                                                                          !%io
   !  ---  if the infiltrating precip rate is nontrivial,                                 !%io
   !         (we consider nontrivial to be a precip total over the time step              !%io
   !         exceeding one one-thousandth of the water holding capacity of                !%io
   !         the first soil layer)                                                        !%io
   !       then call the srt/sstep subroutine pair twice in the manner of                 !%io
   !         time scheme "f" (implicit state, averaged coefficient)                       !%io
   !         of section 2 of kalnay and kanamitsu (1988, mwr, vol 116,                    !%io
   !         pages 1945-1958)to minimize 2-delta-t oscillations in the                    !%io
   !         soil moisture value of the top soil layer that can arise because             !%io
   !         of the extreme nonlinear dependence of the soil hydraulic                    !%io
   !         diffusivity coefficient and the hydraulic conductivity on the                !%io
   !         soil moisture state                                                          !%io
   !       otherwise call the srt/sstep subroutine pair once in the manner of             !%io
   !         time scheme "d" (implicit state, explicit coefficient)                       !%io
   !         of section 2 of kalnay and kanamitsu                                         !%io
   !       pcpdrp is units of kg/m**2/s or mm/s, zsoil is negative depth in m             !%io
                                                                                          !%io
   !     if ( pcpdrp .gt. 0.0 ) then                                                      !%io
               if ((pcpdrp*dt) > (0.001*1000.0*(-zsoil(1))*smcmax(i, jj))) then           !%io
                                                                                          !%io
   !  --- ...  frozen ground version:                                                     !%io
   !           smc states replaced by sh2o states in srt subr.  sh2o & sice states        !%io
   !           included in sstep subr.  frozen ground correction factor, frzx             !%io
   !           added.  all water balance calculations using unfrozen water                !%io
                                                                                          !%io
                  !call srt_gpu &                                                         !%iox
                  !   !  ---  inputs: &                                                   !%iox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%iox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%iox
                  !    !  ---  outputs: &                                                 !%iox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%iox
                  !    !  ---  dummys: &                                                  !%iox
                  !    dmax &                                                             !%iox
                  !    )                                                                  !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%iox
      !           of areal distribution function of soil ice content which equals         !%iox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%iox
      !           based on field data cv depends on areal mean of frozen depth, and       !%iox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%iox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%iox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
      ! ----------------------------------------------------------------------            !%iox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%iox
      !           the infiltration formule from schaake and koren model.                  !%iox
      !           modified by q duan                                                      !%iox
                                                                                          !%iox
                  iohinf = 1                                                              !%iox
                                                                                          !%iox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%iox
      !          soil layers.                                                             !%iox
                                                                                          !%iox
                  sicemax = 0.0                                                           !%iox
                  !$acc loop seq                                                          !%iox
                  do ks = 1, nsoil                                                        !%iox
                     if (sice(ks) > sicemax) then                                         !%iox
                        sicemax = sice(ks)                                                !%iox
                     end if                                                               !%iox
                  end do                                                                  !%iox
                                                                                          !%iox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%iox
                  pddum = pcpdrp                                                          !%iox
                  runoff1(i, jj) = 0.0                                                    !%iox
                                                                                          !%iox
                  if (pcpdrp /= 0.0) then                                                 !%iox
                                                                                          !%iox
      !  --- ...  modified by q. duan, 5/16/94                                            !%iox
                     dt1 = dt/86400.                                                      !%iox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%iox
                     dmax(1) = -zsoil(1)*smcav                                            !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                                                                                          !%iox
                     dice = -zsoil(1)*sice(1)                                             !%iox
                                                                                          !%iox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) &                 !%iox
                               - smcwlt(i, jj))/smcav)                                    !%iox
                     dd = dmax(1)                                                         !%iox
                                                                                          !%iox
                     !$acc loop seq                                                       !%iox
                     do ks = 2, nsoil                                                     !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%iox
                                                                                          !%iox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%iox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) &          !%iox
                                   - smcwlt(i, jj))/smcav)                                !%iox
                        dd = dd + dmax(ks)                                                !%iox
                     end do                                                               !%iox
                                                                                          !%iox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%iox
      !           in below, remove the sqrt in above                                      !%iox
                     val = 1.0 - exp(-kdt*dt1)                                            !%iox
                     ddt = dd*val                                                         !%iox
                                                                                          !%iox
                     px = pcpdrp*dt                                                       !%iox
                     if (px < 0.0) px = 0.0                                               !%iox
                                                                                          !%iox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reduction of infiltration based on frozen ground parameters             !%iox
                                                                                          !%iox
                     fcr = 1.                                                             !%iox
                     if (dice > 1.e-2) then                                               !%iox
                        acrt = cvfrz*frzx/dice                                            !%iox
                        sum = 1.                                                          !%iox
                                                                                          !%iox
                        ialp1 = cvfrz - 1                                                 !%iox
                        !$acc loop seq                                                    !%iox
                        do j = 1, ialp1                                                   !%iox
                           k = 1                                                          !%iox
                                                                                          !%iox
                           !$acc loop seq                                                 !%iox
                           do j1 = j + 1, ialp1                                           !%iox
                              k = k*j1                                                    !%iox
                           end do                                                         !%iox
                                                                                          !%iox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%iox
                        end do                                                            !%iox
                                                                                          !%iox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%iox
                     end if                                                               !%iox
                     infmax = infmax*fcr                                                  !%iox
                                                                                          !%iox
      !  --- ...  correction of infiltration limitation:                                  !%iox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%iox
      !           of hydrolic conductivity                                                !%iox
                                                                                          !%iox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%iox
                     mxsmc = sh2o(i, 1, jj)                                               !%iox
                                                                                          !%iox
                     !call wdfcnd_gpu &                                                   !%ioxC
                     !   !  ---  inputs: &                                                !%ioxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%ioxC
                     !    !  ---  outputs: &                                              !%ioxC
                     !    wdf, wcnd &                                                     !%ioxC
                     !    )                                                               !%ioxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%ioxC
                                                                                          !%ioxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%ioxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%ioxC
                                                                                          !%ioxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%ioxC
                                                                                          !%ioxC
                     expon = bexp + 2.0                                                   !%ioxC
                     wdf = dwsat*factr2**expon                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf                                         !%ioxC
         !                                                                                !%ioxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%ioxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%ioxC
                     if (sicemax > 0.0) then                                              !%ioxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%ioxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%ioxC
                     end if                                                               !%ioxC
                                                                                          !%ioxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%ioxC
                     expon = (2.0*bexp) + 3.0                                             !%ioxC
                     wcnd = dksat*factr2**expon                                           !%ioxC
                     !end call wdfcnd_gpu                                                 !%ioxC
                                                                                          !%iox
                     infmax = max(infmax, wcnd)                                           !%iox
                     infmax = min(infmax, px)                                             !%iox
                                                                                          !%iox
                     if (pcpdrp > infmax) then                                            !%iox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%iox
                        pddum = infmax                                                    !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                  end if   ! end if_pcpdrp_block                                          !%iox
                                                                                          !%iox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%iox
      !          in line below replaced with new approach in 2nd line:                    !%iox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%iox
                                                                                          !%iox
                  mxsmc = sh2o(i, 1, jj)                                                  !%iox
                                                                                          !%iox
                  !call wdfcnd_gpu &                                                      !%ioxC
                  !   !  ---  inputs: &                                                   !%ioxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%ioxC
                  !    !  ---  outputs: &                                                 !%ioxC
                  !    wdf, wcnd &                                                        !%ioxC
                  !    )                                                                  !%ioxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%ioxC
                                                                                          !%ioxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%ioxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%ioxC
                                                                                          !%ioxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%ioxC
                                                                                          !%ioxC
                  expon = bexp + 2.0                                                      !%ioxC
                  wdf = dwsat*factr2**expon                                               !%ioxC
                                                                                          !%ioxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%ioxC
      !           gradient of unfrozen water. the latter gradient can become very         !%ioxC
      !           extreme in freezing/thawing situations, and given the relatively        !%ioxC
      !           few and thick soil layers, this gradient sufferes serious               !%ioxC
      !           trunction errors yielding erroneously high vertical transports of       !%ioxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%ioxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%ioxC
      !                                                                                   !%ioxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%ioxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%ioxC
                  if (sicemax > 0.0) then                                                 !%ioxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%ioxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%ioxC
                  end if                                                                  !%ioxC
                                                                                          !%ioxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%ioxC
                  expon = (2.0*bexp) + 3.0                                                !%ioxC
                  wcnd = dksat*factr2**expon                                              !%ioxC
                  !end call wdfcnd_gpu                                                    !%ioxC
                                                                                          !%iox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%iox
                                                                                          !%iox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%iox
                  ai(1) = 0.0                                                             !%iox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%iox
                  ci(1) = -bi(1)                                                          !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%iox
      !           moisture gradient btwn the top and next to top layers.                  !%iox
                                                                                          !%iox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%iox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%iox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%iox
                                                                                          !%iox
      !  --- ...  initialize ddz2                                                         !%iox
                                                                                          !%iox
                  ddz2 = 0.0                                                              !%iox
                                                                                          !%iox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%iox
                  !$acc loop seq                                                          !%iox
                  do k = 2, nsoil                                                         !%iox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%iox
                     if (k /= nsoil) then                                                 !%iox
                        slopx = 1.0                                                       !%iox
                                                                                          !%iox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%iox
      !           in line below replaced with new approach in 2nd line:                   !%iox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%iox
                        mxsmc2 = sh2o(i, k, jj)                                           !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%iox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%iox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%iox
                                                                                          !%iox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%iox
                                                                                          !%iox
                        ddz2 = 2.0/denom_4                                                !%iox
                        ci(k) = -wdf2*ddz2/denom2                                         !%iox
                                                                                          !%iox
                     else   ! if_k_block                                                  !%iox
                                                                                          !%iox
      !  --- ...  slope of bottom layer is introduced                                     !%iox
                        slopx = slope                                                     !%iox
                                                                                          !%iox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%iox
      !           for this layer                                                          !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%iox
                        dsmdz2 = 0.0                                                      !%iox
                                                                                          !%iox
      !  --- ...  set matrix coef ci to zero                                              !%iox
                                                                                          !%iox
                        ci(k) = 0.0                                                       !%iox
                                                                                          !%iox
                     end if   ! end if_k_block                                            !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%iox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%iox
                     rhstt(k) = numer/(-denom2)                                           !%iox
                                                                                          !%iox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%iox
                                                                                          !%iox
                     ai(k) = -wdf*ddz_1/denom2                                            !%iox
                     bi(k) = -(ai(k) + ci(k))                                             !%iox
                                                                                          !%iox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%iox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%iox
                     if (k == nsoil) then                                                 !%iox
                        runoff2(i, jj) = slopx*wcnd2                                      !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                     if (k /= nsoil) then                                                 !%iox
                        wdf = wdf2                                                        !%iox
                        wcnd = wcnd2                                                      !%iox
                        dsmdz = dsmdz2                                                    !%iox
                        ddz_1 = ddz2                                                      !%iox
                     end if                                                               !%iox
                  end do   ! end do_k_loop                                                !%iox
                  !end call srt_gpu                                                       !%iox
                                                                                          !%io
                  !call sstep_gpu &                                                       !%ioy
                  !   !  ---  inputs: &                                                   !%ioy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%ioy
                  !    !  ---  input/outputs: &                                           !%ioy
                  !    dummy, rhstt, ai, bi, ci, &                                        !%ioy
                  !    !  ---  outputs: &                                                 !%ioy
                  !    sh2ofg, runoff3, smc, &                                            !%ioy
                  !    !  ---  dummys: &                                                  !%ioy
                  !    ciin, rhsttin &                                                    !%ioy
                  !    )                                                                  !%ioy
                  !  --- ...  create 'amount' values of variables to be input to the      !%ioy
   !           tri-diagonal matrix routine.                                               !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  rhstt(k) = rhstt(k)*dt                                                  !%ioy
                  ai(k) = ai(k)*dt                                                        !%ioy
                  bi(k) = 1.+bi(k)*dt                                                     !%ioy
                  ci(k) = ci(k)*dt                                                        !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  copy values for input variables before call to rosr12                      !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  rhsttin(k) = rhstt(k)                                                   !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsold                                                            !%ioy
                  ciin(k) = ci(k)                                                         !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%ioy
                                                                                          !%ioy
               !call rosr12_gpu &                                                         !%ioyv
               !   !  ---  inputs: &                                                      !%ioyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%ioyv
               !    !  ---  input/outputs: &                                              !%ioyv
               !    ciin, &                                                               !%ioyv
               !    !  ---  outputs: &                                                    !%ioyv
               !    ci, rhstt &                                                           !%ioyv
               !    )                                                                     !%ioyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%ioyv
                                                                                          !%ioyv
               ciin(nsoil) = 0.0                                                          !%ioyv
                                                                                          !%ioyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%ioyv
               ci(1) = -ciin(1)/bi(1)                                                     !%ioyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%ioyv
                                                                                          !%ioyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%ioyv
               !$acc loop seq                                                             !%ioyv
               do k = 2, nsoil                                                            !%ioyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%ioyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%ioyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%ioyv
               end do                                                                     !%ioyv
                                                                                          !%ioyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%ioyv
               ci(nsoil) = rhstt(nsoil)                                                   !%ioyv
                                                                                          !%ioyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%ioyv
                                                                                          !%ioyv
               !$acc loop seq                                                             !%ioyv
               do k = 2, nsoil                                                            !%ioyv
                  kk = nsoil - k + 1                                                      !%ioyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%ioyv
               end do                                                                     !%ioyv
               !end call rosr12_gpu                                                       !%ioyv
                                                                                          !%ioy
                                                                                          !%ioy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%ioy
   !           a new value.  min allowable value of smc will be 0.02.                     !%ioy
   !      runoff3(i, jj): runoff within soil layers                                       !%ioy
                                                                                          !%ioy
               wplus = 0.0                                                                !%ioy
               runoff3(i, jj) = 0.0                                                       !%ioy
               ddz = -zsoil(1)                                                            !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  if (k /= 1) then                                                        !%ioy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%ioy
                  end if                                                                  !%ioy
                  sh2ofg(k) = sh2o(i, k, jj) + ci(k) + wplus/ddz                          !%ioy
                                                                                          !%ioy
                  stot = sh2ofg(k) + sice(k)                                              !%ioy
                  if (stot > smcmax(i, jj)) then                                          !%ioy
                     if (k == 1) then                                                     !%ioy
                        ddz = -zsoil(1)                                                   !%ioy
                     else                                                                 !%ioy
                        kk11 = k - 1                                                      !%ioy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%ioy
                     end if                                                               !%ioy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%ioy
                  else                                                                    !%ioy
                     wplus = 0.0                                                          !%ioy
                  end if                                                                  !%ioy
                                                                                          !%ioy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%ioy
                  sh2ofg(k) = max(smc(i, k, jj) - sice(k), 0.0)                           !%ioy
               end do                                                                     !%ioy
               runoff3(i, jj) = wplus                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  update canopy water content/interception (dummy).  convert rhsct to        !%ioy
   !           an 'amount' value and add to previous dummy value to get new dummy.        !%ioy
                                                                                          !%ioy
               dummy = dummy + dt*rhsct                                                   !%ioy
               if (dummy < 1.e-20) dummy = 0.0                                            !%ioy
               dummy = min(dummy, cmcmax)                                                 !%ioy
               !end call sstep_gpu                                                        !%ioy
                                                                                          !%io
                  !$acc loop seq                                                          !%io
                  do k = 1, nsoil                                                         !%io
                     sh2oa(k) = (sh2o(i, k, jj) + sh2ofg(k))*0.5                          !%io
                  end do                                                                  !%io
                                                                                          !%io
                  !call srt_gpu &                                                         !%iox
                  !   !  ---  inputs: &                                                   !%iox
                  !   (nsoil, edir1, et1, sh2o, sh2oa, pcpdrp, zsoil, dwsat, &            !%iox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%iox
                  !    !  ---  outputs: &                                                 !%iox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%iox
                  !    !  ---  dummys: &                                                  !%iox
                  !    dmax &                                                             !%iox
                  !    )                                                                  !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%iox
      !           of areal distribution function of soil ice content which equals         !%iox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%iox
      !           based on field data cv depends on areal mean of frozen depth, and       !%iox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%iox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%iox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
      ! ----------------------------------------------------------------------            !%iox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%iox
      !           the infiltration formule from schaake and koren model.                  !%iox
      !           modified by q duan                                                      !%iox
                                                                                          !%iox
                  iohinf = 1                                                              !%iox
                                                                                          !%iox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%iox
      !          soil layers.                                                             !%iox
                                                                                          !%iox
                  sicemax = 0.0                                                           !%iox
                  !$acc loop seq                                                          !%iox
                  do ks = 1, nsoil                                                        !%iox
                     if (sice(ks) > sicemax) then                                         !%iox
                        sicemax = sice(ks)                                                !%iox
                     end if                                                               !%iox
                  end do                                                                  !%iox
                                                                                          !%iox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%iox
                  pddum = pcpdrp                                                          !%iox
                  runoff1(i, jj) = 0.0                                                    !%iox
                                                                                          !%iox
                  if (pcpdrp /= 0.0) then                                                 !%iox
                                                                                          !%iox
      !  --- ...  modified by q. duan, 5/16/94                                            !%iox
                     dt1 = dt/86400.                                                      !%iox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%iox
                     dmax(1) = -zsoil(1)*smcav                                            !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                                                                                          !%iox
                     dice = -zsoil(1)*sice(1)                                             !%iox
                                                                                          !%iox
                     dmax(1) = dmax(1)*(1.0 - (sh2oa(1) + sice(1) &                       !%iox
                               - smcwlt(i, jj))/smcav)                                    !%iox
                     dd = dmax(1)                                                         !%iox
                                                                                          !%iox
                     !$acc loop seq                                                       !%iox
                     do ks = 2, nsoil                                                     !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%iox
                                                                                          !%iox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%iox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2oa(ks) + sice(ks) &                !%iox
                                   - smcwlt(i, jj))/smcav)                                !%iox
                        dd = dd + dmax(ks)                                                !%iox
                     end do                                                               !%iox
                                                                                          !%iox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%iox
      !           in below, remove the sqrt in above                                      !%iox
                     val = 1.0 - exp(-kdt*dt1)                                            !%iox
                     ddt = dd*val                                                         !%iox
                                                                                          !%iox
                     px = pcpdrp*dt                                                       !%iox
                     if (px < 0.0) px = 0.0                                               !%iox
                                                                                          !%iox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reduction of infiltration based on frozen ground parameters             !%iox
                                                                                          !%iox
                     fcr = 1.                                                             !%iox
                     if (dice > 1.e-2) then                                               !%iox
                        acrt = cvfrz*frzx/dice                                            !%iox
                        sum = 1.                                                          !%iox
                                                                                          !%iox
                        ialp1 = cvfrz - 1                                                 !%iox
                        !$acc loop seq                                                    !%iox
                        do j = 1, ialp1                                                   !%iox
                           k = 1                                                          !%iox
                                                                                          !%iox
                           !$acc loop seq                                                 !%iox
                           do j1 = j + 1, ialp1                                           !%iox
                              k = k*j1                                                    !%iox
                           end do                                                         !%iox
                                                                                          !%iox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%iox
                        end do                                                            !%iox
                                                                                          !%iox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%iox
                     end if                                                               !%iox
                     infmax = infmax*fcr                                                  !%iox
                                                                                          !%iox
      !  --- ...  correction of infiltration limitation:                                  !%iox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%iox
      !           of hydrolic conductivity                                                !%iox
                                                                                          !%iox
      !       mxsmc = max ( sh2oa(1), sh2oa(2) )                                          !%iox
                     mxsmc = sh2oa(1)                                                     !%iox
                                                                                          !%iox
                     !call wdfcnd_gpu &                                                   !%ioxC
                     !   !  ---  inputs: &                                                !%ioxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%ioxC
                     !    !  ---  outputs: &                                              !%ioxC
                     !    wdf, wcnd &                                                     !%ioxC
                     !    )                                                               !%ioxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%ioxC
                                                                                          !%ioxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%ioxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%ioxC
                                                                                          !%ioxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%ioxC
                                                                                          !%ioxC
                     expon = bexp + 2.0                                                   !%ioxC
                     wdf = dwsat*factr2**expon                                            !%ioxC
                                                                                          !%ioxC
         !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the            !%ioxC
         !           vertical gradient of unfrozen water. the latter gradient can         !%ioxC
         !           become very extreme in freezing/thawing situations, and given        !%ioxC
         !           the relatively few and thick soil layers, this gradient              !%ioxC
         !           sufferes serious trunction errors yielding erroneously high          !%ioxC
         !           vertical transports of unfrozen water in both directions from        !%ioxC
         !           huge hydraulic diffusivity.therefore, we found we had to             !%ioxC
         !           arbitrarily constrain wdf                                            !%ioxC
         !                                                                                !%ioxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%ioxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%ioxC
                     if (sicemax > 0.0) then                                              !%ioxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%ioxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%ioxC
                     end if                                                               !%ioxC
                                                                                          !%ioxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%ioxC
                     expon = (2.0*bexp) + 3.0                                             !%ioxC
                     wcnd = dksat*factr2**expon                                           !%ioxC
                     !end call wdfcnd_gpu                                                 !%ioxC
                                                                                          !%iox
                     infmax = max(infmax, wcnd)                                           !%iox
                     infmax = min(infmax, px)                                             !%iox
                                                                                          !%iox
                     if (pcpdrp > infmax) then                                            !%iox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%iox
                        pddum = infmax                                                    !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                  end if   ! end if_pcpdrp_block                                          !%iox
                                                                                          !%iox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%iox
      !          in line below replaced with new approach in 2nd line:                    !%iox
      !          'mxsmc = max(sh2oa(1), sh2oa(2))'                                        !%iox
                                                                                          !%iox
                  mxsmc = sh2oa(1)                                                        !%iox
                                                                                          !%iox
                  !call wdfcnd_gpu &                                                      !%ioxC
                  !   !  ---  inputs: &                                                   !%ioxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%ioxC
                  !    !  ---  outputs: &                                                 !%ioxC
                  !    wdf, wcnd &                                                        !%ioxC
                  !    )                                                                  !%ioxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%ioxC
                                                                                          !%ioxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%ioxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%ioxC
                                                                                          !%ioxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%ioxC
                                                                                          !%ioxC
                  expon = bexp + 2.0                                                      !%ioxC
                  wdf = dwsat*factr2**expon                                               !%ioxC
                                                                                          !%ioxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%ioxC
      !           gradient of unfrozen water. the latter gradient can become very         !%ioxC
      !           extreme in freezing/thawing situations, and given the relatively        !%ioxC
      !           few and thick soil layers, this gradient sufferes serious               !%ioxC
      !           trunction errors yielding erroneously high vertical transports of       !%ioxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%ioxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%ioxC
      !                                                                                   !%ioxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%ioxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%ioxC
                  if (sicemax > 0.0) then                                                 !%ioxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%ioxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%ioxC
                  end if                                                                  !%ioxC
                                                                                          !%ioxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%ioxC
                  expon = (2.0*bexp) + 3.0                                                !%ioxC
                  wcnd = dksat*factr2**expon                                              !%ioxC
                  !end call wdfcnd_gpu                                                    !%ioxC
                                                                                          !%iox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%iox
                                                                                          !%iox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%iox
                  ai(1) = 0.0                                                             !%iox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%iox
                  ci(1) = -bi(1)                                                          !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%iox
      !           moisture gradient btwn the top and next to top layers.                  !%iox
                                                                                          !%iox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%iox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%iox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%iox
                                                                                          !%iox
      !  --- ...  initialize ddz2                                                         !%iox
                                                                                          !%iox
                  ddz2 = 0.0                                                              !%iox
                                                                                          !%iox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%iox
                  !$acc loop seq                                                          !%iox
                  do k = 2, nsoil                                                         !%iox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%iox
                     if (k /= nsoil) then                                                 !%iox
                        slopx = 1.0                                                       !%iox
                                                                                          !%iox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%iox
      !           in line below replaced with new approach in 2nd line:                   !%iox
      !           'mxsmc2 = max (sh2oa(k), sh2oa(k+1))'                                   !%iox
                        mxsmc2 = sh2oa(k)                                                 !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%iox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%iox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%iox
                                                                                          !%iox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%iox
                                                                                          !%iox
                        ddz2 = 2.0/denom_4                                                !%iox
                        ci(k) = -wdf2*ddz2/denom2                                         !%iox
                                                                                          !%iox
                     else   ! if_k_block                                                  !%iox
                                                                                          !%iox
      !  --- ...  slope of bottom layer is introduced                                     !%iox
                        slopx = slope                                                     !%iox
                                                                                          !%iox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%iox
      !           for this layer                                                          !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (sh2oa(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &         !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = sh2oa(nsoil)/smcmax(i, jj)                               !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%iox
                        dsmdz2 = 0.0                                                      !%iox
                                                                                          !%iox
      !  --- ...  set matrix coef ci to zero                                              !%iox
                                                                                          !%iox
                        ci(k) = 0.0                                                       !%iox
                                                                                          !%iox
                     end if   ! end if_k_block                                            !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%iox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%iox
                     rhstt(k) = numer/(-denom2)                                           !%iox
                                                                                          !%iox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%iox
                                                                                          !%iox
                     ai(k) = -wdf*ddz_1/denom2                                            !%iox
                     bi(k) = -(ai(k) + ci(k))                                             !%iox
                                                                                          !%iox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%iox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%iox
                     if (k == nsoil) then                                                 !%iox
                        runoff2(i, jj) = slopx*wcnd2                                      !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                     if (k /= nsoil) then                                                 !%iox
                        wdf = wdf2                                                        !%iox
                        wcnd = wcnd2                                                      !%iox
                        dsmdz = dsmdz2                                                    !%iox
                        ddz_1 = ddz2                                                      !%iox
                     end if                                                               !%iox
                  end do   ! end do_k_loop                                                !%iox
                  !end call srt_gpu                                                       !%iox
                                                                                          !%io
                  !call sstep_gpu &                                                       !%ioy
                  !   !  ---  inputs: &                                                   !%ioy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%ioy
                  !    !  ---  input/outputs: &                                           !%ioy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%ioy
                  !    !  ---  outputs: &                                                 !%ioy
                  !    sh2o, runoff3, smc, &                                              !%ioy
                  !    !  ---  dummys: &                                                  !%ioy
                  !    ciin, rhsttin &                                                    !%ioy
                  !    )                                                                  !%ioy
   !  --- ...  create 'amount' values of variables to be input to the                     !%ioy
   !           tri-diagonal matrix routine.                                               !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  rhstt(k) = rhstt(k)*dt                                                  !%ioy
                  ai(k) = ai(k)*dt                                                        !%ioy
                  bi(k) = 1.+bi(k)*dt                                                     !%ioy
                  ci(k) = ci(k)*dt                                                        !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  copy values for input variables before call to rosr12                      !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  rhsttin(k) = rhstt(k)                                                   !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsold                                                            !%ioy
                  ciin(k) = ci(k)                                                         !%ioy
               end do                                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  call rosr12 to solve the tri-diagonal matrix                               !%ioy
                                                                                          !%ioy
               !call rosr12_gpu &                                                         !%ioyv
               !   !  ---  inputs: &                                                      !%ioyv
               !   (nsoil, ai, bi, rhsttin, &                                             !%ioyv
               !    !  ---  input/outputs: &                                              !%ioyv
               !    ciin, &                                                               !%ioyv
               !    !  ---  outputs: &                                                    !%ioyv
               !    ci, rhstt &                                                           !%ioyv
               !    )                                                                     !%ioyv
               !  --- ...  initialize eqn coef ciin for the lowest soil layer             !%ioyv
                                                                                          !%ioyv
               ciin(nsoil) = 0.0                                                          !%ioyv
                                                                                          !%ioyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%ioyv
               ci(1) = -ciin(1)/bi(1)                                                     !%ioyv
               rhstt(1) = rhsttin(1)/bi(1)                                                !%ioyv
                                                                                          !%ioyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%ioyv
               !$acc loop seq                                                             !%ioyv
               do k = 2, nsoil                                                            !%ioyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%ioyv
                  rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                          !%ioyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%ioyv
               end do                                                                     !%ioyv
                                                                                          !%ioyv
   !  --- ...  set ci to rhstt for lowest soil layer                                      !%ioyv
               ci(nsoil) = rhstt(nsoil)                                                   !%ioyv
                                                                                          !%ioyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%ioyv
                                                                                          !%ioyv
               !$acc loop seq                                                             !%ioyv
               do k = 2, nsoil                                                            !%ioyv
                  kk = nsoil - k + 1                                                      !%ioyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                                  !%ioyv
               end do                                                                     !%ioyv
               !end call rosr12_gpu                                                       !%ioyv
                                                                                          !%ioy
                                                                                          !%ioy
   !  --- ...  sum the previous smc value and the matrix solution to get                  !%ioy
   !           a new value.  min allowable value of smc will be 0.02.                     !%ioy
   !      runoff3(i, jj): runoff within soil layers                                       !%ioy
                                                                                          !%ioy
               wplus = 0.0                                                                !%ioy
               runoff3(i, jj) = 0.0                                                       !%ioy
               ddz = -zsoil(1)                                                            !%ioy
                                                                                          !%ioy
               !$acc loop seq                                                             !%ioy
               do k = 1, nsoil                                                            !%ioy
                  if (k /= 1) then                                                        !%ioy
                     ddz = zsoil(k - 1) - zsoil(k)                                        !%ioy
                  end if                                                                  !%ioy
                  sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                     !%ioy
                                                                                          !%ioy
                  stot = sh2o(i, k, jj) + sice(k)                                         !%ioy
                  if (stot > smcmax(i, jj)) then                                          !%ioy
                     if (k == 1) then                                                     !%ioy
                        ddz = -zsoil(1)                                                   !%ioy
                     else                                                                 !%ioy
                        kk11 = k - 1                                                      !%ioy
                        ddz = -zsoil(k) + zsoil(kk11)                                     !%ioy
                     end if                                                               !%ioy
                     wplus = (stot - smcmax(i, jj))*ddz                                   !%ioy
                  else                                                                    !%ioy
                     wplus = 0.0                                                          !%ioy
                  end if                                                                  !%ioy
                                                                                          !%ioy
                  smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                     !%ioy
                  sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                      !%ioy
               end do                                                                     !%ioy
               runoff3(i, jj) = wplus                                                     !%ioy
                                                                                          !%ioy
   !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert rhsct      !%ioy
   !           to an 'amount' value and add to previous cmc(i, jj) value to get new       !%ioy
   !           cmc(i, jj).                                                                !%ioy
                                                                                          !%ioy
               cmc(i, jj) = cmc(i, jj) + dt*rhsct                                         !%ioy
               if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                                  !%ioy
               cmc(i, jj) = min(cmc(i, jj), cmcmax)                                       !%ioy
               !end call sstep_gpu                                                        !%ioy
                                                                                          !%io
               else                                                                       !%io
                                                                                          !%io
                  !call srt_gpu &                                                         !%iox
                  !   !  ---  inputs: &                                                   !%iox
                  !   (nsoil, edir1, et1, sh2o, sh2o, pcpdrp, zsoil, dwsat, &             !%iox
                  !    dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice, &         !%iox
                  !    !  ---  outputs: &                                                 !%iox
                  !    rhstt, runoff1, runoff2, ai, bi, ci, &                             !%iox
                  !    !  ---  dummys: &                                                  !%iox
                  !    dmax &                                                             !%iox
                  !    )                                                                  !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reference frozen ground parameter, cvfrz, is a shape parameter          !%iox
      !           of areal distribution function of soil ice content which equals         !%iox
      !           1/cv. cv is a coefficient of spatial variation of soil ice content.     !%iox
      !           based on field data cv depends on areal mean of frozen depth, and       !%iox
      !           it close to constant = 0.6 if areal mean frozen depth is above 20       !%iox
      !           cm(i, jj). that is why parameter cvfrz = 3 (int{1/0.6*0.6}).            !%iox
      !           current logic doesn't allow cvfrz be bigger than 3                      !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
                                                                                          !%iox
      ! ----------------------------------------------------------------------            !%iox
      !  --- ...  determine rainfall infiltration rate and runoff.  include               !%iox
      !           the infiltration formule from schaake and koren model.                  !%iox
      !           modified by q duan                                                      !%iox
                                                                                          !%iox
                  iohinf = 1                                                              !%iox
                                                                                          !%iox
      !  --- ... let sicemax be the greatest, if any, frozen water content within         !%iox
      !          soil layers.                                                             !%iox
                                                                                          !%iox
                  sicemax = 0.0                                                           !%iox
                  !$acc loop seq                                                          !%iox
                  do ks = 1, nsoil                                                        !%iox
                     if (sice(ks) > sicemax) then                                         !%iox
                        sicemax = sice(ks)                                                !%iox
                     end if                                                               !%iox
                  end do                                                                  !%iox
                                                                                          !%iox
      !  --- ...  determine rainfall infiltration rate and runoff                         !%iox
                  pddum = pcpdrp                                                          !%iox
                  runoff1(i, jj) = 0.0                                                    !%iox
                                                                                          !%iox
                  if (pcpdrp /= 0.0) then                                                 !%iox
                                                                                          !%iox
      !  --- ...  modified by q. duan, 5/16/94                                            !%iox
                     dt1 = dt/86400.                                                      !%iox
                     smcav = smcmax(i, jj) - smcwlt(i, jj)                                !%iox
                     dmax(1) = -zsoil(1)*smcav                                            !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                                                                                          !%iox
                     dice = -zsoil(1)*sice(1)                                             !%iox
                                                                                          !%iox
                     dmax(1) = dmax(1)*(1.0 - (sh2o(i, 1, jj) + sice(1) &                 !%iox
                               - smcwlt(i, jj))/smcav)                                    !%iox
                     dd = dmax(1)                                                         !%iox
                                                                                          !%iox
                     !$acc loop seq                                                       !%iox
                     do ks = 2, nsoil                                                     !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
                        dice = dice + (zsoil(ks - 1) - zsoil(ks))*sice(ks)                !%iox
                                                                                          !%iox
                        dmax(ks) = (zsoil(ks - 1) - zsoil(ks))*smcav                      !%iox
                        dmax(ks) = dmax(ks)*(1.0 - (sh2o(i, ks, jj) + sice(ks) &          !%iox
                                   - smcwlt(i, jj))/smcav)                                !%iox
                        dd = dd + dmax(ks)                                                !%iox
                     end do                                                               !%iox
                                                                                          !%iox
      !  --- ...  val = (1.-exp(-kdt*sqrt(dt1)))                                          !%iox
      !           in below, remove the sqrt in above                                      !%iox
                     val = 1.0 - exp(-kdt*dt1)                                            !%iox
                     ddt = dd*val                                                         !%iox
                                                                                          !%iox
                     px = pcpdrp*dt                                                       !%iox
                     if (px < 0.0) px = 0.0                                               !%iox
                                                                                          !%iox
                     infmax = (px*(ddt/(px + ddt)))/dt                                    !%iox
                                                                                          !%iox
      !  --- ...  frozen ground version:                                                  !%iox
      !           reduction of infiltration based on frozen ground parameters             !%iox
                                                                                          !%iox
                     fcr = 1.                                                             !%iox
                     if (dice > 1.e-2) then                                               !%iox
                        acrt = cvfrz*frzx/dice                                            !%iox
                        sum = 1.                                                          !%iox
                                                                                          !%iox
                        ialp1 = cvfrz - 1                                                 !%iox
                        !$acc loop seq                                                    !%iox
                        do j = 1, ialp1                                                   !%iox
                           k = 1                                                          !%iox
                                                                                          !%iox
                           !$acc loop seq                                                 !%iox
                           do j1 = j + 1, ialp1                                           !%iox
                              k = k*j1                                                    !%iox
                           end do                                                         !%iox
                                                                                          !%iox
                           sum = sum + (acrt**(cvfrz - j))/float(k)                       !%iox
                        end do                                                            !%iox
                                                                                          !%iox
                        fcr = 1.0 - exp(-acrt)*sum                                        !%iox
                     end if                                                               !%iox
                     infmax = infmax*fcr                                                  !%iox
                                                                                          !%iox
      !  --- ...  correction of infiltration limitation:                                  !%iox
      !           if infmax .le. hydrolic conductivity assign infmax the value            !%iox
      !           of hydrolic conductivity                                                !%iox
                                                                                          !%iox
      !       mxsmc = max ( sh2o(1), sh2o(2) )                                            !%iox
                     mxsmc = sh2o(i, 1, jj)                                               !%iox
                                                                                          !%iox
                     !call wdfcnd_gpu &                                                   !%ioxC
                     !   !  ---  inputs: &                                                !%ioxC
                     !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                   !%ioxC
                     !    !  ---  outputs: &                                              !%ioxC
                     !    wdf, wcnd &                                                     !%ioxC
                     !    )                                                               !%ioxC
         !  --- ...  calc the ratio of the actual to the max psbl soil h2o content        !%ioxC
                                                                                          !%ioxC
                     factr1 = 0.2/smcmax(i, jj)                                           !%ioxC
                     factr2 = mxsmc/smcmax(i, jj)                                         !%ioxC
                                                                                          !%ioxC
         !  --- ...  prep an expntl coef and calc the soil water diffusivity              !%ioxC
                                                                                          !%ioxC
                     expon = bexp + 2.0                                                   !%ioxC
                     wdf = dwsat*factr2**expon                                            !%ioxC
                                                                                          !%ioxC
         !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the            !%ioxC
         !           vertical gradient of unfrozen water. the latter gradient can         !%ioxC
         !           become very extreme in freezing/thawing situations, and given        !%ioxC
         !           the relatively few and thick soil layers, this gradient              !%ioxC
         !           sufferes serious trunction errors yielding erroneously high          !%ioxC
         !           vertical transports of unfrozen water in both directions from        !%ioxC
         !           huge hydraulic diffusivity.therefore, we found we had to             !%ioxC
         !           arbitrarily constrain wdf                                            !%ioxC
         !                                                                                !%ioxC
         !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                 !%ioxC
         !           weighted approach.......  pablo grunmann, 28_sep_1999.               !%ioxC
                     if (sicemax > 0.0) then                                              !%ioxC
                        vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                          !%ioxC
                        wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon               !%ioxC
                     end if                                                               !%ioxC
                                                                                          !%ioxC
         !  --- ...  reset the expntl coef and calc the hydraulic conductivity            !%ioxC
                     expon = (2.0*bexp) + 3.0                                             !%ioxC
                     wcnd = dksat*factr2**expon                                           !%ioxC
                     !end call wdfcnd_gpu                                                 !%ioxC
                                                                                          !%iox
                     infmax = max(infmax, wcnd)                                           !%iox
                     infmax = min(infmax, px)                                             !%iox
                                                                                          !%iox
                     if (pcpdrp > infmax) then                                            !%iox
                        runoff1(i, jj) = pcpdrp - infmax                                  !%iox
                        pddum = infmax                                                    !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                  end if   ! end if_pcpdrp_block                                          !%iox
                                                                                          !%iox
      !  --- ... to avoid spurious drainage behavior, 'upstream differencing'             !%iox
      !          in line below replaced with new approach in 2nd line:                    !%iox
      !          'mxsmc = max(sh2o(1), sh2o(2))'                                          !%iox
                                                                                          !%iox
                  mxsmc = sh2o(i, 1, jj)                                                  !%iox
                                                                                          !%iox
                  !call wdfcnd_gpu &                                                      !%ioxC
                  !   !  ---  inputs: &                                                   !%ioxC
                  !   (mxsmc, smcmax, bexp, dksat, dwsat, sicemax, &                      !%ioxC
                  !    !  ---  outputs: &                                                 !%ioxC
                  !    wdf, wcnd &                                                        !%ioxC
                  !    )                                                                  !%ioxC
      !  --- ...  calc the ratio of the actual to the max psbl soil h2o content           !%ioxC
                                                                                          !%ioxC
                  factr1 = 0.2/smcmax(i, jj)                                              !%ioxC
                  factr2 = mxsmc/smcmax(i, jj)                                            !%ioxC
                                                                                          !%ioxC
      !  --- ...  prep an expntl coef and calc the soil water diffusivity                 !%ioxC
                                                                                          !%ioxC
                  expon = bexp + 2.0                                                      !%ioxC
                  wdf = dwsat*factr2**expon                                               !%ioxC
                                                                                          !%ioxC
      !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the vertical      !%ioxC
      !           gradient of unfrozen water. the latter gradient can become very         !%ioxC
      !           extreme in freezing/thawing situations, and given the relatively        !%ioxC
      !           few and thick soil layers, this gradient sufferes serious               !%ioxC
      !           trunction errors yielding erroneously high vertical transports of       !%ioxC
      !           unfrozen water in both directions from huge hydraulic diffusivity.      !%ioxC
      !           therefore, we found we had to arbitrarily constrain wdf                 !%ioxC
      !                                                                                   !%ioxC
      !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)                    !%ioxC
      !           weighted approach.......  pablo grunmann, 28_sep_1999.                  !%ioxC
                  if (sicemax > 0.0) then                                                 !%ioxC
                     vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                             !%ioxC
                     wdf = vkwgt*wdf + (1.0 - vkwgt)*dwsat*factr1**expon                  !%ioxC
                  end if                                                                  !%ioxC
                                                                                          !%ioxC
      !  --- ...  reset the expntl coef and calc the hydraulic conductivity               !%ioxC
                  expon = (2.0*bexp) + 3.0                                                !%ioxC
                  wcnd = dksat*factr2**expon                                              !%ioxC
                  !end call wdfcnd_gpu                                                    !%ioxC
                                                                                          !%iox
      !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer           !%iox
                                                                                          !%iox
                  ddz_1 = 1.0/(-.5*zsoil(2))                                              !%iox
                  ai(1) = 0.0                                                             !%iox
                  bi(1) = wdf*ddz_1/(-zsoil(1))                                           !%iox
                  ci(1) = -bi(1)                                                          !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for the top layer after calc'ng the vertical soil            !%iox
      !           moisture gradient btwn the top and next to top layers.                  !%iox
                                                                                          !%iox
                  dsmdz = (sh2o(i, 1, jj) - sh2o(i, 2, jj))/(-.5*zsoil(2))                !%iox
                  rhstt(1) = (wdf*dsmdz + wcnd - pddum + edir1 + et1(1))/zsoil(1)         !%iox
                  sstt = wdf*dsmdz + wcnd + edir1 + et1(1)                                !%iox
                                                                                          !%iox
      !  --- ...  initialize ddz2                                                         !%iox
                                                                                          !%iox
                  ddz2 = 0.0                                                              !%iox
                                                                                          !%iox
      !  --- ...  loop thru the remaining soil layers, repeating the abv process          !%iox
                  !$acc loop seq                                                          !%iox
                  do k = 2, nsoil                                                         !%iox
                     denom2 = (zsoil(k - 1) - zsoil(k))                                   !%iox
                     if (k /= nsoil) then                                                 !%iox
                        slopx = 1.0                                                       !%iox
                                                                                          !%iox
      !  --- ...  again, to avoid spurious drainage behavior, 'upstream differencing'     !%iox
      !           in line below replaced with new approach in 2nd line:                   !%iox
      !           'mxsmc2 = max (sh2o(k), sh2o(k+1))'                                     !%iox
                        mxsmc2 = sh2o(i, k, jj)                                           !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (mxsmc2, smcmax, bexp, dksat, dwsat, sicemax, &               !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = mxsmc2/smcmax(i, jj)                                     !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc some partial products for later use in calc'ng rhstt               !%iox
                        denom_4 = (zsoil(k - 1) - zsoil(k + 1))                           !%iox
                        dsmdz2 = (sh2o(i, k, jj) - sh2o(i, k + 1, jj))/(denom_4*0.5)      !%iox
                                                                                          !%iox
      !  --- ...  calc the matrix coef, ci, after calc'ng its partial product             !%iox
                                                                                          !%iox
                        ddz2 = 2.0/denom_4                                                !%iox
                        ci(k) = -wdf2*ddz2/denom2                                         !%iox
                                                                                          !%iox
                     else   ! if_k_block                                                  !%iox
                                                                                          !%iox
      !  --- ...  slope of bottom layer is introduced                                     !%iox
                        slopx = slope                                                     !%iox
                                                                                          !%iox
      !  --- ...  retrieve the soil water diffusivity and hydraulic conductivity          !%iox
      !           for this layer                                                          !%iox
                                                                                          !%iox
                        !call wdfcnd_gpu &                                                !%ioxC
                        !   !  ---  inputs: &                                             !%ioxC
                        !   (sh2o(nsoil), smcmax, bexp, dksat, dwsat, sicemax, &          !%ioxC
                        !    !  ---  outputs: &                                           !%ioxC
                        !    wdf2, wcnd2 &                                                !%ioxC
                        !    )                                                            !%ioxC
            !  --- ...  calc the ratio of the actual to the max psbl soil h2o content     !%ioxC
                                                                                          !%ioxC
                        factr1 = 0.2/smcmax(i, jj)                                        !%ioxC
                        factr2 = sh2o(i, nsoil, jj)/smcmax(i, jj)                         !%ioxC
                                                                                          !%ioxC
            !  --- ...  prep an expntl coef and calc the soil water diffusivity           !%ioxC
                                                                                          !%ioxC
                        expon = bexp + 2.0                                                !%ioxC
                        wdf2 = dwsat*factr2**expon                                        !%ioxC
                                                                                          !%ioxC
            !  --- ...  frozen soil hydraulic diffusivity.  very sensitive to the         !%ioxC
            !           vertical gradient of unfrozen water. the latter gradient can      !%ioxC
            !           become very extreme in freezing/thawing situations, and given     !%ioxC
            !           the relatively few and thick soil layers, this gradient           !%ioxC
            !           sufferes serious trunction errors yielding erroneously high       !%ioxC
            !           vertical transports of unfrozen water in both directions from     !%ioxC
            !           huge hydraulic diffusivity.therefore, we found we had to          !%ioxC
            !           arbitrarily constrain wdf2                                        !%ioxC
            !                                                                             !%ioxC
            !           version d_10cm:  .......  factr1 = 0.2/smcmax(i, jj)              !%ioxC
            !           weighted approach.......  pablo grunmann, 28_sep_1999.            !%ioxC
                        if (sicemax > 0.0) then                                           !%ioxC
                           vkwgt = 1.0/(1.0 + (500.0*sicemax)**3.0)                       !%ioxC
                           wdf2 = vkwgt*wdf2 + (1.0 - vkwgt)*dwsat*factr1**expon          !%ioxC
                        end if                                                            !%ioxC
                                                                                          !%ioxC
            !  --- ...  reset the expntl coef and calc the hydraulic conductivity         !%ioxC
                        expon = (2.0*bexp) + 3.0                                          !%ioxC
                        wcnd2 = dksat*factr2**expon                                       !%ioxC
                        !end call wdfcnd_gpu                                              !%ioxC
                                                                                          !%iox
      !  --- ...  calc a partial product for later use in calc'ng rhstt                   !%iox
                        dsmdz2 = 0.0                                                      !%iox
                                                                                          !%iox
      !  --- ...  set matrix coef ci to zero                                              !%iox
                                                                                          !%iox
                        ci(k) = 0.0                                                       !%iox
                                                                                          !%iox
                     end if   ! end if_k_block                                            !%iox
                                                                                          !%iox
      !  --- ...  calc rhstt for this layer after calc'ng its numerator                   !%iox
                     numer = wdf2*dsmdz2 + slopx*wcnd2 - wdf*dsmdz - wcnd + et1(k)        !%iox
                     rhstt(k) = numer/(-denom2)                                           !%iox
                                                                                          !%iox
      !  --- ...  calc matrix coefs, ai, and bi for this layer                            !%iox
                                                                                          !%iox
                     ai(k) = -wdf*ddz_1/denom2                                            !%iox
                     bi(k) = -(ai(k) + ci(k))                                             !%iox
                                                                                          !%iox
      !  --- ...  reset values of wdf, wcnd, dsmdz, and ddz_1 for loop to next lyr        !%iox
      !      runoff2(i, jj):  sub-surface or baseflow runoff                              !%iox
                     if (k == nsoil) then                                                 !%iox
                        runoff2(i, jj) = slopx*wcnd2                                      !%iox
                     end if                                                               !%iox
                                                                                          !%iox
                     if (k /= nsoil) then                                                 !%iox
                        wdf = wdf2                                                        !%iox
                        wcnd = wcnd2                                                      !%iox
                        dsmdz = dsmdz2                                                    !%iox
                        ddz_1 = ddz2                                                      !%iox
                     end if                                                               !%iox
                  end do   ! end do_k_loop                                                !%iox
                  !end call srt_gpu                                                       !%iox
                                                                                          !%io
                  !call sstep_gpu &                                                       !%ioy
                  !   !  ---  inputs: &                                                   !%ioy
                  !   (nsoil, sh2o, rhsct, dt, smcmax, cmcmax, zsoil, sice, &             !%ioy
                  !    !  ---  input/outputs: &                                           !%ioy
                  !    cmc, rhstt, ai, bi, ci, &                                          !%ioy
                  !    !  ---  outputs: &                                                 !%ioy
                  !    sh2o, runoff3, smc, &                                              !%ioy
                  !    !  ---  dummys: &                                                  !%ioy
                  !    ciin, rhsttin &                                                    !%ioy
                  !    )                                                                  !%ioy
      !  --- ...  create 'amount' values of variables to be input to the                  !%ioy
      !           tri-diagonal matrix routine.                                            !%ioy
                  !$acc loop seq                                                          !%ioy
                  do k = 1, nsoil                                                         !%ioy
                     rhstt(k) = rhstt(k)*dt                                               !%ioy
                     ai(k) = ai(k)*dt                                                     !%ioy
                     bi(k) = 1.+bi(k)*dt                                                  !%ioy
                     ci(k) = ci(k)*dt                                                     !%ioy
                  end do                                                                  !%ioy
                                                                                          !%ioy
      !  --- ...  copy values for input variables before call to rosr12                   !%ioy
                                                                                          !%ioy
                  !$acc loop seq                                                          !%ioy
                  do k = 1, nsoil                                                         !%ioy
                     rhsttin(k) = rhstt(k)                                                !%ioy
                  end do                                                                  !%ioy
                                                                                          !%ioy
                  !$acc loop seq                                                          !%ioy
                  do k = 1, nsold                                                         !%ioy
                     ciin(k) = ci(k)                                                      !%ioy
                  end do                                                                  !%ioy
                                                                                          !%ioy
      !  --- ...  call rosr12 to solve the tri-diagonal matrix                            !%ioy
                                                                                          !%ioy
                  !call rosr12_gpu &                                                      !%ioyv
                  !   !  ---  inputs: &                                                   !%ioyv
                  !   (nsoil, ai, bi, rhsttin, &                                          !%ioyv
                  !    !  ---  input/outputs: &                                           !%ioyv
                  !    ciin, &                                                            !%ioyv
                  !    !  ---  outputs: &                                                 !%ioyv
                  !    ci, rhstt &                                                        !%ioyv
                  !    )                                                                  !%ioyv
                  !  --- ...  initialize eqn coef ciin for the lowest soil layer          !%ioyv
                                                                                          !%ioyv
                  ciin(nsoil) = 0.0                                                       !%ioyv
                                                                                          !%ioyv
      !  --- ...  solve the coefs for the 1st soil layer                                  !%ioyv
                  ci(1) = -ciin(1)/bi(1)                                                  !%ioyv
                  rhstt(1) = rhsttin(1)/bi(1)                                             !%ioyv
                                                                                          !%ioyv
      !  --- ...  solve the coefs for soil layers 2 thru nsoil                            !%ioyv
                  !$acc loop seq                                                          !%ioyv
                  do k = 2, nsoil                                                         !%ioyv
                     ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                     !%ioyv
                     rhstt(k) = (rhsttin(k) - ai(k)*rhstt(k - 1)) &                       !%ioyv
                                *(1.0/(bi(k) + ai(k)*ci(k - 1)))                          !%ioyv
                  end do                                                                  !%ioyv
                                                                                          !%ioyv
      !  --- ...  set ci to rhstt for lowest soil layer                                   !%ioyv
                  ci(nsoil) = rhstt(nsoil)                                                !%ioyv
                                                                                          !%ioyv
      !  --- ...  adjust ci for soil layers 2 thru nsoil                                  !%ioyv
                                                                                          !%ioyv
                  !$acc loop seq                                                          !%ioyv
                  do k = 2, nsoil                                                         !%ioyv
                     kk = nsoil - k + 1                                                   !%ioyv
                     ci(kk) = ci(kk)*ci(kk + 1) + rhstt(kk)                               !%ioyv
                  end do                                                                  !%ioyv
                  !end call rosr12_gpu                                                    !%ioyv
                                                                                          !%ioy
                                                                                          !%ioy
      !  --- ...  sum the previous smc value and the matrix solution to get               !%ioy
      !           a new value.  min allowable value of smc will be 0.02.                  !%ioy
      !      runoff3(i, jj): runoff within soil layers                                    !%ioy
                                                                                          !%ioy
                  wplus = 0.0                                                             !%ioy
                  runoff3(i, jj) = 0.0                                                    !%ioy
                  ddz = -zsoil(1)                                                         !%ioy
                                                                                          !%ioy
                  !$acc loop seq                                                          !%ioy
                  do k = 1, nsoil                                                         !%ioy
                     if (k /= 1) then                                                     !%ioy
                        ddz = zsoil(k - 1) - zsoil(k)                                     !%ioy
                     end if                                                               !%ioy
                     sh2o(i, k, jj) = sh2o(i, k, jj) + ci(k) + wplus/ddz                  !%ioy
                                                                                          !%ioy
                     stot = sh2o(i, k, jj) + sice(k)                                      !%ioy
                     if (stot > smcmax(i, jj)) then                                       !%ioy
                        if (k == 1) then                                                  !%ioy
                           ddz = -zsoil(1)                                                !%ioy
                        else                                                              !%ioy
                           kk11 = k - 1                                                   !%ioy
                           ddz = -zsoil(k) + zsoil(kk11)                                  !%ioy
                        end if                                                            !%ioy
                        wplus = (stot - smcmax(i, jj))*ddz                                !%ioy
                     else                                                                 !%ioy
                        wplus = 0.0                                                       !%ioy
                     end if                                                               !%ioy
                                                                                          !%ioy
                     smc(i, k, jj) = max(min(stot, smcmax(i, jj)), 0.02)                  !%ioy
                     sh2o(i, k, jj) = max(smc(i, k, jj) - sice(k), 0.0)                   !%ioy
                  end do                                                                  !%ioy
                  runoff3(i, jj) = wplus                                                  !%ioy
                                                                                          !%ioy
      !  --- ...  update canopy water content/interception (cmc(i, jj)).  convert         !%ioy
      !           rhsct to an 'amount' value and add to previous cmc(i, jj) value to      !%ioy
      !           get new cmc(i, jj).                                                     !%ioy
                                                                                          !%ioy
                  cmc(i, jj) = cmc(i, jj) + dt*rhsct                                      !%ioy
                  if (cmc(i, jj) < 1.e-20) cmc(i, jj) = 0.0                               !%ioy
                  cmc(i, jj) = min(cmc(i, jj), cmcmax)                                    !%ioy
                  !end call sstep_gpu                                                     !%ioy
                                                                                          !%io
               end if                                                                     !%io
               !end call smflx_gpu                                                        !%io
                                                                                          !%i
            end if                                                                        !%i
                                                                                          !%i
!  --- ...  before call shflx in this snowpack case, set zz1 and yy arguments to          !%i
!           special values that ensure that ground heat flux calculated in shflx          !%i
!           matches that already computer for below the snowpack, thus the sfc            !%i
!           heat flux to be computed in shflx will effectively be the flux at the         !%i
!           snow top surface.  t11 is a dummy arguement so we will not use the            !%i
!           skin temp value as revised by shflx.                                          !%i
                                                                                          !%i
            zz1 = 1.0                                                                     !%i
            yy = stc(i, 1, jj) - 0.5*ssoil(i, jj)*zsoil(1)*zz1/df1                        !%i
            t11 = t1(i, jj)                                                               !%i
                                                                                          !%i
!  --- ...  shflx will calc/update the soil temps.  note:  the sub-sfc heat flux          !%i
!           (ssoil1) and the skin temp (t11) output from this shflx call are not          !%i
!           used  in any subsequent calculations. rather, they are dummy variables        !%i
!           here in the snopac case, since the skin temp and sub-sfc heat flux are        !%i
!           updated instead near the beginning of the call to snopac.                     !%i
                                                                                          !%i
            !call shflx_gpu &                                                             !%in
            !   !  ---  inputs: &                                                         !%in
            !   (nsoil, smc, smcmax, dt, yy, zz1, zsoil, zbot, &                          !%in
            !    psisat, bexp, df1, ice, quartz, csoil, vegtyp, ivegsrc,&                 !%in
            !    !  ---  input/outputs: &                                                 !%in
            !    stc, t11, tbot, sh2o, df1, &                                             !%in
            !    !  ---  outputs: &                                                       !%in
            !    ssoil1, &                                                                !%in
            !    !  --- dummys: &                                                         !%in
            !    ciin, rhstsin, ai, bi, ci, rhsts, stcf, stsoil &                         !%in
            !    )                                                                        !%in
            oldt1 = t11                                                                   !%in
            !$acc loop seq                                                                !%in
            do i1 = 1, nsoil                                                              !%in
               stsoil(i1) = stc(i, i1, jj)                                                !%in
            end do                                                                        !%in
                                                                                          !%in
!  --- ...  hrt routine calcs the right hand side of the soil temp dif eqn                !%in
            if (ice /= 0) then                                                            !%in
                                                                                          !%in
!  --- ...  sea-ice case, glacial-ice case                                                !%in
                                                                                          !%in
               !call hrtice_gpu &                                                         !%int
               !   !  ---  inputs: &                                                      !%int
               !   (nsoil, stc, zsoil, yy, zz1, df1, ice, &                               !%int
               !    !  ---  input/outputs: &                                              !%int
               !    tbot, &                                                               !%int
               !    !  ---  outputs: &                                                    !%int
               !    rhsts, ai, bi, ci &                                                   !%int
               !    )                                                                     !%int
   !  --- ...  set a nominal universal value of the sea-ice specific heat capacity,       !%int
   !           hcpct = 1880.0*917.0 = 1.72396e+6 (source:  fei chen, 1995)                !%int
   !           set bottom of sea-ice pack temperature: tbot(i, jj) = 271.16               !%int
   !           set a nominal universal value of glacial-ice specific heat capacity,       !%int
   !           hcpct = 2100.0*900.0 = 1.89000e+6 (source:  bob grumbine, 2005)            !%int
   !           tbot(i, jj) passed in as argument, value from global data set              !%int
               if (ice == 1) then                                                         !%int
   !  --- ...  sea-ice                                                                    !%int
                  hcpct = 1.72396e+6                                                      !%int
                  tbot(i, jj) = 271.16                                                    !%int
               else                                                                       !%int
   !  --- ...  glacial-ice                                                                !%int
                  hcpct = 1.89000e+6                                                      !%int
               end if                                                                     !%int
                                                                                          !%int
   !  --- ...  the input argument df1 is a universally constant value of sea-ice          !%int
   !           and glacial-ice thermal diffusivity, set in sflx as df1 = 2.2.             !%int
                                                                                          !%int
   !  --- ...  set ice pack depth.  use tbot(i, jj) as ice pack lower boundary            !%int
   !           temperature (that of unfrozen sea water at bottom of sea ice pack).        !%int
   !           assume ice pack is of n=nsoil layers spanning a uniform constant ice       !%int
   !           pack thickness as defined by zsoil(nsoil) in routine sflx.                 !%int
   !           if glacial-ice, set zbot_1 = -25 meters                                    !%int
               if (ice == 1) then                                                         !%int
   !  --- ...  sea-ice                                                                    !%int
                  zbot_1 = zsoil(nsoil)                                                   !%int
               else                                                                       !%int
   !  --- ...  glacial-ice                                                                !%int
                  zbot_1 = -25.0                                                          !%int
               end if                                                                     !%int
                                                                                          !%int
   !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer              !%int
               ddz_3 = 1.0/(-0.5*zsoil(2))                                                !%int
               ai(1) = 0.0                                                                !%int
               ci(1) = (df1*ddz_3)/(zsoil(1)*hcpct)                                       !%int
               bi(1) = -ci(1) + df1/(0.5*zsoil(1)*zsoil(1)*hcpct*zz1)                     !%int
                                                                                          !%int
   !  --- ...  calc the vertical soil temp gradient btwn the top and 2nd soil             !%int
   !           layers. recalc/adjust the soil heat flux.  use the gradient and            !%int
   !           flux to calc rhsts for the top soil layer.                                 !%int
                                                                                          !%int
               dtsdz = (stc(i, 1, jj) - stc(i, 2, jj))/(-0.5*zsoil(2))                    !%int
               ssoil_1 = df1*(stc(i, 1, jj) - yy)/(0.5*zsoil(1)*zz1)                      !%int
               rhsts(1) = (df1*dtsdz - ssoil_1)/(zsoil(1)*hcpct)                          !%int
                                                                                          !%int
   !  --- ...  initialize ddz2_2                                                          !%int
                                                                                          !%int
               ddz2_2 = 0.0                                                               !%int
                                                                                          !%int
   !  --- ...  loop thru the remaining soil layers, repeating the above process           !%int
               !$acc loop seq                                                             !%int
               do k = 2, nsoil                                                            !%int
                                                                                          !%int
                  if (k /= nsoil) then                                                    !%int
                                                                                          !%int
   !  --- ...  calc the vertical soil temp gradient thru this layer.                      !%int
                     denom_5 = 0.5*(zsoil(k - 1) - zsoil(k + 1))                          !%int
                     dtsdz2 = (stc(i, k, jj) - stc(i, k + 1, jj))/denom_5                 !%int
                                                                                          !%int
   !  --- ...  calc the matrix coef, ci, after calc'ng its partial product.               !%int
                                                                                          !%int
                     ddz2_2 = 2.0/(zsoil(k - 1) - zsoil(k + 1))                           !%int
                     ci(k) = -df1*ddz2_2/((zsoil(k - 1) - zsoil(k))*hcpct)                !%int
                                                                                          !%int
                  else                                                                    !%int
                                                                                          !%int
   !  --- ...  calc the vertical soil temp gradient thru the lowest layer.                !%int
                     dtsdz2 = (stc(i, k, jj) - tbot(i, jj)) &                             !%int
                              /(0.5*(zsoil(k - 1) + zsoil(k)) - zbot_1)                   !%int
                                                                                          !%int
   !  --- ...  set matrix coef, ci to zero.                                               !%int
                                                                                          !%int
                     ci(k) = 0.0                                                          !%int
                                                                                          !%int
                  end if   ! end if_k_block                                               !%int
                                                                                          !%int
   !  --- ...  calc rhsts for this layer after calc'ng a partial product.                 !%int
                  denom_5 = (zsoil(k) - zsoil(k - 1))*hcpct                               !%int
                  rhsts(k) = (df1*dtsdz2 - df1*dtsdz)/denom_5                             !%int
                                                                                          !%int
   !  --- ...  calc matrix coefs, ai, and bi for this layer.                              !%int
                                                                                          !%int
                  ai(k) = -df1*ddz_3/((zsoil(k - 1) - zsoil(k))*hcpct)                    !%int
                  bi(k) = -(ai(k) + ci(k))                                                !%int
                                                                                          !%int
   !  --- ...  reset values of dtsdz and ddz_3 for loop to next soil lyr.                 !%int
                                                                                          !%int
                  dtsdz = dtsdz2                                                          !%int
                  ddz_3 = ddz2_2                                                          !%int
                                                                                          !%int
               end do   ! end do_k_loop                                                   !%int
               !end call hrtice_gpu                                                       !%int
                                                                                          !%in
               !call hstep_gpu &                                                          !%inu
               !   !  ---  inputs: &                                                      !%inu
               !   (nsoil, stc, dt, &                                                     !%inu
               !    !  ---  input/outputs: &                                              !%inu
               !    rhsts, ai, bi, ci, &                                                  !%inu
               !    !  ---  outputs: &                                                    !%inu
               !    stcf, &                                                               !%inu
               !    !  ---  dummys: &                                                     !%inu
               !    ciin, rhstsin &                                                       !%inu
               !    )                                                                     !%inu
   !  --- ...  create finite difference values for use in rosr12 routine                  !%inu
               !$acc loop seq                                                             !%inu
               do k = 1, nsoil                                                            !%inu
                  rhsts(k) = rhsts(k)*dt                                                  !%inu
                  ai(k) = ai(k)*dt                                                        !%inu
                  bi(k) = 1.0 + bi(k)*dt                                                  !%inu
                  ci(k) = ci(k)*dt                                                        !%inu
               end do                                                                     !%inu
                                                                                          !%inu
   !  --- ...  copy values for input variables before call to rosr12                      !%inu
                                                                                          !%inu
               !$acc loop seq                                                             !%inu
               do k = 1, nsoil                                                            !%inu
                  rhstsin(k) = rhsts(k)                                                   !%inu
               end do                                                                     !%inu
                                                                                          !%inu
               !$acc loop seq                                                             !%inu
               do k = 1, nsold                                                            !%inu
                  ciin(k) = ci(k)                                                         !%inu
               end do                                                                     !%inu
                                                                                          !%inu
   !  --- ...  solve the tri-diagonal matrix equation                                     !%inu
                                                                                          !%inu
               !call rosr12_gpu &                                                         !%inuv
               !   !  ---  inputs: &                                                      !%inuv
               !   (nsoil, ai, bi, rhstsin, &                                             !%inuv
               !    !  ---  input/outputs: &                                              !%inuv
               !    ciin, &                                                               !%inuv
               !    !  ---  outputs: &                                                    !%inuv
               !    ci, rhsts &                                                           !%inuv
               !    )                                                                     !%inuv
   !  --- ...  initialize eqn coef ciin for the lowest soil layer                         !%inuv
                                                                                          !%inuv
               ciin(nsoil) = 0.0                                                          !%inuv
                                                                                          !%inuv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%inuv
               ci(1) = -ciin(1)/bi(1)                                                     !%inuv
               rhsts(1) = rhstsin(1)/bi(1)                                                !%inuv
                                                                                          !%inuv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%inuv
               !$acc loop seq                                                             !%inuv
               do k = 2, nsoil                                                            !%inuv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%inuv
                  rhsts(k) = (rhstsin(k) - ai(k)*rhsts(k - 1)) &                          !%inuv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%inuv
               end do                                                                     !%inuv
                                                                                          !%inuv
   !  --- ...  set ci to rhsts for lowest soil layer                                      !%inuv
               ci(nsoil) = rhsts(nsoil)                                                   !%inuv
                                                                                          !%inuv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%inuv
                                                                                          !%inuv
               !$acc loop seq                                                             !%inuv
               do k = 2, nsoil                                                            !%inuv
                  kk = nsoil - k + 1                                                      !%inuv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhsts(kk)                                  !%inuv
               end do                                                                     !%inuv
               !end call rosr12_gpu                                                       !%inuv
                                                                                          !%inu
   !  --- ...  calc/update the soil temps using matrix solution                           !%inu
                                                                                          !%inu
               !$acc loop seq                                                             !%inu
               do k = 1, nsoil                                                            !%inu
                  stcf(k) = stc(i, k, jj) + ci(k)                                         !%inu
               end do                                                                     !%inu
               !end call hstep_gpu                                                        !%inu
                                                                                          !%in
            else                                                                          !%in
                                                                                          !%in
!  --- ...  land-mass case                                                                !%in
                                                                                          !%in
               !call hrt_gpu &                                                            !%ins
               !   !  ---  inputs: &                                                      !%ins
               !   (nsoil, stc, smc, smcmax, zsoil, yy, zz1, tbot, &                      !%ins
               !    zbot, psisat, dt, bexp, df1, quartz, csoil, vegtyp, ivegsrc,&         !%ins
               !    !  ---  input/outputs: &                                              !%ins
               !    sh2o, df1, &                                                          !%ins
               !    !  ---  outputs: &                                                    !%ins
               !    rhsts, ai, bi, ci &                                                   !%ins
               !    )                                                                     !%ins
               csoil_loc = csoil                                                          !%ins
               if (ivegsrc .ge. 1) then                                                   !%ins
   !urban                                                                                 !%ins
                  if (vegtyp(i, jj) == 13) then                                           !%ins
                     csoil_loc = 3.0e6                                                    !%ins
                  end if                                                                  !%ins
               end if                                                                     !%ins
                                                                                          !%ins
   !  --- ...  initialize logical for soil layer temperature averaging.                   !%ins
                                                                                          !%ins
               itavg = .true.                                                             !%ins
   !     itavg = .false.                                                                  !%ins
                                                                                          !%ins
   !  ===  begin section for top soil layer                                               !%ins
                                                                                          !%ins
   !  --- ...  calc the heat capacity of the top soil layer                               !%ins
               hcpct_1 = sh2o(i, 1, jj)*cph2o2 + (1.0 - smcmax(i, jj))*csoil_loc &        !%ins
                       + (smcmax(i, jj) - smc(i, 1, jj))*cp2 + (smc(i, 1, jj) &           !%ins
                       - sh2o(i, 1, jj))*cpice1                                           !%ins
                                                                                          !%ins
   !  --- ...  calc the matrix coefficients ai, bi, and ci for the top layer              !%ins
                                                                                          !%ins
               ddz_2 = 1.0/(-0.5*zsoil(2))                                                !%ins
               ai(1) = 0.0                                                                !%ins
               ci(1) = (df1*ddz_2)/(zsoil(1)*hcpct_1)                                     !%ins
               bi(1) = -ci(1) + df1/(0.5*zsoil(1)*zsoil(1)*hcpct_1*zz1)                   !%ins
                                                                                          !%ins
   !  --- ...  calculate the vertical soil temp gradient btwn the 1st and 2nd soil        !%ins
   !           layers.  then calculate the subsurface heat flux. use the temp             !%ins
   !           gradient and subsfc heat flux to calc "right-hand side tendency            !%ins
   !           terms", or "rhsts", for top soil layer.                                    !%ins
                                                                                          !%ins
               dtsdz_1 = (stc(i, 1, jj) - stc(i, 2, jj))/(-0.5*zsoil(2))                  !%ins
               ssoil_2 = df1*(stc(i, 1, jj) - yy)/(0.5*zsoil(1)*zz1)                      !%ins
               rhsts(1) = (df1*dtsdz_1 - ssoil_2)/(zsoil(1)*hcpct_1)                      !%ins
                                                                                          !%ins
   !  --- ...  next capture the vertical difference of the heat flux at top and           !%ins
   !           bottom of first soil layer for use in heat flux constraint applied to      !%ins
   !           potential soil freezing/thawing in routine snksrc.                         !%ins
                                                                                          !%ins
               qtot = ssoil_2 - df1*dtsdz_1                                               !%ins
                                                                                          !%ins
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%ins
   !           set temp "tsurf" at top of soil column (for use in freezing soil           !%ins
   !           physics later in subroutine snksrc).  if snowpack content is               !%ins
   !           zero, then tsurf expression below gives tsurf = skin temp.  if             !%ins
   !           snowpack is nonzero (hence argument zz1=1), then tsurf expression          !%ins
   !           below yields soil column top temperature under snowpack.  then             !%ins
   !           calculate temperature at bottom interface of 1st soil layer for use        !%ins
   !           later in subroutine snksrc                                                 !%ins
               if (itavg) then                                                            !%ins
                  tsurf = (yy + (zz1 - 1)*stc(i, 1, jj))/zz1                              !%ins
                                                                                          !%ins
                  !call tbnd_gpu &                                                        !%insz
                  !   !  ---  inputs: &                                                   !%insz
                  !   (stc(1), stc(2), zsoil, zbot, 1, nsoil, &                           !%insz
                  !    !  ---  outputs: &                                                 !%insz
                  !    tbk &                                                              !%insz
                  !    )                                                                  !%insz
      !  --- ...  use surface temperature on the top of the first layer                   !%insz
                  k = 1                                                                   !%insz
                  if (k == 1) then                                                        !%insz
                     zup = 0.0                                                            !%insz
                  else                                                                    !%insz
                     zup = zsoil(k - 1)                                                   !%insz
                  end if                                                                  !%insz
                                                                                          !%insz
      !  --- ...  use depth of the constant bottom temperature when interpolate           !%insz
      !           temperature into the last layer boundary                                !%insz
                                                                                          !%insz
                  if (k == nsoil) then                                                    !%insz
                     zb = 2.0*zbot - zsoil(k)                                             !%insz
                  else                                                                    !%insz
                     zb = zsoil(k + 1)                                                    !%insz
                  end if                                                                  !%insz
                                                                                          !%insz
      !  --- ...  linear interpolation between the average layer temperatures             !%insz
                  tbk = stc(i, k, jj) + (stc(i, k + 1, jj) - stc(i, k, jj)) &             !%insz
                        *(zup - zsoil(k))/(zup - zb)                                      !%insz
                  !end call tbnd_gpu                                                      !%insz
                                                                                          !%ins
               end if                                                                     !%ins
                                                                                          !%ins
   !  --- ...  calculate frozen water content in 1st soil layer.                          !%ins
               sice_1 = smc(i, 1, jj) - sh2o(i, 1, jj)                                    !%ins
                                                                                          !%ins
   !  --- ...  if frozen water present or any of layer-1 mid-point or bounding            !%ins
   !           interface temperatures below freezing, then call snksrc to                 !%ins
   !           compute heat source/sink (and change in frozen water content)              !%ins
   !           due to possible soil water phase change                                    !%ins
                                                                                          !%ins
               if ((sice_1 > 0.0) .or. (tsurf < tfreez) .or. &                            !%ins
                   (stc(i, 1, jj) < tfreez) .or. (tbk < tfreez)) then                     !%ins
                  if (itavg) then                                                         !%ins
                                                                                          !%ins
                     !call tmpavg_gpu &                                                   !%insA
                     !   !  ---  inputs: &                                                !%insA
                     !   (tsurf, stc(1), tbk, zsoil, nsoil, 1, &                          !%insA
                     !    !  ---  outputs: &                                              !%insA
                     !    tavg &                                                          !%insA
                     !    )                                                               !%insA
                     k = 1                                                                !%insA
                     if (k == 1) then                                                     !%insA
                        dz = -zsoil(1)                                                    !%insA
                     else                                                                 !%insA
                        dz = zsoil(k - 1) - zsoil(k)                                      !%insA
                     end if                                                               !%insA
                     dzh = dz*0.5                                                         !%insA
                                                                                          !%insA
                     if (tsurf < tfreez) then                                             !%insA
                        if (stc(i, k, jj) < tfreez) then                                  !%insA
                           if (tbk < tfreez) then ! tsurf, stc(k), tbk < t0               !%insA
                              tavg = (tsurf + 2.0*stc(i, k, jj) + tbk)/4.0                !%insA
                           else ! tsurf & stc(k) < t0,  tbk >= t0                         !%insA
                              x0 = (tfreez - stc(i, k, jj))*dzh/(tbk - stc(i, k, jj))     !%insA
                              tavg = 0.5*(tsurf*dzh + stc(i, k, jj)*(dzh + x0) &          !%insA
                                     + tfreez*(2.*dzh - x0))/dz                           !%insA
                           end if                                                         !%insA
                        else                                                              !%insA
                           if (tbk < tfreez) then ! tsurf < t0, stc(k) >= t0, tbk < t0    !%insA
                              xup = (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)          !%insA
                              xdn = dzh - (tfreez - stc(i, k, jj))*dzh &                  !%insA
                                    /(tbk - stc(i, k, jj))                                !%insA
                              tavg = 0.5*(tsurf*xup + tfreez*(2.*dz - xup - xdn) &        !%insA
                                     + tbk*xdn)/dz                                        !%insA
                           else ! tsurf < t0, stc(k) >= t0, tbk >= t0                     !%insA
                              xup = (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)          !%insA
                              tavg = 0.5*(tsurf*xup + tfreez*(2.*dz - xup))/dz            !%insA
                           end if                                                         !%insA
                        end if                                                            !%insA
                                                                                          !%insA
                     else    ! if_tup_block                                               !%insA
                        if (stc(i, k, jj) < tfreez) then                                  !%insA
                           if (tbk < tfreez) then ! tsurf >= t0, stc(k) < t0, tbk < t0    !%insA
                              xup = dzh - (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)    !%insA
                              tavg = 0.5*(tfreez*(dz - xup) + stc(i, k, jj) &             !%insA
                                     *(dzh + xup) + tbk*dzh)/dz                           !%insA
                                                                                          !%insA
                           else ! tsurf >= t0, stc(k) < t0, tbk >= t0                     !%insA
                              xup = dzh - (tfreez - tsurf)*dzh/(stc(i, k, jj) - tsurf)    !%insA
                              xdn = (tfreez - stc(i, k, jj))*dzh/(tbk - stc(i, k, jj))    !%insA
                              tavg = 0.5*(tfreez*(2.*dz - xup - xdn) &                    !%insA
                                     + stc(i, k, jj)*(xup + xdn))/dz                      !%insA
                           end if                                                         !%insA
                        else                                                              !%insA
                           if (tbk < tfreez) then ! tsurf >= t0, stc(k) >= t0, tbk < t0   !%insA
                              xdn = dzh - (tfreez - stc(i, k, jj))*dzh &                  !%insA
                                    /(tbk - stc(i, k, jj))                                !%insA
                              tavg = (tfreez*(dz - xdn) + 0.5*(tfreez + tbk)*xdn)/dz      !%insA
                           else ! tsurf >= t0, stc(k) >= t0, tbk >= t0                    !%insA
                              tavg = (tsurf + 2.0*stc(i, k, jj) + tbk)/4.0                !%insA
                           end if                                                         !%insA
                        end if                                                            !%insA
                                                                                          !%insA
                     end if   ! end if_tup_block                                          !%insA
                     !end call tmpavg_gpu                                                 !%insA
                                                                                          !%ins
                  else                                                                    !%ins
                     tavg = stc(i, 1, jj)                                                 !%ins
                                                                                          !%ins
                  end if   ! end if_itavg_block                                           !%ins
                                                                                          !%ins
                  !call snksrc_gpu &                                                      !%insw
                  !   !  ---  inputs: &                                                   !%insw
                  !   (nsoil, 1, tavg, smc(1), smcmaxt, bexp, dt, &                       !%insw
                  !    qtot, zsoil, ivegsrc, vegtyp   !    !  ---  input/outputs: &       !%insw
                  !    sh2o(1), df1, &                                                    !%insw
                  !    !  ---  outputs: &                                                 !%insw
                  !    tsnsr &                                                            !%insw
                  !    )                                                                  !%insw
                  k = 1                                                                   !%insw
                  if (ivegsrc .ge. 1) then                                                !%insw
                     if (vegtyp(i, jj) == 13) then                                        !%insw
                        df1 = 3.24                                                        !%insw
                     end if                                                               !%insw
                  end if                                                                  !%insw
      !                                                                                   !%insw
      !===> ...  begin here                                                               !%insw
      !                                                                                   !%insw
                  if (k == 1) then                                                        !%insw
                     dz_1 = -zsoil(1)                                                     !%insw
                  else                                                                    !%insw
                     dz_1 = zsoil(k - 1) - zsoil(k)                                       !%insw
                  end if                                                                  !%insw
                                                                                          !%insw
      !  --- ...  via function frh2o, compute potential or 'equilibrium' unfrozen         !%insw
      !           supercooled free water for given soil type and soil layer temperature.  !%insw
      !           function frh20 invokes eqn (17) from v. koren et al (1999, jgr, vol.    !%insw
      !           104, pg 19573).  (aside:  latter eqn in journal in centigrade units.    !%insw
      !           routine frh2o use form of eqn in kelvin units.)                         !%insw
                                                                                          !%insw
      !     free = frh2o( tavg,smc(k),sh2o(k),smcmax,bexp,psisat )                        !%insw
                                                                                          !%insw
                  !call frh2o_gpu &                                                       !%inswr
                  !   !  ---  inputs: &                                                   !%inswr
                  !   (tavg, smc(k), sh2o(k), smcmax, bexp, psisat, &                     !%inswr
                  !    !  ---  outputs: &                                                 !%inswr
                  !    free &                                                             !%inswr
                  !    )                                                                  !%inswr
                                                                                          !%inswr
      !                                                                                   !%inswr
      !===> ...  begin here                                                               !%inswr
      !                                                                                   !%inswr
      !  --- ...  limits on parameter b: b < 5.5  (use parameter blim)                    !%inswr
      !           simulations showed if b > 5.5 unfrozen water content is                 !%inswr
      !           non-realistically high at very low temperatures.                        !%inswr
                  bx = bexp                                                               !%inswr
                  if (bexp > blim) bx = blim                                              !%inswr
                                                                                          !%inswr
      !  --- ...  initializing iterations counter and iterative solution flag.            !%inswr
                                                                                          !%inswr
                  nlog = 0                                                                !%inswr
                  kcount = 0                                                              !%inswr
                                                                                          !%inswr
      !  --- ...  if temperature not significantly below freezing (t0), sh2o(k) = smc(k)  !%inswr
                  if (tavg > (tfreez - 1.e-3)) then                                       !%inswr
                     free = smc(i, k, jj)                                                 !%inswr
                                                                                          !%inswr
                  else                                                                    !%inswr
                     if (ck /= 0.0) then                                                  !%inswr
                                                                                          !%inswr
      !  --- ...  option 1: iterated solution for nonzero ck                              !%inswr
      !                     in koren et al, jgr, 1999, eqn 17                             !%inswr
                                                                                          !%inswr
      !  --- ...  initial guess for swl (frozen content)                                  !%inswr
                        swl = smc(i, k, jj) - sh2o(i, k, jj)                              !%inswr
                                                                                          !%inswr
      !  --- ...  keep within bounds.                                                     !%inswr
                                                                                          !%inswr
                        swl = max(min(swl, smc(i, k, jj) - 0.02), 0.0)                    !%inswr
                                                                                          !%inswr
      !  --- ...  start of iterations                                                     !%inswr
                        do while ((nlog < 10) .and. (kcount == 0))                        !%inswr
                           nlog = nlog + 1                                                !%inswr
                                                                                          !%inswr
                           df = dlog((psisat*gs2/lsubf)*((1.0 + ck*swl)**2.0) &           !%inswr
                                     *(smcmax(i, jj)/(smc(i, k, jj) - swl))**bx) &        !%inswr
                                     - dlog(-(tavg - tfreez)/tavg)                        !%inswr
                                                                                          !%inswr
                           denom_1 = 2.0*ck/(1.0 + ck*swl) + bx/(smc(i, k, jj) - swl)     !%inswr
                           swlk = swl - df/denom_1                                        !%inswr
                                                                                          !%inswr
      !  --- ...  bounds useful for mathematical solution.                                !%inswr
                                                                                          !%inswr
                           swlk = max(min(swlk, smc(i, k, jj) - 0.02), 0.0)               !%inswr
                                                                                          !%inswr
      !  --- ...  mathematical solution bounds applied.                                   !%inswr
                                                                                          !%inswr
                           dswl = abs(swlk - swl)                                         !%inswr
                           swl = swlk                                                     !%inswr
                                                                                          !%inswr
      !  --- ...  if more than 10 iterations, use explicit method (ck=0 approx.)          !%inswr
      !           when dswl less or eq. error, no more iterations required.               !%inswr
                           if (dswl <= error) then                                        !%inswr
                              kcount = kcount + 1                                         !%inswr
                           end if                                                         !%inswr
                        end do   !  end do_while_loop                                     !%inswr
                                                                                          !%inswr
      !  --- ...  bounds applied within do-block are valid for physical solution.         !%inswr
                        free = smc(i, k, jj) - swl                                        !%inswr
                                                                                          !%inswr
                     end if   ! end if_ck_block                                           !%inswr
                                                                                          !%inswr
      !  --- ...  option 2: explicit solution for flerchinger eq. i1.e. ck=0              !%inswr
      !                     in koren et al., jgr, 1999, eqn 17                            !%inswr
      !           apply physical bounds to flerchinger solution                           !%inswr
                     if (kcount == 0) then                                                !%inswr
                        fk = (((lsubf/(gs2*(-psisat))) &                                  !%inswr
                               *((tavg - tfreez)/tavg))**(-1/bx))*smcmax(i, jj)           !%inswr
                                                                                          !%inswr
                        fk = max(fk, 0.02)                                                !%inswr
                                                                                          !%inswr
                        free = min(fk, smc(i, k, jj))                                     !%inswr
                     end if                                                               !%inswr
                                                                                          !%inswr
                  end if   ! end if_tkelv_block                                           !%inswr
                  !end call frh2o_gpu                                                     !%inswr
                                                                                          !%insw
                                                                                          !%insw
         !  --- ...  in next block of code, invoke eqn 18 of v. koren et al (1999,        !%insw
         !           jgr, vol. 104, pg 19573.)  that is, first estimate the new           !%insw
         !           amountof liquid water, 'xh2o', implied by the sum of (1) the         !%insw
         !           liquid water at the begin of current time step, and (2) the          !%insw
         !           freeze of thaw change in liquidwater implied by the heat flux        !%insw
         !           'qtot' passed in from routine hrt.second, determine if xh2o          !%insw
         !           needs to be bounded by 'free' (equil amt) orif 'free' needs to       !%insw
         !           be bounded by xh2o.                                                  !%insw
                                                                                          !%insw
                  xh2o = sh2o(i, k, jj) + qtot*dt/(dh2o*lsubf*dz_1)                       !%insw
                                                                                          !%insw
      !  --- ...  first, if freezing and remaining liquid less than lower bound, then     !%insw
      !           reduce extent of freezing, thereby letting some or all of heat flux     !%insw
      !           qtot cool the soil temp later in routine hrt.                           !%insw
                  if (xh2o < sh2o(i, k, jj) .and. xh2o < free) then                       !%insw
                     if (free > sh2o(i, k, jj)) then                                      !%insw
                        xh2o = sh2o(i, k, jj)                                             !%insw
                     else                                                                 !%insw
                        xh2o = free                                                       !%insw
                     end if                                                               !%insw
                  end if                                                                  !%insw
                                                                                          !%insw
      !  --- ...  second, if thawing and the increase in liquid water greater than        !%insw
      !           upper bound, then reduce extent of thaw, thereby letting some or        !%insw
      !           all of heat flux qtot warm the soil temp later in routine hrt.          !%insw
                                                                                          !%insw
                  if (xh2o > sh2o(i, k, jj) .and. xh2o > free) then                       !%insw
                     if (free < sh2o(i, k, jj)) then                                      !%insw
                        xh2o = sh2o(i, k, jj)                                             !%insw
                     else                                                                 !%insw
                        xh2o = free                                                       !%insw
                     end if                                                               !%insw
                  end if                                                                  !%insw
                                                                                          !%insw
                  xh2o = max(min(xh2o, smc(i, k, jj)), 0.0)                               !%insw
                                                                                          !%insw
      !  --- ...  calculate phase-change heat source/sink term for use in routine hrt     !%insw
      !           and update liquid water to reflcet final freeze/thaw increment.         !%insw
                                                                                          !%insw
                  tsnsr = -dh2o*lsubf*dz_1*(xh2o - sh2o(i, k, jj))/dt                     !%insw
                  sh2o(i, k, jj) = xh2o                                                   !%insw
                  !end call snksrc_gpu                                                    !%insw
                  rhsts(1) = rhsts(1) - tsnsr/(zsoil(1)*hcpct_1)                          !%ins
                                                                                          !%ins
               end if   ! end if_sice_block                                               !%ins
                                                                                          !%ins
   !  ===  this ends section for top soil layer.                                          !%ins
                                                                                          !%ins
   !  --- ...  initialize ddz2_1                                                          !%ins
                                                                                          !%ins
               ddz2_1 = 0.0                                                               !%ins
                                                                                          !%ins
   !  --- ...  loop thru the remaining soil layers, repeating the above process           !%ins
   !           (except subsfc or "ground" heat flux not repeated in lower layers)         !%ins
               df1k = df1                                                                 !%ins
                                                                                          !%ins
               !$acc loop seq                                                             !%ins
               do k = 2, nsoil                                                            !%ins
                                                                                          !%ins
   !  --- ...  calculate heat capacity for this soil layer.                               !%ins
                  hcpct_1 = sh2o(i, k, jj)*cph2o2 + (1.0 - smcmax(i, jj))*csoil_loc &     !%ins
                          + (smcmax(i, jj) - smc(i, k, jj))*cp2 + (smc(i, k, jj) &        !%ins
                          - sh2o(i, k, jj))*cpice1                                        !%ins
                                                                                          !%ins
                  if (k /= nsoil) then                                                    !%ins
                                                                                          !%ins
   !  --- ...  this section for layer 2 or greater, but not last layer.                   !%ins
   !           calculate thermal diffusivity for this layer.                              !%ins
                                                                                          !%ins
                     !call tdfcnd_gpu &                                                   !%insl
                     !   !  ---  inputs: &                                                !%insl
                     !   (smc(k), quartz, smcmax, sh2o(k), &                              !%insl
                     !    !  ---  outputs: &                                              !%insl
                     !    df1n &                                                          !%insl
                     !    )                                                               !%insl
         !  --- ...  if the soil has any moisture content compute a partial sum/product   !%insl
         !           otherwise use a constant value which works well with most soils      !%insl
                                                                                          !%insl
         !  --- ...  saturation ratio:                                                    !%insl
                     satratio = smc(i, k, jj)/smcmax(i, jj)                               !%insl
                                                                                          !%insl
         !  --- ...  parameters  w/(m.k)                                                  !%insl
                     thkice = 2.2                                                         !%insl
                     thkw = 0.57                                                          !%insl
                     thko = 2.0                                                           !%insl
         !     if (quartz <= 0.2) thko = 3.0                                              !%insl
                     thkqtz = 7.7                                                         !%insl
                                                                                          !%insl
         !  --- ...  solids' conductivity                                                 !%insl
                                                                                          !%insl
                     thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                       !%insl
                                                                                          !%insl
         !  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0.                 !%insl
         !           (100% frozen))                                                       !%insl
                                                                                          !%insl
                     xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)           !%insl
                                                                                          !%insl
         !  --- ...  unfrozen volume for saturation (porosity*xunfroz)                    !%insl
                                                                                          !%insl
                     xu = xunfroz*smcmax(i, jj)                                           !%insl
                                                                                          !%insl
         !  --- ...  saturated thermal conductivity                                       !%insl
                                                                                          !%insl
                     thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu) &     !%insl
                              *thkw**(xu)                                                 !%insl
                                                                                          !%insl
         !  --- ...  dry density in kg/m3                                                 !%insl
                                                                                          !%insl
                     gammd = (1.0 - smcmax(i, jj))*2700.0                                 !%insl
                                                                                          !%insl
         !  --- ...  dry thermal conductivity in w.m-1.k-1                                !%insl
                                                                                          !%insl
                     thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                 !%insl
                     if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then      ! frozen      !%insl
                        ake = satratio                                                    !%insl
                                                                                          !%insl
                     else                                  ! unfrozen                     !%insl
                                                                                          !%insl
         !  --- ...  range of validity for the kersten number (ake)                       !%insl
                        if (satratio > 0.1) then                                          !%insl
                                                                                          !%insl
         !  --- ...  kersten number (using "fine" formula, valid for soils containing     !%insl
         !           at least 5% of particles with diameter less than 2.e-6 meters.)      !%insl
         !           (for "coarse" formula, see peters-lidard et al., 1998).              !%insl
                                                                                          !%insl
                           ake = log10(satratio) + 1.0                                    !%insl
                                                                                          !%insl
                        else                                                              !%insl
                                                                                          !%insl
         !  --- ...  use k = kdry                                                         !%insl
                           ake = 0.0                                                      !%insl
                                                                                          !%insl
                        end if   ! end if_satratio_block                                  !%insl
                                                                                          !%insl
                     end if   ! end if_sh2o+0.0005_block                                  !%insl
                                                                                          !%insl
         !  --- ...  thermal conductivity                                                 !%insl
                     df1n = ake*(thksat - thkdry) + thkdry                                !%insl
                     !end call tdfcnd_gpu                                                 !%insl
   !urban                                                                                 !%ins
                     if (ivegsrc .ge. 1) then                                             !%ins
                        if (vegtyp(i, jj) == 13) then                                     !%ins
                           df1n = 3.24                                                    !%ins
                        end if                                                            !%ins
                     end if                                                               !%ins
                                                                                          !%ins
   !  --- ...  calc the vertical soil temp gradient thru this layer                       !%ins
                     denom_2 = 0.5*(zsoil(k - 1) - zsoil(k + 1))                          !%ins
                     dtsdz2_1 = (stc(i, k, jj) - stc(i, k + 1, jj))/denom_2               !%ins
                                                                                          !%ins
   !  --- ...  calc the matrix coef, ci, after calc'ng its partial product                !%ins
                                                                                          !%ins
                     ddz2_1 = 2.0/(zsoil(k - 1) - zsoil(k + 1))                           !%ins
                     ci(k) = -df1n*ddz2_1/((zsoil(k - 1) - zsoil(k))*hcpct_1)             !%ins
                                                                                          !%ins
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%ins
   !           calculate temp at bottom of layer.                                         !%ins
                     if (itavg) then                                                      !%ins
                                                                                          !%ins
                        !call tbnd_gpu &                                                  !%insz
                        !   !  ---  inputs: &                                             !%insz
                        !   (stc(k), stc(k + 1), zsoil, zbot, k, nsoil, &                 !%insz
                        !    !  ---  outputs: &                                           !%insz
                        !    tbk1 &                                                       !%insz
                        !    )                                                            !%insz
            !  --- ...  use surface temperature on the top of the first layer             !%insz
                        if (k == 1) then                                                  !%insz
                           zup = 0.0                                                      !%insz
                        else                                                              !%insz
                           zup = zsoil(k - 1)                                             !%insz
                        end if                                                            !%insz
                                                                                          !%insz
            !  --- ...  use depth of the constant bottom temperature when interpolate     !%insz
            !           temperature into the last layer boundary                          !%insz
                                                                                          !%insz
                        if (k == nsoil) then                                              !%insz
                           zb = 2.0*zbot - zsoil(k)                                       !%insz
                        else                                                              !%insz
                           zb = zsoil(k + 1)                                              !%insz
                        end if                                                            !%insz
                                                                                          !%insz
            !  --- ...  linear interpolation between the average layer temperatures       !%insz
                        tbk1 = stc(i, k, jj) + (stc(i, k + 1, jj) - stc(i, k, jj)) &      !%insz
                               *(zup - zsoil(k))/(zup - zb)                               !%insz
                        !end call tbnd_gpu                                                !%insz
                                                                                          !%ins
                     end if                                                               !%ins
                                                                                          !%ins
                  else                                                                    !%ins
                                                                                          !%ins
   !  --- ...  special case of bottom soil layer:  calculate thermal diffusivity          !%ins
   !           for bottom layer.                                                          !%ins
                                                                                          !%ins
                     !call tdfcnd_gpu &                                                   !%insl
                     !   !  ---  inputs: &                                                !%insl
                     !   (smc(k), quartz, smcmax, sh2o(k), &                              !%insl
                     !    !  ---  outputs: &                                              !%insl
                     !    df1n &                                                          !%insl
                     !    )                                                               !%insl
         !  --- ...  if the soil has any moisture content compute a partial sum/product   !%insl
         !           otherwise use a constant value which works well with most soils      !%insl
                                                                                          !%insl
         !  --- ...  saturation ratio:                                                    !%insl
                     satratio = smc(i, k, jj)/smcmax(i, jj)                               !%insl
                                                                                          !%insl
         !  --- ...  parameters  w/(m.k)                                                  !%insl
                     thkice = 2.2                                                         !%insl
                     thkw = 0.57                                                          !%insl
                     thko = 2.0                                                           !%insl
         !     if (quartz <= 0.2) thko = 3.0                                              !%insl
                     thkqtz = 7.7                                                         !%insl
                                                                                          !%insl
         !  --- ...  solids' conductivity                                                 !%insl
                                                                                          !%insl
                     thks = (thkqtz**quartz)*(thko**(1.0 - quartz))                       !%insl
                                                                                          !%insl
         !  --- ...  unfrozen fraction (from 1., i1.e., 100%liquid, to 0.                 !%insl
         !           (100% frozen))                                                       !%insl
                                                                                          !%insl
                     xunfroz = (sh2o(i, k, jj) + 1.e-9)/(smc(i, k, jj) + 1.e-9)           !%insl
                                                                                          !%insl
         !  --- ...  unfrozen volume for saturation (porosity*xunfroz)                    !%insl
                                                                                          !%insl
                     xu = xunfroz*smcmax(i, jj)                                           !%insl
                                                                                          !%insl
         !  --- ...  saturated thermal conductivity                                       !%insl
                                                                                          !%insl
                     thksat = thks**(1.-smcmax(i, jj))*thkice**(smcmax(i, jj) - xu) &     !%insl
                              *thkw**(xu)                                                 !%insl
                                                                                          !%insl
         !  --- ...  dry density in kg/m3                                                 !%insl
                                                                                          !%insl
                     gammd = (1.0 - smcmax(i, jj))*2700.0                                 !%insl
                                                                                          !%insl
         !  --- ...  dry thermal conductivity in w.m-1.k-1                                !%insl
                                                                                          !%insl
                     thkdry = (0.135*gammd + 64.7)/(2700.0 - 0.947*gammd)                 !%insl
                     if (sh2o(i, k, jj) + 0.0005 < smc(i, k, jj)) then      ! frozen      !%insl
                        ake = satratio                                                    !%insl
                                                                                          !%insl
                     else                                  ! unfrozen                     !%insl
                                                                                          !%insl
         !  --- ...  range of validity for the kersten number (ake)                       !%insl
                        if (satratio > 0.1) then                                          !%insl
                                                                                          !%insl
         !  --- ...  kersten number (using "fine" formula, valid for soils containing     !%insl
         !           at least 5% of particles with diameter less than 2.e-6 meters.)      !%insl
         !           (for "coarse" formula, see peters-lidard et al., 1998).              !%insl
                                                                                          !%insl
                           ake = log10(satratio) + 1.0                                    !%insl
                                                                                          !%insl
                        else                                                              !%insl
                                                                                          !%insl
         !  --- ...  use k = kdry                                                         !%insl
                           ake = 0.0                                                      !%insl
                                                                                          !%insl
                        end if   ! end if_satratio_block                                  !%insl
                                                                                          !%insl
                     end if   ! end if_sh2o+0.0005_block                                  !%insl
                                                                                          !%insl
         !  --- ...  thermal conductivity                                                 !%insl
                     df1n = ake*(thksat - thkdry) + thkdry                                !%insl
                     !end call tdfcnd_gpu                                                 !%insl
   !urban                                                                                 !%ins
                     if (ivegsrc .ge. 1) then                                             !%ins
                        if (vegtyp(i, jj) == 13) then                                     !%ins
                           df1n = 3.24                                                    !%ins
                        end if                                                            !%ins
                     end if                                                               !%ins
                                                                                          !%ins
   !  --- ...  calc the vertical soil temp gradient thru bottom layer.                    !%ins
                     denom_2 = 0.5*(zsoil(k - 1) + zsoil(k)) - zbot                       !%ins
                     dtsdz2_1 = (stc(i, k, jj) - tbot(i, jj))/denom_2                     !%ins
                                                                                          !%ins
   !  --- ...  set matrix coef, ci to zero if bottom layer.                               !%ins
                                                                                          !%ins
                     ci(k) = 0.0                                                          !%ins
                                                                                          !%ins
   !  --- ...  if temperature averaging invoked (itavg=true; else skip):                  !%ins
   !           calculate temp at bottom of last layer.                                    !%ins
                     if (itavg) then                                                      !%ins
                                                                                          !%ins
                        !call tbnd_gpu &                                                  !%insz
                        !   !  ---  inputs: &                                             !%insz
                        !   (stc(k), tbot, zsoil, zbot, k, nsoil, &                       !%insz
                        !    !  ---  outputs: &                                           !%insz
                        !    tbk1 &                                                       !%insz
                        !    )                                                            !%insz
            !  --- ...  use surface temperature on the top of the first layer             !%insz
                        if (k == 1) then                                                  !%insz
                           zup = 0.0                                                      !%insz
                        else                                                              !%insz
                           zup = zsoil(k - 1)                                             !%insz
                        end if                                                            !%insz
                                                                                          !%insz
            !  --- ...  use depth of the constant bottom temperature when interpolate     !%insz
            !           temperature into the last layer boundary                          !%insz
                                                                                          !%insz
                        if (k == nsoil) then                                              !%insz
                           zb = 2.0*zbot - zsoil(k)                                       !%insz
                        else                                                              !%insz
                           zb = zsoil(k + 1)                                              !%insz
                        end if                                                            !%insz
                                                                                          !%insz
            !  --- ...  linear interpolation between the average layer temperatures       !%insz
                        tbk1 = stc(i, k, jj) + (tbot(i, jj) - stc(i, k, jj)) &            !%insz
                               *(zup - zsoil(k))/(zup - zb)                               !%insz
                        !end call tbnd_gpu                                                !%insz
                                                                                          !%ins
                     end if                                                               !%ins
                                                                                          !%ins
                  end if   ! end if_k_block                                               !%ins
                                                                                          !%ins
   !  --- ...  calculate rhsts for this layer after calc'ng a partial product.            !%ins
                  denom_2 = (zsoil(k) - zsoil(k - 1))*hcpct_1                             !%ins
                  rhsts(k) = (df1n*dtsdz2_1 - df1k*dtsdz_1)/denom_2                       !%ins
                                                                                          !%ins
                  qtot = -1.0*denom_2*rhsts(k)                                            !%ins
                  sice_1 = smc(i, k, jj) - sh2o(i, k, jj)                                 !%ins
                  if ((sice_1 > 0.0) .or. (tbk < tfreez) .or. &                           !%ins
                      (stc(i, k, jj) < tfreez) .or. (tbk1 < tfreez)) then                 !%ins
                     if (itavg) then                                                      !%ins
                                                                                          !%ins
                        !call tmpavg_gpu &                                                !%insA
                        !   !  ---  inputs: &                                             !%insA
                        !   (tbk, stc(k), tbk1, zsoil, nsoil, k, &                        !%insA
                        !    !  ---  outputs: &                                           !%insA
                        !    tavg &                                                       !%insA
                        !    )                                                            !%insA
                        if (k == 1) then                                                  !%insA
                           dz = -zsoil(1)                                                 !%insA
                        else                                                              !%insA
                           dz = zsoil(k - 1) - zsoil(k)                                   !%insA
                        end if                                                            !%insA
                        dzh = dz*0.5                                                      !%insA
                                                                                          !%insA
                        if (tbk < tfreez) then                                            !%insA
                           if (stc(i, k, jj) < tfreez) then                               !%insA
                              if (tbk1 < tfreez) then ! tbk, stc(k), tbk1 < t0            !%insA
                                 tavg = (tbk + 2.0*stc(i, k, jj) + tbk1)/4.0              !%insA
                              else ! tbk & stc(k) < t0,  tbk1 >= t0                       !%insA
                                 x0 = (tfreez - stc(i, k, jj))*dzh &                      !%insA
                                      /(tbk1 - stc(i, k, jj))                             !%insA
                                 tavg = 0.5*(tbk*dzh + stc(i, k, jj)*(dzh + x0) &         !%insA
                                        + tfreez*(2.*dzh - x0))/dz                        !%insA
                              end if                                                      !%insA
                           else                                                           !%insA
                              if (tbk1 < tfreez) then ! tbk < t0,                         !%insA
                                                      ! stc(k) >= t0, tbk1 < t0           !%insA
                                 xup = (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)           !%insA
                                 xdn = dzh - (tfreez - stc(i, k, jj))*dzh &               !%insA
                                       /(tbk1 - stc(i, k, jj))                            !%insA
                                 tavg = 0.5*(tbk*xup + tfreez*(2.*dz - xup - xdn) &       !%insA
                                        + tbk1*xdn)/dz                                    !%insA
                              else    ! tbk < t0, stc(k) >= t0, tbk1 >= t0                !%insA
                                 xup = (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)           !%insA
                                 tavg = 0.5*(tbk*xup + tfreez*(2.*dz - xup))/dz           !%insA
                              end if                                                      !%insA
                           end if                                                         !%insA
                                                                                          !%insA
                        else    ! if_tup_block                                            !%insA
                           if (stc(i, k, jj) < tfreez) then                               !%insA
                              if (tbk1 < tfreez) then  ! tbk >= t0,                       !%insA
                                                       ! stc(k) < t0, tbk1 < t0           !%insA
                                 xup = dzh - (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)     !%insA
                                 tavg = 0.5*(tfreez*(dz - xup) + stc(i, k, jj) &          !%insA
                                        *(dzh + xup) + tbk1*dzh)/dz                       !%insA
                                                                                          !%insA
                              else ! tbk >= t0, stc(k) < t0, tbk1 >= t0                   !%insA
                                 xup = dzh - (tfreez - tbk)*dzh/(stc(i, k, jj) - tbk)     !%insA
                                 xdn = (tfreez - stc(i, k, jj))*dzh &                     !%insA
                                       /(tbk1 - stc(i, k, jj))                            !%insA
                                 tavg = 0.5*(tfreez*(2.*dz - xup - xdn) &                 !%insA
                                        + stc(i, k, jj)*(xup + xdn))/dz                   !%insA
                              end if                                                      !%insA
                           else                                                           !%insA
                              if (tbk1 < tfreez) then ! tbk >= t0,                        !%insA
                                                      ! stc(k) >= t0, tbk1 < t0           !%insA
                                 xdn = dzh - (tfreez - stc(i, k, jj)) &                   !%insA
                                       *dzh/(tbk1 - stc(i, k, jj))                        !%insA
                                 tavg = (tfreez*(dz - xdn) &                              !%insA
                                       + 0.5*(tfreez + tbk1)*xdn)/dz                      !%insA
                              else   ! tbk >= t0, stc(k) >= t0, tbk1 >= t0                !%insA
                                 tavg = (tbk + 2.0*stc(i, k, jj) + tbk1)/4.0              !%insA
                              end if                                                      !%insA
                           end if                                                         !%insA
                                                                                          !%insA
                        end if   ! end if_tup_block                                       !%insA
                        !end call tmpavg_gpu                                              !%insA
                                                                                          !%ins
                     else                                                                 !%ins
                        tavg = stc(i, k, jj)                                              !%ins
                     end if                                                               !%ins
                                                                                          !%ins
                     !call snksrc_gpu &                                                   !%insw
                     !   !  ---  inputs: &                                                !%insw
                     !   (nsoil, k, tavg, smc(k), smcmax, psisat, bexp, dt, &             !%insw
                     !    qtot, zsoil, ivegsrc, vegtyp, &                                 !%insw
                     !    !  ---  input/outputs: &                                        !%insw
                     !    sh2o(k), df1, &                                                 !%insw
                     !    !  ---  outputs: &                                              !%insw
                     !    tsnsr &                                                         !%insw
                     !    )                                                               !%insw
                     if (ivegsrc .ge. 1) then                                             !%insw
                        if (vegtyp(i, jj) == 13) then                                     !%insw
                           df1 = 3.24                                                     !%insw
                        end if                                                            !%insw
                     end if                                                               !%insw
         !                                                                                !%insw
         !===> ...  begin here                                                            !%insw
         !                                                                                !%insw
                     if (k == 1) then                                                     !%insw
                        dz_1 = -zsoil(1)                                                  !%insw
                     else                                                                 !%insw
                        dz_1 = zsoil(k - 1) - zsoil(k)                                    !%insw
                     end if                                                               !%insw
                                                                                          !%insw
         !  --- ...  via function frh2o, compute potential or 'equilibrium' unfrozen      !%insw
         !           supercooled free water for given soil type and soil layer            !%insw
         !           temperature. function frh20 invokes eqn (17) from v. koren et al     !%insw
         !           (1999, jgr, vol. 104, pg 19573).  (aside:  latter eqn in journal     !%insw
         !           in centigrade units. routine frh2o use form of eqn in kelvin         !%insw
         !           units.)                                                              !%insw
                                                                                          !%insw
         !     free = frh2o( tavg,smc(k),sh2o(k),smcmax(i, jj),bexp,psisat )              !%insw
                                                                                          !%insw
                     !call frh2o_gpu &                                                    !%inswr
                     !   !  ---  inputs: &                                                !%inswr
                     !   (tavg, smc(k), sh2o(k), smcmax(i, jj), bexp, psisat, &           !%inswr
                     !    !  ---  outputs: &                                              !%inswr
                     !    free &                                                          !%inswr
                     !    )                                                               !%inswr
                                                                                          !%inswr
         !                                                                                !%inswr
         !===> ...  begin here                                                            !%inswr
         !                                                                                !%inswr
         !  --- ...  limits on parameter b: b < 5.5  (use parameter blim)                 !%inswr
         !           simulations showed if b > 5.5 unfrozen water content is              !%inswr
         !           non-realistically high at very low temperatures.                     !%inswr
                     bx = bexp                                                            !%inswr
                     if (bexp > blim) bx = blim                                           !%inswr
                                                                                          !%inswr
         !  --- ...  initializing iterations counter and iterative solution flag.         !%inswr
                                                                                          !%inswr
                     nlog = 0                                                             !%inswr
                     kcount = 0                                                           !%inswr
                                                                                          !%inswr
         !  --- ...  if temperature not significantly below freezing (t0),                !%inswr
         !           sh2o(k) = smc(k)                                                     !%inswr
                     if (tavg > (tfreez - 1.e-3)) then                                    !%inswr
                        free = smc(i, k, jj)                                              !%inswr
                                                                                          !%inswr
                     else                                                                 !%inswr
                        if (ck /= 0.0) then                                               !%inswr
                                                                                          !%inswr
         !  --- ...  option 1: iterated solution for nonzero ck                           !%inswr
         !                     in koren et al, jgr, 1999, eqn 17                          !%inswr
                                                                                          !%inswr
         !  --- ...  initial guess for swl (frozen content)                               !%inswr
                           swl = smc(i, k, jj) - sh2o(i, k, jj)                           !%inswr
                                                                                          !%inswr
         !  --- ...  keep within bounds.                                                  !%inswr
                                                                                          !%inswr
                           swl = max(min(swl, smc(i, k, jj) - 0.02), 0.0)                 !%inswr
                                                                                          !%inswr
         !  --- ...  start of iterations                                                  !%inswr
                           do while ((nlog < 10) .and. (kcount == 0))                     !%inswr
                              nlog = nlog + 1                                             !%inswr
                                                                                          !%inswr
                              df = dlog((psisat*gs2/lsubf)*((1.0 + ck*swl)**2.0) &        !%inswr
                                        *(smcmax(i, jj)/(smc(i, k, jj) - swl))**bx) &     !%inswr
                                        - dlog(-(tavg - tfreez)/tavg)                     !%inswr
                                                                                          !%inswr
                              denom_1 = 2.0*ck/(1.0 + ck*swl) + bx/(smc(i, k, jj) - swl)  !%inswr
                              swlk = swl - df/denom_1                                     !%inswr
                                                                                          !%inswr
         !  --- ...  bounds useful for mathematical solution.                             !%inswr
                                                                                          !%inswr
                              swlk = max(min(swlk, smc(i, k, jj) - 0.02), 0.0)            !%inswr
                                                                                          !%inswr
         !  --- ...  mathematical solution bounds applied.                                !%inswr
                                                                                          !%inswr
                              dswl = abs(swlk - swl)                                      !%inswr
                              swl = swlk                                                  !%inswr
                                                                                          !%inswr
         !  --- ...  if more than 10 iterations, use explicit method (ck=0 approx.)       !%inswr
         !           when dswl less or eq. error, no more iterations required.            !%inswr
                              if (dswl <= error) then                                     !%inswr
                                 kcount = kcount + 1                                      !%inswr
                              end if                                                      !%inswr
                           end do   !  end do_while_loop                                  !%inswr
                                                                                          !%inswr
         !  --- ...  bounds applied within do-block are valid for physical solution.      !%inswr
                           free = smc(i, k, jj) - swl                                     !%inswr
                                                                                          !%inswr
                        end if   ! end if_ck_block                                        !%inswr
                                                                                          !%inswr
         !  --- ...  option 2: explicit solution for flerchinger eq. i1.e. ck=0           !%inswr
         !                     in koren et al., jgr, 1999, eqn 17                         !%inswr
         !           apply physical bounds to flerchinger solution                        !%inswr
                        if (kcount == 0) then                                             !%inswr
                           fk = (((lsubf/(gs2*(-psisat))) &                               !%inswr
                                  *((tavg - tfreez)/tavg))**(-1/bx))*smcmax(i, jj)        !%inswr
                                                                                          !%inswr
                           fk = max(fk, 0.02)                                             !%inswr
                                                                                          !%inswr
                           free = min(fk, smc(i, k, jj))                                  !%inswr
                        end if                                                            !%inswr
                                                                                          !%inswr
                     end if   ! end if_tkelv_block                                        !%inswr
                     !end call frh2o_gpu                                                  !%inswr
                                                                                          !%insw
                                                                                          !%insw
         !  --- ...  in next block of code, invoke eqn 18 of v. koren et al (1999,        !%insw
         !           jgr, vol. 104, pg 19573.)  that is, first estimate the new           !%insw
         !           amountof liquid water, 'xh2o', implied by the sum of (1) the         !%insw
         !           liquid water at the begin of current time step, and (2) the          !%insw
         !           freeze of thaw change in liquidwater implied by the heat flux        !%insw
         !           'qtot' passed in from routine hrt.second, determine if xh2o          !%insw
         !           needs to be bounded by 'free' (equil amt) orif 'free' needs to       !%insw
         !           be bounded by xh2o.                                                  !%insw
                                                                                          !%insw
                     xh2o = sh2o(i, k, jj) + qtot*dt/(dh2o*lsubf*dz_1)                    !%insw
                                                                                          !%insw
         !  --- ...  first, if freezing and remaining liquid less than lower bound,       !%insw
         !           then reduce extent of freezing, thereby letting some or all of       !%insw
         !           heat flux qtot cool the soil temp later in routine hrt.              !%insw
                     if (xh2o < sh2o(i, k, jj) .and. xh2o < free) then                    !%insw
                        if (free > sh2o(i, k, jj)) then                                   !%insw
                           xh2o = sh2o(i, k, jj)                                          !%insw
                        else                                                              !%insw
                           xh2o = free                                                    !%insw
                        end if                                                            !%insw
                     end if                                                               !%insw
                                                                                          !%insw
         !  --- ...  second, if thawing and the increase in liquid water greater than     !%insw
         !           upper bound, then reduce extent of thaw, thereby letting some or     !%insw
         !           all of heat flux qtot warm the soil temp later in routine hrt.       !%insw
                                                                                          !%insw
                     if (xh2o > sh2o(i, k, jj) .and. xh2o > free) then                    !%insw
                        if (free < sh2o(i, k, jj)) then                                   !%insw
                           xh2o = sh2o(i, k, jj)                                          !%insw
                        else                                                              !%insw
                           xh2o = free                                                    !%insw
                        end if                                                            !%insw
                     end if                                                               !%insw
                                                                                          !%insw
                     xh2o = max(min(xh2o, smc(i, k, jj)), 0.0)                            !%insw
                                                                                          !%insw
         !  --- ...  calculate phase-change heat source/sink term for use in routine      !%insw
         !           hrt and update liquid water to reflcet final freeze/thaw             !%insw
         !           increment.                                                           !%insw
                                                                                          !%insw
                     tsnsr = -dh2o*lsubf*dz_1*(xh2o - sh2o(i, k, jj))/dt                  !%insw
                     sh2o(i, k, jj) = xh2o                                                !%insw
                     !end call snksrc_gpu                                                 !%insw
                     rhsts(k) = rhsts(k) - tsnsr/denom_2                                  !%ins
                  end if                                                                  !%ins
                                                                                          !%ins
   !  --- ...  calc matrix coefs, ai, and bi for this layer.                              !%ins
                  ai(k) = -df1*ddz_2/((zsoil(k - 1) - zsoil(k))*hcpct_1)                  !%ins
                  bi(k) = -(ai(k) + ci(k))                                                !%ins
                                                                                          !%ins
   !  --- ...  reset values of df1, dtsdz_1, ddz_2, and tbk for loop to next soil         !%ins
   !           layer.                                                                     !%ins
                                                                                          !%ins
                  tbk = tbk1                                                              !%ins
                  df1k = df1n                                                             !%ins
                  dtsdz_1 = dtsdz2_1                                                      !%ins
                  ddz_2 = ddz2_1                                                          !%ins
                                                                                          !%ins
               end do   ! end do_k_loop                                                   !%ins
               !end call hrt_gpu                                                          !%ins
                                                                                          !%in
               !call hstep_gpu &                                                          !%iny
               !   !  ---  inputs: &                                                      !%iny
               !   (nsoil, stc, dt, &                                                     !%iny
               !    !  ---  input/outputs: &                                              !%iny
               !    rhsts, ai, bi, ci, &                                                  !%iny
               !    !  ---  outputs: &                                                    !%iny
               !    stcf, &                                                               !%iny
               !    !  ---  dummys: &                                                     !%iny
               !    ciin, rhstsin &                                                       !%iny
               !    )                                                                     !%iny
   !  --- ...  create finite difference values for use in rosr12 routine                  !%iny
               !$acc loop seq                                                             !%iny
               do k = 1, nsoil                                                            !%iny
                  rhsts(k) = rhsts(k)*dt                                                  !%iny
                  ai(k) = ai(k)*dt                                                        !%iny
                  bi(k) = 1.0 + bi(k)*dt                                                  !%iny
                  ci(k) = ci(k)*dt                                                        !%iny
               end do                                                                     !%iny
                                                                                          !%iny
   !  --- ...  copy values for input variables before call to rosr12                      !%iny
                                                                                          !%iny
               !$acc loop seq                                                             !%iny
               do k = 1, nsoil                                                            !%iny
                  rhstsin(k) = rhsts(k)                                                   !%iny
               end do                                                                     !%iny
                                                                                          !%iny
               !$acc loop seq                                                             !%iny
               do k = 1, nsold                                                            !%iny
                  ciin(k) = ci(k)                                                         !%iny
               end do                                                                     !%iny
                                                                                          !%iny
   !  --- ...  solve the tri-diagonal matrix equation                                     !%iny
                                                                                          !%iny
               !call rosr12_gpu &                                                         !%inyv
               !   !  ---  inputs: &                                                      !%inyv
               !   (nsoil, ai, bi, rhstsin, &                                             !%inyv
               !    !  ---  input/outputs: &                                              !%inyv
               !    ciin, &                                                               !%inyv
               !    !  ---  outputs: &                                                    !%inyv
               !    ci, rhsts &                                                           !%inyv
               !    )                                                                     !%inyv
   !  --- ...  initialize eqn coef ciin for the lowest soil layer                         !%inyv
                                                                                          !%inyv
               ciin(nsoil) = 0.0                                                          !%inyv
                                                                                          !%inyv
   !  --- ...  solve the coefs for the 1st soil layer                                     !%inyv
               ci(1) = -ciin(1)/bi(1)                                                     !%inyv
               rhsts(1) = rhstsin(1)/bi(1)                                                !%inyv
                                                                                          !%inyv
   !  --- ...  solve the coefs for soil layers 2 thru nsoil                               !%inyv
               !$acc loop seq                                                             !%inyv
               do k = 2, nsoil                                                            !%inyv
                  ci(k) = -ciin(k)*(1.0/(bi(k) + ai(k)*ci(k - 1)))                        !%inyv
                  rhsts(k) = (rhstsin(k) - ai(k)*rhsts(k - 1)) &                          !%inyv
                             *(1.0/(bi(k) + ai(k)*ci(k - 1)))                             !%inyv
               end do                                                                     !%inyv
                                                                                          !%inyv
   !  --- ...  set ci to rhsts for lowest soil layer                                      !%inyv
               ci(nsoil) = rhsts(nsoil)                                                   !%inyv
                                                                                          !%inyv
   !  --- ...  adjust ci for soil layers 2 thru nsoil                                     !%inyv
                                                                                          !%inyv
               !$acc loop seq                                                             !%inyv
               do k = 2, nsoil                                                            !%inyv
                  kk = nsoil - k + 1                                                      !%inyv
                  ci(kk) = ci(kk)*ci(kk + 1) + rhsts(kk)                                  !%inyv
               end do                                                                     !%inyv
               !end call rosr12_gpu                                                       !%inyv
                                                                                          !%iny
   !  --- ...  calc/update the soil temps using matrix solution                           !%iny
                                                                                          !%iny
               !$acc loop seq                                                             !%iny
               do k = 1, nsoil                                                            !%iny
                  stcf(k) = stc(i, k, jj) + ci(k)                                         !%iny
               end do                                                                     !%iny
               !end call hstep_gpu                                                        !%iny
                                                                                          !%in
            end if                                                                        !%in
            !$acc loop seq                                                                !%in
            do i1 = 1, nsoil                                                              !%in
               stc(i, i1, jj) = stcf(i1)                                                  !%in
            end do                                                                        !%in
                                                                                          !%in
!  --- ...  in the no snowpack case (via routine nopac branch,) update the grnd           !%in
!           (skin) temperature here in response to the updated soil temperature           !%in
!           profile above.  (note: inspection of routine snopac shows that t11            !%in
!           below is a dummy variable only, as skin temperature is updated                !%in
!           differently in routine snopac)                                                !%in
            t11 = (yy + (zz1 - 1.0)*stc(i, 1, jj))/zz1                                    !%in
            t11 = ctfil1*t11 + ctfil2*oldt1                                               !%in
            !$acc loop seq                                                                !%in
            do i1 = 1, nsoil                                                              !%in
               stc(i, i1, jj) = ctfil1*stc(i, i1, jj) + ctfil2*stsoil(i1)                 !%in
            end do                                                                        !%in
                                                                                          !%in
!  --- ...  calculate surface soil heat flux                                              !%in
            ssoil1 = df1*(stc(i, 1, jj) - t11)/(0.5*zsoil(1))                             !%in
            !end call shflx_gpu                                                           !%in
                                                                                          !%i
!  --- ...  snow depth and density adjustment based on snow compaction.  yy is            !%i
!           assumed to be the soil temperture at the top of the soil column.              !%i
                                                                                          !%i
            if (ice == 0) then              ! for non-glacial land                        !%i
               if (sneqv(i, jj) > 0.0) then                                               !%i
                                                                                          !%i
                  !call snowpack_gpu &                                                    !%ip
                  !   !  ---  inputs: &                                                   !%ip
                  !   (sneqv, dt, t1, yy, &                                               !%ip
                  !    !  ---  input/outputs: &                                           !%ip
                  !    snowh, sndens &                                                    !%ip
                  !    )                                                                  !%ip
      !  --- ...  conversion into simulation units                                        !%ip
                  snowhc = snowh(i, jj)*100.0                                             !%ip
                  esdc = sneqv(i, jj)*100.0                                               !%ip
                  dthr = dt/3600.0                                                        !%ip
                  tsnowc = t1(i, jj) - tfreez                                             !%ip
                  tsoilc = yy - tfreez                                                    !%ip
                                                                                          !%ip
      !  --- ...  calculating of average temperature of snow pack                         !%ip
                                                                                          !%ip
                  tavgc = 0.5*(tsnowc + tsoilc)                                           !%ip
                                                                                          !%ip
      !  --- ...  calculating of snow depth and density as a result of compaction         !%ip
      !           sndens=ds0*(exp(bfac*sneqv(i, jj))-1.)/(bfac*sneqv(i, jj))              !%ip
      !           bfac=dthr*c1*exp(0.08*tavgc-c2*ds0)                                     !%ip
      !     note: bfac*sneqv(i, jj) in sndens eqn above has to be carefully treated       !%ip
      !           numerically below:                                                      !%ip
      !        c1 is the fractional increase in density (1/(cm(i, jj)*hr))                !%ip
      !        c2 is a constant (cm3/g) kojima estimated as 21 cms/g                      !%ip
                  if (esdc > 1.e-2) then                                                  !%ip
                     esdcx = esdc                                                         !%ip
                  else                                                                    !%ip
                     esdcx = 1.e-2                                                        !%ip
                  end if                                                                  !%ip
                  bfac = dthr*c1*exp(0.08*tavgc - c2*sndens)                              !%ip
                                                                                          !%ip
      !     dsx = sndens * ((dexp(bfac*esdc)-1.0) / (bfac*esdc))                          !%ip
                                                                                          !%ip
      !  --- ...  the function of the form (e**x-1)/x imbedded in above expression        !%ip
      !           for dsx was causing numerical difficulties when the denominator "x"     !%ip
      !           (i1.e. bfac*esdc) became zero or approached zero (despite the fact      !%ip
      !           that the analytical function (e**x-1)/x has a well defined limit        !%ip
      !           as "x" approaches zero), hence below we replace the (e**x-1)/x          !%ip
      !           expression with an equivalent, numerically well-behaved                 !%ip
      !           polynomial expansion.                                                   !%ip
                                                                                          !%ip
      !  --- ...  number of terms of polynomial expansion, and hence its accuracy,        !%ip
      !           is governed by iteration limit "ipol".                                  !%ip
      !           ipol greater than 9 only makes a difference on double                   !%ip
      !           precision (relative errors given in percent %).                         !%ip
      !       ipol=9, for rel.error <~ 1.6 e-6 % (8 significant digits)                   !%ip
      !       ipol=8, for rel.error <~ 1.8 e-5 % (7 significant digits)                   !%ip
      !       ipol=7, for rel.error <~ 1.8 e-4 % ...                                      !%ip
                                                                                          !%ip
                  ipol = 4                                                                !%ip
                  pexp = 0.0                                                              !%ip
                  !$acc loop seq                                                          !%ip
                  do j = ipol, 1, -1                                                      !%ip
      !       pexp = (1.0 + pexp)*bfac*esdc /real(j+1)                                    !%ip
                     pexp = (1.0 + pexp)*bfac*esdcx/real(j + 1)                           !%ip
                  end do                                                                  !%ip
                  pexp = pexp + 1.                                                        !%ip
                                                                                          !%ip
                  dsx = sndens*pexp                                                       !%ip
                                                                                          !%ip
      !  --- ...  above line ends polynomial substitution                                 !%ip
      !           end of koren formulation                                                !%ip
                                                                                          !%ip
      !! --- ...  base formulation (cogley et al., 1990)                                  !%ip
      !           convert density from g/cm3 to kg/m3                                     !%ip
                                                                                          !%ip
      !!      dsm = sndens * 1000.0                                                       !%ip
                                                                                          !%ip
      !!      dsx = dsm + dt*0.5*dsm*gs2*sneqv(i, jj) /                             &     !%ip
      !!   &        (1.e7*exp(-0.02*dsm + kn/(tavgc+273.16)-14.643))                      !%ip
                                                                                          !%ip
      !! --- ...  convert density from kg/m3 to g/cm3                                     !%ip
                                                                                          !%ip
      !!      dsx = dsx / 1000.0                                                          !%ip
                                                                                          !%ip
      !! --- ...  end of cogley et al. formulation                                        !%ip
                                                                                          !%ip
      !  --- ...  set upper/lower limit on snow density                                   !%ip
                                                                                          !%ip
                  dsx = max(min(dsx, 0.40), 0.05)                                         !%ip
                  sndens = dsx                                                            !%ip
                                                                                          !%ip
      !  --- ...  update of snow depth and density depending on liquid water              !%ip
      !           during snowmelt.  assumed that 13% of liquid water can be               !%ip
      !           stored in snow per day during snowmelt till snow density 0.40.          !%ip
                  if (tsnowc >= 0.0) then                                                 !%ip
                     dw = 0.13*dthr/24.0                                                  !%ip
                     sndens = sndens*(1.0 - dw) + dw                                      !%ip
                     if (sndens > 0.40) sndens = 0.40                                     !%ip
                  end if                                                                  !%ip
                                                                                          !%ip
      !  --- ...  calculate snow depth (cm(i, jj)) from snow water equivalent and snow    !%ip
      !           density. change snow depth units to meters                              !%ip
                  snowhc = esdc/sndens                                                    !%ip
                  snowh(i, jj) = snowhc*0.01                                              !%ip
                  !end call snowpack_gpu                                                  !%ip
                                                                                          !%i
               else                                                                       !%i
                                                                                          !%i
                  sneqv(i, jj) = 0.0                                                      !%i
                  snowh(i, jj) = 0.0                                                      !%i
                  sndens = 0.0                                                            !%i
!         sncond = 1.0                                                                    !%i
                  sncovr(i, jj) = 0.0                                                     !%i
                                                                                          !%i
               end if   ! end if_sneqv_block                                              !%i
                                                                                          !%i
!  --- ...  over sea-ice or glacial-ice, if s.w.e. (sneqv(i, jj)) below threshold         !%i
!           lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then set at         !%i
!           lower bound and store the source increment in subsurface runoff/              !%i
!           baseflow (runoff2(i, jj)).  note:  runoff2(i, jj) is then a negative          !%i
!           value (as a flag) over sea-ice or glacial-ice, in order to achieve water      !%i
!           balance.                                                                      !%i
            elseif (ice == 1) then          ! for sea-ice                                 !%i
               if (sneqv(i, jj) >= 0.01) then                                             !%i
                                                                                          !%i
                  !call snowpack_gpu &                                                    !%ip
                  !   !  ---  inputs: &                                                   !%ip
                  !   (sneqv(i, jj), dt, t1(i, jj), yy, &                                 !%ip
                  !    !  ---  input/outputs: &                                           !%ip
                  !    snowh(i, jj), sndens &                                             !%ip
                  !    )                                                                  !%ip
      !  --- ...  conversion into simulation units                                        !%ip
                  snowhc = snowh(i, jj)*100.0                                             !%ip
                  esdc = sneqv(i, jj)*100.0                                               !%ip
                  dthr = dt/3600.0                                                        !%ip
                  tsnowc = t1(i, jj) - tfreez                                             !%ip
                  tsoilc = yy - tfreez                                                    !%ip
                                                                                          !%ip
      !  --- ...  calculating of average temperature of snow pack                         !%ip
                                                                                          !%ip
                  tavgc = 0.5*(tsnowc + tsoilc)                                           !%ip
                                                                                          !%ip
      !  --- ...  calculating of snow depth and density as a result of compaction         !%ip
      !           sndens=ds0*(exp(bfac*sneqv(i, jj))-1.)/(bfac*sneqv(i, jj))              !%ip
      !           bfac=dthr*c1*exp(0.08*tavgc-c2*ds0)                                     !%ip
      !     note: bfac*sneqv(i, jj) in sndens eqn above has to be carefully treated       !%ip
      !           numerically below:                                                      !%ip
      !        c1 is the fractional increase in density (1/(cm(i, jj)*hr))                !%ip
      !        c2 is a constant (cm3/g) kojima estimated as 21 cms/g                      !%ip
                  if (esdc > 1.e-2) then                                                  !%ip
                     esdcx = esdc                                                         !%ip
                  else                                                                    !%ip
                     esdcx = 1.e-2                                                        !%ip
                  end if                                                                  !%ip
                  bfac = dthr*c1*exp(0.08*tavgc - c2*sndens)                              !%ip
                                                                                          !%ip
      !     dsx = sndens * ((dexp(bfac*esdc)-1.0) / (bfac*esdc))                          !%ip
                                                                                          !%ip
      !  --- ...  the function of the form (e**x-1)/x imbedded in above expression        !%ip
      !           for dsx was causing numerical difficulties when the denominator "x"     !%ip
      !           (i1.e. bfac*esdc) became zero or approached zero (despite the fact      !%ip
      !           that the analytical function (e**x-1)/x has a well defined limit        !%ip
      !           as "x" approaches zero), hence below we replace the (e**x-1)/x          !%ip
      !           expression with an equivalent, numerically well-behaved                 !%ip
      !           polynomial expansion.                                                   !%ip
                                                                                          !%ip
      !  --- ...  number of terms of polynomial expansion, and hence its accuracy,        !%ip
      !           is governed by iteration limit "ipol".                                  !%ip
      !           ipol greater than 9 only makes a difference on double                   !%ip
      !           precision (relative errors given in percent %).                         !%ip
      !       ipol=9, for rel.error <~ 1.6 e-6 % (8 significant digits)                   !%ip
      !       ipol=8, for rel.error <~ 1.8 e-5 % (7 significant digits)                   !%ip
      !       ipol=7, for rel.error <~ 1.8 e-4 % ...                                      !%ip
                                                                                          !%ip
                  ipol = 4                                                                !%ip
                  pexp = 0.0                                                              !%ip
                  !$acc loop seq                                                          !%ip
                  do j = ipol, 1, -1                                                      !%ip
      !       pexp = (1.0 + pexp)*bfac*esdc /real(j+1)                                    !%ip
                     pexp = (1.0 + pexp)*bfac*esdcx/real(j + 1)                           !%ip
                  end do                                                                  !%ip
                  pexp = pexp + 1.                                                        !%ip
                                                                                          !%ip
                  dsx = sndens*pexp                                                       !%ip
                                                                                          !%ip
      !  --- ...  above line ends polynomial substitution                                 !%ip
      !           end of koren formulation                                                !%ip
                                                                                          !%ip
      !! --- ...  base formulation (cogley et al., 1990)                                  !%ip
      !           convert density from g/cm3 to kg/m3                                     !%ip
                                                                                          !%ip
      !!      dsm = sndens * 1000.0                                                       !%ip
                                                                                          !%ip
      !!      dsx = dsm + dt*0.5*dsm*gs2*sneqv(i, jj) /                             &     !%ip
      !!   &        (1.e7*exp(-0.02*dsm + kn/(tavgc+273.16)-14.643))                      !%ip
                                                                                          !%ip
      !! --- ...  convert density from kg/m3 to g/cm3                                     !%ip
                                                                                          !%ip
      !!      dsx = dsx / 1000.0                                                          !%ip
                                                                                          !%ip
      !! --- ...  end of cogley et al. formulation                                        !%ip
                                                                                          !%ip
      !  --- ...  set upper/lower limit on snow density                                   !%ip
                                                                                          !%ip
                  dsx = max(min(dsx, 0.40), 0.05)                                         !%ip
                  sndens = dsx                                                            !%ip
                                                                                          !%ip
      !  --- ...  update of snow depth and density depending on liquid water              !%ip
      !           during snowmelt.  assumed that 13% of liquid water can be               !%ip
      !           stored in snow per day during snowmelt till snow density 0.40.          !%ip
                  if (tsnowc >= 0.0) then                                                 !%ip
                     dw = 0.13*dthr/24.0                                                  !%ip
                     sndens = sndens*(1.0 - dw) + dw                                      !%ip
                     if (sndens > 0.40) sndens = 0.40                                     !%ip
                  end if                                                                  !%ip
                                                                                          !%ip
      !  --- ...  calculate snow depth (cm(i, jj)) from snow water equivalent and snow    !%ip
      !           density. change snow depth units to meters                              !%ip
                  snowhc = esdc/sndens                                                    !%ip
                  snowh(i, jj) = snowhc*0.01                                              !%ip
                  !end call snowpack_gpu                                                  !%ip
                                                                                          !%i
               else                                                                       !%i
                                                                                          !%i
!         sndens = sneqv(i, jj) / snowh(i, jj)                                            !%i
!         runoff2(i, jj) = -(0.01 - sneqv(i, jj)) / dt                                    !%i
                  sneqv(i, jj) = 0.01                                                     !%i
                  snowh(i, jj) = 0.05                                                     !%i
                  sncovr(i, jj) = 1.0                                                     !%i
!         snowh(i, jj) = sneqv(i, jj) / sndens                                            !%i
                                                                                          !%i
               end if   ! end if_sneqv_block                                              !%i
                                                                                          !%i
            else                            ! for glacial-ice                             !%i
               if (sneqv(i, jj) >= 0.10) then                                             !%i
                                                                                          !%i
                  !call snowpack_gpu &                                                    !%ip
                  !   !  ---  inputs: &                                                   !%ip
                  !   (sneqv, dt, t1, yy, &                                               !%ip
                  !    !  ---  input/outputs: &                                           !%ip
                  !    snowh, sndens &                                                    !%ip
                  !    )                                                                  !%ip
      !  --- ...  conversion into simulation units                                        !%ip
                  snowhc = snowh(i, jj)*100.0                                             !%ip
                  esdc = sneqv(i, jj)*100.0                                               !%ip
                  dthr = dt/3600.0                                                        !%ip
                  tsnowc = t1(i, jj) - tfreez                                             !%ip
                  tsoilc = yy - tfreez                                                    !%ip
                                                                                          !%ip
      !  --- ...  calculating of average temperature of snow pack                         !%ip
                                                                                          !%ip
                  tavgc = 0.5*(tsnowc + tsoilc)                                           !%ip
                                                                                          !%ip
      !  --- ...  calculating of snow depth and density as a result of compaction         !%ip
      !           sndens=ds0*(exp(bfac*sneqv(i, jj))-1.)/(bfac*sneqv(i, jj))              !%ip
      !           bfac=dthr*c1*exp(0.08*tavgc-c2*ds0)                                     !%ip
      !     note: bfac*sneqv(i, jj) in sndens eqn above has to be carefully treated       !%ip
      !           numerically below:                                                      !%ip
      !        c1 is the fractional increase in density (1/(cm(i, jj)*hr))                !%ip
      !        c2 is a constant (cm3/g) kojima estimated as 21 cms/g                      !%ip
                  if (esdc > 1.e-2) then                                                  !%ip
                     esdcx = esdc                                                         !%ip
                  else                                                                    !%ip
                     esdcx = 1.e-2                                                        !%ip
                  end if                                                                  !%ip
                  bfac = dthr*c1*exp(0.08*tavgc - c2*sndens)                              !%ip
                                                                                          !%ip
      !     dsx = sndens * ((dexp(bfac*esdc)-1.0) / (bfac*esdc))                          !%ip
                                                                                          !%ip
      !  --- ...  the function of the form (e**x-1)/x imbedded in above expression        !%ip
      !           for dsx was causing numerical difficulties when the denominator "x"     !%ip
      !           (i1.e. bfac*esdc) became zero or approached zero (despite the fact      !%ip
      !           that the analytical function (e**x-1)/x has a well defined limit        !%ip
      !           as "x" approaches zero), hence below we replace the (e**x-1)/x          !%ip
      !           expression with an equivalent, numerically well-behaved                 !%ip
      !           polynomial expansion.                                                   !%ip
                                                                                          !%ip
      !  --- ...  number of terms of polynomial expansion, and hence its accuracy,        !%ip
      !           is governed by iteration limit "ipol".                                  !%ip
      !           ipol greater than 9 only makes a difference on double                   !%ip
      !           precision (relative errors given in percent %).                         !%ip
      !       ipol=9, for rel.error <~ 1.6 e-6 % (8 significant digits)                   !%ip
      !       ipol=8, for rel.error <~ 1.8 e-5 % (7 significant digits)                   !%ip
      !       ipol=7, for rel.error <~ 1.8 e-4 % ...                                      !%ip
                                                                                          !%ip
                  ipol = 4                                                                !%ip
                  pexp = 0.0                                                              !%ip
                  !$acc loop seq                                                          !%ip
                  do j = ipol, 1, -1                                                      !%ip
      !       pexp = (1.0 + pexp)*bfac*esdc /real(j+1)                                    !%ip
                     pexp = (1.0 + pexp)*bfac*esdcx/real(j + 1)                           !%ip
                  end do                                                                  !%ip
                  pexp = pexp + 1.                                                        !%ip
                                                                                          !%ip
                  dsx = sndens*pexp                                                       !%ip
                                                                                          !%ip
      !  --- ...  above line ends polynomial substitution                                 !%ip
      !           end of koren formulation                                                !%ip
                                                                                          !%ip
      !! --- ...  base formulation (cogley et al., 1990)                                  !%ip
      !           convert density from g/cm3 to kg/m3                                     !%ip
                                                                                          !%ip
      !!      dsm = sndens * 1000.0                                                       !%ip
                                                                                          !%ip
      !!      dsx = dsm + dt*0.5*dsm*gs2*sneqv(i, jj) /                             &     !%ip
      !!   &        (1.e7*exp(-0.02*dsm + kn/(tavgc+273.16)-14.643))                      !%ip
                                                                                          !%ip
      !! --- ...  convert density from kg/m3 to g/cm3                                     !%ip
                                                                                          !%ip
      !!      dsx = dsx / 1000.0                                                          !%ip
                                                                                          !%ip
      !! --- ...  end of cogley et al. formulation                                        !%ip
                                                                                          !%ip
      !  --- ...  set upper/lower limit on snow density                                   !%ip
                                                                                          !%ip
                  dsx = max(min(dsx, 0.40), 0.05)                                         !%ip
                  sndens = dsx                                                            !%ip
                                                                                          !%ip
      !  --- ...  update of snow depth and density depending on liquid water              !%ip
      !           during snowmelt.  assumed that 13% of liquid water can be               !%ip
      !           stored in snow per day during snowmelt till snow density 0.40.          !%ip
                  if (tsnowc >= 0.0) then                                                 !%ip
                     dw = 0.13*dthr/24.0                                                  !%ip
                     sndens = sndens*(1.0 - dw) + dw                                      !%ip
                     if (sndens > 0.40) sndens = 0.40                                     !%ip
                  end if                                                                  !%ip
                                                                                          !%ip
      !  --- ...  calculate snow depth (cm(i, jj)) from snow water equivalent and snow    !%ip
      !           density. change snow depth units to meters                              !%ip
                  snowhc = esdc/sndens                                                    !%ip
                  snowh(i, jj) = snowhc*0.01                                              !%ip
                  !end call snowpack_gpu                                                  !%ip
                                                                                          !%i
               else                                                                       !%i
                                                                                          !%i
!         sndens = sneqv(i, jj) / snowh(i, jj)                                            !%i
!         runoff2(i, jj) = -(0.10 - sneqv(i, jj)) / dt                                    !%i
                  sneqv(i, jj) = 0.10                                                     !%i
                  snowh(i, jj) = 0.50                                                     !%i
                  sncovr(i, jj) = 1.0                                                     !%i
!         snowh(i, jj) = sneqv(i, jj) / sndens                                            !%i
                                                                                          !%i
               end if   ! end if_sneqv_block                                              !%i
                                                                                          !%i
            end if   ! end if_ice_block                                                   !%i
            !end call snopac_gpu                                                          !%i

         end if

!  --- ...  prepare sensible heat (h) for return to parent model
         sheat(i, jj) = -(ch(i, jj)*cp1*sfcprs(i, jj))&
                        /(rd1*t2v)*(th2(i, jj) - t1(i, jj))

!  --- ...  convert units and/or sign of total evap (eta(i, jj)), potential evap 
!           (etp(i, jj)), subsurface heat flux (s), and runoffs for what parent model 
!           expects convert eta(i, jj) from kg m-2 s-1 to w m-2
!     eta(i, jj) = eta(i, jj) * lsubc
!     etp(i, jj) = etp(i, jj) * lsubc
         edir(i, jj) = edir(i, jj)*lsubc
         ec(i, jj) = ec(i, jj)*lsubc

         !$acc loop seq
         do k = 1, 4
            et(i, k, jj) = et(i, k, jj)*lsubc
         end do
         ett(i, jj) = ett(i, jj)*lsubc
         esnow(i, jj) = esnow(i, jj)*lsubs
         etp(i, jj) = etp(i, jj)*((1.0 - sncovr(i, jj))*lsubc + sncovr(i, jj)*lsubs)

         if (etp(i, jj) > 0.) then
            eta(i, jj) = edir(i, jj) + ec(i, jj) + ett(i, jj) + esnow(i, jj)
         else
            eta(i, jj) = etp(i, jj)
         end if
         beta(i, jj) = eta(i, jj)/etp(i, jj)

!  --- ...  convert the sign of soil heat flux so that:
!           ssoil(i, jj)>0: warm the surface  (night time)
!           ssoil(i, jj)<0: cool the surface  (day time)
         ssoil(i, jj) = -1.0*ssoil(i, jj)
         if (ice == 0) then

!  --- ...  for the case of land (but not glacial-ice):
!           convert runoff3(i, jj) (internal layer runoff from supersat) from m
!           to m s-1 and add to subsurface runoff/baseflow (runoff2(i, jj)).
!           runoff2(i, jj) is already a rate at this point.
            runoff3(i, jj) = runoff3(i, jj)/dt
            runoff2(i, jj) = runoff2(i, jj) + runoff3(i, jj)

         else

!  --- ...  for the case of sea-ice (ice=1) or glacial-ice (ice=-1), add any
!           snowmelt directly to surface runoff (runoff1(i, jj)) since there is no
!           soil medium, and thus no call to subroutine smflx (for soil
!           moisture tendency).
            runoff1(i, jj) = snomlt(i, jj)/dt
         end if

!  --- ...  total column soil moisture in meters (soilm(i, jj)) and root-zone
!           soil moisture availability (fraction) relative to porosity/saturation
         soilm(i, jj) = -1.0*smc(i, 1, jj)*zsoil(1)
         !$acc loop seq
         do k = 2, nsoil
            soilm(i, jj) = soilm(i, jj) + smc(i, k, jj)*(zsoil(k - 1) - zsoil(k))
         end do

         soilwm = -1.0*(smcmax(i, jj) - smcwlt(i, jj))*zsoil(1)
         soilww = -1.0*(smc(i, 1, jj) - smcwlt(i, jj))*zsoil(1)
         !$acc loop seq
         do k = 2, nroot(i, jj)
            soilwm = soilwm + (smcmax(i, jj) - smcwlt(i, jj))*(zsoil(k - 1) - zsoil(k))
            soilww = soilww + (smc(i, k, jj) - smcwlt(i, jj))*(zsoil(k - 1) - zsoil(k))
         end do

         soilw(i, jj) = soilww/soilwm
                  end if
               end if
            end do
         end do
!
         return


!...................................
      end subroutine sflx_gpu
!-----------------------------------
