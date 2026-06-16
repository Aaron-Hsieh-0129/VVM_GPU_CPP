module sfc_diff_wrf_exact_mod
  use machine, only: kind_phys
  implicit none
  private
  public :: sfc_diff_wrf_myj_scalar, myjsfcinit

  real(kind=kind_phys), parameter :: CP_WRF = 1004.6_kind_phys

  integer, parameter :: ITRMX = 5
  real(kind=kind_phys), parameter :: EXCML   = 0.0001_kind_phys
  real(kind=kind_phys), parameter :: EXCMS   = 0.0001_kind_phys
  real(kind=kind_phys), parameter :: VKARMAN = 0.4_kind_phys
  real(kind=kind_phys), parameter :: ZTFC    = 1.0_kind_phys
  real(kind=kind_phys), parameter :: ELOCP   = 2.72e6_kind_phys / CP_WRF
  real(kind=kind_phys), parameter :: EPSU2   = 1.0e-6_kind_phys
  real(kind=kind_phys), parameter :: EPSUST  = 1.0e-9_kind_phys
  real(kind=kind_phys), parameter :: SQVISC  = 258.2_kind_phys
  real(kind=kind_phys), parameter :: RIC     = 0.505_kind_phys
  real(kind=kind_phys), parameter :: EPSZT   = 1.0e-28_kind_phys

  integer, parameter :: KZTM  = 10001
  integer, parameter :: KZTM2 = KZTM - 2

  real(kind=kind_phys), parameter :: WWST   = 1.2_kind_phys
  real(kind=kind_phys), parameter :: WWST2  = WWST * WWST
  real(kind=kind_phys), parameter :: ZTMIN2 = -5.0_kind_phys

  real(kind=kind_phys), parameter :: RD_WRF   = 287.04_kind_phys
  real(kind=kind_phys), parameter :: G_WRF    = 9.81_kind_phys
  real(kind=kind_phys), parameter :: P0_WRF   = 100000.0_kind_phys
  real(kind=kind_phys), parameter :: RCP_WRF  = RD_WRF / CP_WRF
  real(kind=kind_phys), parameter :: P608_WRF = 0.608_kind_phys

  real(kind=kind_phys), save :: dzeta2
  real(kind=kind_phys), save :: ztmax2
  real(kind=kind_phys), save :: psih2(KZTM)
  real(kind=kind_phys), save :: psim2(KZTM)
  logical, save :: wrf_tables_ready = .false.

!$acc declare create(dzeta2, ztmax2, psih2, psim2)

contains

  subroutine myjsfcinit()
    integer :: k
    real(kind=kind_phys) :: x, zeta1, zeta2, zrng1, zrng2
    real(kind=kind_phys) :: ztmax1, dzeta1
    real(kind=kind_phys), parameter :: pihf = 3.1415926_kind_phys / 2.0_kind_phys
    real(kind=kind_phys), parameter :: eps  = 1.0e-6_kind_phys
    real(kind=kind_phys), parameter :: ztmin1 = -5.0_kind_phys

    if (.not. wrf_tables_ready) then
      ztmax1 = 1.0_kind_phys
      ztmax2 = 1.0_kind_phys

      zrng1 = ztmax1 - ztmin1
      zrng2 = ztmax2 - ztmin2

      dzeta1 = zrng1 / real(KZTM-1, kind_phys)
      dzeta2 = zrng2 / real(KZTM-1, kind_phys)

      zeta1 = ztmin1
      zeta2 = ztmin2

      do k = 1, KZTM
        if (zeta2 < 0.0_kind_phys) then
          x = sqrt(sqrt(1.0_kind_phys - 16.0_kind_phys*zeta2))
          psim2(k) = -2.0_kind_phys*log((x+1.0_kind_phys)/2.0_kind_phys) &
                     -log((x*x+1.0_kind_phys)/2.0_kind_phys) &
                     +2.0_kind_phys*atan(x) - pihf
          psih2(k) = -2.0_kind_phys*log((x*x+1.0_kind_phys)/2.0_kind_phys)
        else
          psim2(k) = 0.7_kind_phys*zeta2 &
                     +0.75_kind_phys*zeta2*(6.0_kind_phys-0.35_kind_phys*zeta2) &
                     *exp(-0.35_kind_phys*zeta2)
          psih2(k) = 0.7_kind_phys*zeta2 &
                     +0.75_kind_phys*zeta2*(6.0_kind_phys-0.35_kind_phys*zeta2) &
                     *exp(-0.35_kind_phys*zeta2)
        endif

        if (k == KZTM) then
          ztmax1 = zeta1
          ztmax2 = zeta2
        endif

        zeta1 = zeta1 + dzeta1
        zeta2 = zeta2 + dzeta2
      enddo

      ztmax1 = ztmax1 - eps
      ztmax2 = ztmax2 - eps

!$acc update device(dzeta2, ztmax2, psih2, psim2)

      wrf_tables_ready = .true.
    endif
  end subroutine myjsfcinit

  subroutine sfc_diff_wrf_myj_scalar(zsl, zsl_wind, z0, z0base, sfcprs, tz0, &
                                     tlow, qz0, qlow, sfcspd, czil, rib, &
                                     akms, akhs, vegtyp, isurban, iz0tlnd, &
                                     ustar_out)
!$acc routine seq
    real(kind=kind_phys), intent(in) :: zsl ! height from ground to grid center
    real(kind=kind_phys), intent(in) :: zsl_wind ! Height above ground (m) of atmospheric wind fields
    real(kind=kind_phys), intent(in) :: z0
    real(kind=kind_phys), intent(in) :: z0base
    real(kind=kind_phys), intent(in) :: sfcprs
    real(kind=kind_phys), intent(in) :: tz0
    real(kind=kind_phys), intent(in) :: tlow
    real(kind=kind_phys), intent(in) :: qz0
    real(kind=kind_phys), intent(in) :: qlow
    real(kind=kind_phys), intent(in) :: sfcspd
    real(kind=kind_phys), intent(in) :: czil
    integer, intent(in) :: vegtyp
    integer, intent(in) :: isurban
    integer, intent(in) :: iz0tlnd
    real(kind=kind_phys), intent(inout) :: akms
    real(kind=kind_phys), intent(inout) :: akhs
    real(kind=kind_phys), intent(out) :: rib
    real(kind=kind_phys), intent(out) :: ustar_out

    real(kind=kind_phys) :: thlow, thz0, thelow, cwmlow
    real(kind=kind_phys) :: ustar, rlmo
    integer :: itr, k
    real(kind=kind_phys) :: czil_local, zilfc
    real(kind=kind_phys) :: a, b, btgh, btgx, cxchl, cxchs, czetmax, dthv, du2
    real(kind=kind_phys) :: elfc, pshz, pshzl, psmz, psmzl, rdzt
    real(kind=kind_phys) :: rlogt, rlogu, rz, rzst, rzsu, simh, simm, tem, thm
    real(kind=kind_phys) :: ustark, wstar2, zetalt, zetalu, zetat, zetau
    real(kind=kind_phys) :: zq, zslt, zslu, zt, zu, zzil

    ! Literal active _KWM_VERSION_ SFCDIF_MYJ logic.
    thlow = tlow * (P0_WRF/sfcprs) ** RCP_WRF
    thz0  = tz0  * (P0_WRF/sfcprs) ** RCP_WRF
    thelow = thlow

    cxchl = EXCML / zsl

    btgx = G_WRF / thlow
    elfc = VKARMAN * btgx

    btgh = btgx * 1000.0_kind_phys

    thm = (thelow + thz0) * 0.5_kind_phys
    tem = (tlow + tz0) * 0.5_kind_phys

    a = thm * P608_WRF
    b = (ELOCP/tem - 1.0_kind_phys - P608_WRF) * thm
    cwmlow = 0.0_kind_phys

    dthv = ((thelow-thz0) * ((qlow+qz0+cwmlow)*(0.5_kind_phys*P608_WRF)+1.0_kind_phys) &
            +(qlow-qz0+cwmlow)*a + cwmlow*b)

    du2 = max(sfcspd*sfcspd, EPSU2)
    rib = btgx * dthv * zsl_wind * zsl_wind / du2 / zsl

    zu = z0
    zt = zu * ZTFC
    zslu = zsl_wind + zu
    rzsu = zslu / zu
    rlogu = log(rzsu)
    zslt = zsl + zu

    if ((iz0tlnd == 0) .or. (vegtyp == isurban)) then
      zilfc = -czil * VKARMAN * SQVISC
    else
      czil_local = 10.0_kind_phys ** (-0.40_kind_phys * (z0 / 0.07_kind_phys))
      zilfc = -czil_local * VKARMAN * SQVISC
    endif

    czetmax = 10.0_kind_phys
    if (dthv > 0.0_kind_phys) then
      if (rib < RIC) then
        zzil = zilfc * (1.0_kind_phys + (rib/RIC)*(rib/RIC)*czetmax)
      else
        zzil = zilfc * (1.0_kind_phys + czetmax)
      endif
    else
      zzil = zilfc
    endif

    if (btgh * akhs * dthv /= 0.0_kind_phys) then
      wstar2 = WWST2 * abs(btgh * akhs * dthv) ** (2.0_kind_phys/3.0_kind_phys)
    else
      wstar2 = 0.0_kind_phys
    endif
    ustar = max(sqrt(akms * sqrt(du2 + wstar2)), EPSUST)

    do itr = 1, ITRMX
      zt = max(exp(zzil * sqrt(ustar*z0base)) * z0base, EPSZT)

      rzst = zslt / zt
      rlogt = log(rzst)

      rlmo = elfc * akhs * dthv / ustar**3
      zetalu = zslu * rlmo
      zetalt = zslt * rlmo
      zetau  = zu   * rlmo
      zetat  = zt   * rlmo

      zetalu = min(max(zetalu, ZTMIN2), ztmax2)
      zetalt = min(max(zetalt, ZTMIN2), ztmax2)
      zetau  = min(max(zetau,  ZTMIN2/rzsu), ztmax2/rzsu)
      zetat  = min(max(zetat,  ZTMIN2/rzst), ztmax2/rzst)

      rz = (zetau - ZTMIN2) / dzeta2
      k = int(rz)
      rdzt = rz - real(k, kind_phys)
      k = min(k, KZTM2)
      k = max(k, 0)
      psmz = (psim2(k+2)-psim2(k+1))*rdzt + psim2(k+1)

      rz = (zetalu - ZTMIN2) / dzeta2
      k = int(rz)
      rdzt = rz - real(k, kind_phys)
      k = min(k, KZTM2)
      k = max(k, 0)
      psmzl = (psim2(k+2)-psim2(k+1))*rdzt + psim2(k+1)

      simm = psmzl - psmz + rlogu

      rz = (zetat - ZTMIN2) / dzeta2
      k = int(rz)
      rdzt = rz - real(k, kind_phys)
      k = min(k, KZTM2)
      k = max(k, 0)
      pshz = (psih2(k+2)-psih2(k+1))*rdzt + psih2(k+1)

      rz = (zetalt - ZTMIN2) / dzeta2
      k = int(rz)
      rdzt = rz - real(k, kind_phys)
      k = min(k, KZTM2)
      k = max(k, 0)
      pshzl = (psih2(k+2)-psih2(k+1))*rdzt + psih2(k+1)

      simh = pshzl - pshz + rlogt

      ustark = ustar * VKARMAN
      akms = max(ustark/simm, cxchl)
      akhs = max(ustark/simh, cxchl)

      if (dthv <= 0.0_kind_phys) then
        wstar2 = WWST2 * abs(btgh*akhs*dthv) ** (2.0_kind_phys/3.0_kind_phys)
      else
        wstar2 = 0.0_kind_phys
      endif
      ustar = max(sqrt(akms * sqrt(du2 + wstar2)), EPSUST)
    enddo

    ustar_out = ustar
  end subroutine sfc_diff_wrf_myj_scalar

end module sfc_diff_wrf_exact_mod

subroutine sfc_diff_gpu(myim, im, lev, ncld, &
     zsl, zsl_wind, z0, z0base, sfcprs, tz0, tlow, qz0, qlow, sfcspd, &
     czil, vegtype, isurban, iz0tlnd, flag_iter, &
     cm, ch, rb, stress, ustar, async_id)

  ! WRF/MYJ coefficient calculator for VVM GPU experiments.
  !
  ! Contract for this WRF path:
  !   cm == AKMS, m/s, same quantity returned by WRF SFCDIF_MYJ.
  !   ch == AKHS, m/s, same quantity returned by WRF SFCDIF_MYJ.
  !
  ! This is not the old GFS convention where cm/ch are dimensionless.

  use machine, only: kind_phys
  use param, only: my_max
  use index, only: jlistnum
  use sfc_diff_wrf_exact_mod, only: myjsfcinit, sfc_diff_wrf_myj_scalar
  implicit none

  integer, intent(in) :: im, lev, ncld, async_id
  integer, intent(in) :: myim(my_max)
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: zsl, zsl_wind
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: z0, z0base
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: sfcprs
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: tz0, tlow
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: qz0, qlow
  real(kind=kind_phys), dimension(im,my_max), intent(in) :: sfcspd, czil
  integer, dimension(im,my_max), intent(in) :: vegtype
  integer, intent(in) :: isurban, iz0tlnd
  logical, dimension(im,my_max), intent(in) :: flag_iter
  real(kind=kind_phys), dimension(im,my_max), intent(inout) :: cm, ch
  real(kind=kind_phys), dimension(im,my_max), intent(inout) :: rb, stress, ustar

  integer :: i, jj
  real(kind=kind_phys) :: akms, akhs, rib, ust

!$acc parallel loop gang vector collapse(2) private(i,jj,akms,akhs,rib,ust) async(async_id)
  do jj = 1, jlistnum
    do i = 1, im
      if (i <= myim(jj)) then
          akms = cm(i,jj)
          akhs = ch(i,jj)
          ! Aaron - debug print
          ! if (i == 1 .and. jj == 1) then
          !   print *, 'GPU_URBAN_BEFORE_SFCDIF', i, jj, &
          !        'zsl=', zsl(i,jj), &
          !        'zsl_wind=', zsl_wind(i,jj), &
          !        'z0=', z0(i,jj), &
          !        'z0base=', z0base(i,jj), &
          !        'sfcprs=', sfcprs(i,jj), &
          !        'tz0=', tz0(i,jj), &
          !        'tlow=', tlow(i,jj), &
          !        'qz0=', qz0(i,jj), &
          !        'qlow=', qlow(i,jj), &
          !        'sfcspd=', sfcspd(i,jj), &
          !        'czil=', czil(i,jj), &
          !        'rib=', rib, &
          !        'akms=', akms, &
          !        'akhs=', akhs, &
          !        'vegtype=', vegtype(i,jj), &
          !        'isurban=', isurban, &
          !        'iz0tlnd=', iz0tlnd, &
          !        'ustar=', ust
          ! endif

          call sfc_diff_wrf_myj_scalar(zsl(i,jj), zsl_wind(i,jj), z0(i,jj), z0base(i,jj), &
               sfcprs(i,jj), tz0(i,jj), tlow(i,jj), qz0(i,jj), qlow(i,jj), &
               sfcspd(i,jj), czil(i,jj), rib, akms, akhs, vegtype(i,jj), &
               isurban, iz0tlnd, ust)

          cm(i,jj) = akms
          ch(i,jj) = akhs
          rb(i,jj) = rib
          ustar(i,jj) = ust

          ! WRF SFCDIF_MYJ does not return stress. This is only a diagnostic
          ! consistent with AKMS = Cd * |V|.
          stress(i,jj) = akms * sfcspd(i,jj)
          ! Aaron - debug print
          ! if (i == 1 .and. jj == 1) then
          !   print *, 'GPU_URBAN_AFTER_SFCDIF', i, jj, &
          !        'zsl=', zsl(i,jj), &
          !        'zsl_wind=', zsl_wind(i,jj), &
          !        'z0=', z0(i,jj), &
          !        'z0base=', z0base(i,jj), &
          !        'sfcprs=', sfcprs(i,jj), &
          !        'tz0=', tz0(i,jj), &
          !        'tlow=', tlow(i,jj), &
          !        'qz0=', qz0(i,jj), &
          !        'qlow=', qlow(i,jj), &
          !        'sfcspd=', sfcspd(i,jj), &
          !        'czil=', czil(i,jj), &
          !        'rib=', rib, &
          !        'akms=', akms, &
          !        'akhs=', akhs, &
          !        'vegtype=', vegtype(i,jj), &
          !        'isurban=', isurban, &
          !        'iz0tlnd=', iz0tlnd, &
          !        'ustar=', ust
          ! endif
      endif
    enddo
  enddo

end subroutine sfc_diff_gpu
