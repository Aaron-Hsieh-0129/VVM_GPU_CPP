module vvm_land_interface
    use iso_c_binding, only: c_double, c_float, c_int, c_bool
    use index, only: jlistnum
    use param, only: my_max
    use namelist_soilveg, only: z0_data, ems1_data, ems2_data
    implicit none

#ifdef VVM_USE_DOUBLE_PRECISION
    integer, parameter :: c_vvm_real = c_double
#else
    integer, parameter :: c_vvm_real = c_float
#endif
    
    integer, parameter :: nxpvs = 7501
    real :: c1xpvs, c2xpvs, tbpvs(nxpvs)
    common/fpvscom/ c1xpvs, c2xpvs, tbpvs

contains

    subroutine run_vvm_land_wrapper(use_tco_ocean, nx, ny, nsoil, dt, &
        islimsk, vegtype, soiltyp, slopetyp, &
        sigmaf, sfemis, alb, shdmin, shdmax, &
        t1, q1, u1, v1, ps, prcp, swdn, lwdn, swnet, hgt, prslki_in, &
        stc, smc, slc, tskin, canopy, snwdph, sneqv, &
        hflux, qflux, evap, gfx, zorl, cmx, & 
        lai, rdlai2d) bind(c, name="run_vvm_land_wrapper")
        
        integer(c_int), value :: use_tco_ocean
        integer(c_int), value :: nx, ny, nsoil
        real(c_vvm_real), value :: dt
        
        integer(c_int), intent(in)    :: islimsk(nx,ny), vegtype(nx,ny), soiltyp(nx,ny), slopetyp(nx,ny)
        real(c_vvm_real), intent(in)    :: t1(nx,ny), q1(nx,ny), u1(nx,ny), v1(nx,ny)
        real(c_vvm_real), intent(in)    :: ps(nx,ny), prcp(nx,ny), swdn(nx,ny), lwdn(nx,ny), swnet(nx,ny)
        real(c_vvm_real), intent(in)    :: hgt(nx,ny) ! VVM height (m)
        real(c_vvm_real), intent(in)    :: prslki_in(nx,ny) ! VVM portion (pi(sfc)/pi(air)) 

        real(c_vvm_real), intent(in)    :: sigmaf(nx,ny)
        real(c_vvm_real), intent(in)    :: shdmin(nx,ny), shdmax(nx,ny)

        real(c_vvm_real), intent(inout) :: alb(nx,ny), sfemis(nx,ny)
        real(c_vvm_real), intent(inout) :: stc(nx,nsoil,ny), smc(nx,nsoil,ny), slc(nx,nsoil,ny)
        real(c_vvm_real), intent(inout) :: tskin(nx,ny), canopy(nx,ny), snwdph(nx,ny), sneqv(nx,ny), zorl(nx,ny), cmx(nx, ny)
        real(c_vvm_real), intent(inout) :: hflux(nx,ny), qflux(nx,ny), evap(nx,ny), gfx(nx,ny)

        real(c_vvm_real), intent(inout) :: lai(nx, ny)
        logical(c_bool), value          :: rdlai2d

        real(8), parameter :: g = 9.806_c_vvm_real
        real(8), parameter :: r = 287.04_c_vvm_real
        real(8), parameter :: cp = 1004.5_c_vvm_real
        real(8), parameter :: hltm = 2500000.0_c_vvm_real

        integer(c_int) :: myim(ny)
        integer(c_int) :: i, j, ncld, isot, ivegsrc, async_id, iter, lsm
        real(8)        :: xmin, xmax, xinc, x_val, t_val
        logical        :: redrag, mom4ice
        
        real(c_vvm_real) :: psi(nx,ny), prsl1(nx,ny), prslki(nx,ny)
        real(c_vvm_real) :: tg(nx,ny), z0rl(nx,ny), cd(nx,ny), cdq(nx,ny), rb(nx,ny)
        real(c_vvm_real) :: stress(nx,ny), fm(nx,ny), fh(nx,ny), ustar(nx,ny), sfcw(nx,ny)
        real(c_vvm_real) :: ddvel(nx,ny), fm10(nx,ny), fh2(nx,ny), fh10(nx,ny)
        real(c_vvm_real) :: tsurf(nx,ny)
        logical        :: flag_iter(nx,ny), flag_guess(nx,ny)
        
        real(c_vvm_real) :: qsurf(nx,ny), ep1d(nx,ny)
        real(c_vvm_real) :: lwdn_eff(nx,ny), swnet_eff(nx,ny)
        real(c_vvm_real) :: tgclim(nx,ny)
        real(c_vvm_real) :: snoalb(nx,ny), albedo2(nx,ny)
        real(c_vvm_real) :: tprcp(nx,ny), srflag(nx,ny), sncover(nx,ny)
        real(c_vvm_real) :: drain(nx,ny), runof(nx,ny)
        real(c_vvm_real) :: zice(nx,ny), cice(nx,ny), xtice(nx,ny), snomt(nx,ny)
        
        real(c_vvm_real) :: u10(nx,ny), v10(nx,ny), t2(nx,ny), q2(nx,ny)
        real(c_vvm_real) :: rh2(nx,ny), rh10(nx,ny)
        real(c_vvm_real) :: ro2(nx,ny)
        
        jlistnum = ny
        my_max = ny
        isot = 1
        ivegsrc = 1
        call set_soilveg(isot, ivegsrc)
        
        xmin = 180.0_c_vvm_real
        xmax = 330.0_c_vvm_real
        xinc = (xmax - xmin) / (nxpvs - 1)
        c2xpvs = 1.0_c_vvm_real / xinc
        c1xpvs = 1.0_c_vvm_real - xmin * c2xpvs
        do i = 1, nxpvs
            x_val = xmin + (i - 1) * xinc
            t_val = x_val
            tbpvs(i) = 611.2_c_vvm_real * exp(17.67_c_vvm_real * (t_val - 273.15_c_vvm_real) / (t_val - 29.65_c_vvm_real))
        end do
        
        do j = 1, ny
            myim(j) = nx
        end do
        !$acc enter data copyin(c1xpvs, c2xpvs, tbpvs, myim)

        !$acc data present(islimsk, vegtype, soiltyp, slopetyp, &
        !$acc              sigmaf, sfemis, alb, shdmin, shdmax, &
        !$acc              t1, q1, u1, v1, ps, prcp, swdn, lwdn, swnet, hgt, &
        !$acc              stc, smc, slc, tskin, canopy, snwdph, sneqv, hflux, qflux, evap, gfx, zorl, cmx) &
        !$acc      create(psi, prsl1, prslki, tg, z0rl, cd, cdq, rb, stress, &
        !$acc             fm, fh, ustar, sfcw, ddvel, fm10, fh2, fh10, &
        !$acc             tsurf, flag_iter, flag_guess, qsurf, &
        !$acc             lwdn_eff, swnet_eff, &
        !$acc             ep1d, tgclim, snoalb, albedo2, &
        !$acc             tprcp, srflag, sncover, drain, runof, zice, cice, xtice, snomt, &
        !$acc             u10, v10, t2, q2, rh2, rh10, ro2)

        ncld = 1
        isot = 1; ivegsrc = 1; async_id = 1
        lsm = 1
        redrag = .false.
        mom4ice = .false.
        
        !$acc parallel loop collapse(2) async(async_id)
        do j = 1, ny
            do i = 1, nx
                psi(i,j) = ps(i,j)
                prsl1(i,j) = ps(i,j)
                prslki(i,j) = prslki_in(i,j)

                ro2(i,j) = ps(i,j) / (r * t1(i,j) * (1.0_c_vvm_real + 0.608_c_vvm_real * q1(i,j)))

                flag_iter(i,j) = .true.
                flag_guess(i,j) = .false.
                
                tgclim(i,j) = 285.0_c_vvm_real 
                snoalb(i,j) = 0.60_c_vvm_real
                albedo2(i,j) = 0.0_c_vvm_real
                z0rl(i,j) = zorl(i,j) * 100.0_c_vvm_real  ! NCEP zorl to cm
                
                tprcp(i,j) = prcp(i,j) * dt / 1000.0_c_vvm_real  !  mm/s times dt to m
                
                ! srflag (rain or snow)
                if (t1(i,j) <= 273.15_c_vvm_real) then
                    srflag(i,j) = 1.0_c_vvm_real
                else
                    srflag(i,j) = 0.0_c_vvm_real
                end if
                
                tsurf(i,j) = tskin(i,j)
                tg(i,j) = tskin(i,j)
                ddvel(i,j) = 0.0_c_vvm_real
                qsurf(i,j) = 0.0_c_vvm_real
                drain(i,j) = 0.0_c_vvm_real
                runof(i,j) = 0.0_c_vvm_real
                ep1d(i,j) = 0.0_c_vvm_real
                hflux(i,j) = 0.0_c_vvm_real
                qflux(i,j) = 0.0_c_vvm_real
                sncover(i,j) = 0.0_c_vvm_real
                snomt(i,j) = 0.0_c_vvm_real
                zice(i,j) = 0.0_c_vvm_real
                cice(i,j) = 0.0_c_vvm_real
                xtice(i,j) = 0.0_c_vvm_real


                if (islimsk(i,j) == 1) then
                    sfemis(i,j) = ems2_data(vegtype(i,j)) + sigmaf(i,j) * &
                                  (ems1_data(vegtype(i,j)) - ems2_data(vegtype(i,j)))
                else
                    sfemis(i,j) = 0.98D0
                end if

                lwdn_eff(i,j) = lwdn(i,j) * sfemis(i,j)
                swnet_eff(i,j) = swdn(i,j) * (1.0_c_vvm_real - alb(i,j))
            end do
        end do
        
        ! =================================================================
        ! 2-step iteration (sfcw < 2m/s unstable)
        ! =================================================================
        do iter = 1, 2
            call sfc_diff_gpu(myim, nx, 1, ncld, psi, u1, v1, t1, q1, &
                 hgt, snwdph, tg, z0rl, cd, cdq, rb, prsl1, prslki, islimsk, &
                 stress, fm, fh, ustar, sfcw, ddvel, fm10, fh2, fh10, sigmaf, &
                 vegtype, shdmax, ivegsrc, tsurf, flag_iter, redrag, async_id)
                 
            !$acc parallel loop collapse(2) async(async_id)
            do j = 1, ny
                do i = 1, nx
                    if ((iter == 1) .and. (sfcw(i,j) < 2.0_c_vvm_real)) then
                        flag_guess(i,j) = .true.
                    end if
                end do
            end do
            
            if (use_tco_ocean == 1) then
                call sfc_ocean_gpu(myim, nx, 1, ncld, psi, u1, v1, t1, q1, &
                     tg, cd, cdq, prsl1, prslki, islimsk, ddvel, flag_iter, &
                     qsurf, gfx, qflux, hflux, ep1d, async_id)
            end if
                 
            call sfc_drv_gpu(myim, nx, nsoil, 1, ncld, psi, u1, v1, t1, q1, soiltyp, &
                 vegtype, sigmaf, sfemis, lwdn_eff, swdn, swnet_eff, dt, &
                 tgclim, cd, cdq, prsl1, prslki, hgt, islimsk, ddvel, slopetyp, &
                 shdmin, shdmax, snoalb, alb, flag_iter, flag_guess, isot, ivegsrc, sneqv, &
                 snwdph, tg, tprcp, srflag, smc, stc, slc, &
                 canopy, tsurf, z0rl, sncover, qsurf, gfx, drain, qflux, &
                 hflux, ep1d, runof, albedo2, 1, 1, 1, lai, rdlai2d, async_id)
                 
            call sfc_sice_gpu(myim, nx, nsoil, 1, ncld, psi, u1, v1, t1, q1, dt, sfemis, lwdn_eff, &
                 swdn, swnet_eff, srflag, cd, cdq, prsl1, prslki, islimsk, ddvel, &
                 flag_iter, mom4ice, lsm, zice, cice, xtice, sneqv, &
                 tg, tprcp, stc, ep1d, snwdph, qsurf, snomt, gfx, &
                 qflux, hflux, async_id)
                 
            if (iter == 1) then
                !$acc parallel loop collapse(2) async(async_id)
                do j = 1, ny
                    do i = 1, nx
                        flag_iter(i,j) = .false.
                        flag_guess(i,j) = .false.
                        if ((islimsk(i,j) == 1) .and. (sfcw(i,j) < 2.0_c_vvm_real)) then
                            flag_iter(i,j) = .true.
                        end if
                    end do
                end do
            end if
        end do
        
        call sfc_diag_gpu(myim, nx, 1, ncld, psi, u1, v1, t1, q1, tg, &
                          qsurf, u10, v10, t2, q2, prslki, qflux, fm, fh, fm10, fh2, fh10, &
                          rh2, rh10, async_id)

        !$acc parallel loop collapse(2) async(async_id)
        do j = 1, ny
            do i = 1, nx

                if (islimsk(i,j) == 0 .and. use_tco_ocean == 0) then
                    hflux(i,j) = 0.0_c_vvm_real
                    evap(i,j)  = 0.0_c_vvm_real
                else
                    hflux(i,j) = hflux(i,j) * ro2(i,j) / ((ps(i,j) / 100000.0_c_vvm_real)**(r/cp))
                    evap(i,j)  = qflux(i,j) * ro2(i,j)
                    tskin(i,j) = tsurf(i,j)
                endif

                if (islimsk(i,j) == 0) then
                    ! ocean 
                    zorl(i,j) = max(ustar(i,j) * ustar(i,j) * 0.014_c_vvm_real / g, 1.5D-5)
                else
                    ! land 
                    zorl(i,j) = z0rl(i,j) / 100.0_c_vvm_real ! cm -> m
                end if

                cmx(i,j) = cd(i,j) * sfcw(i,j)

                alb(i,j) = albedo2(i,j)
            end do
        end do
        
        !$acc wait(async_id)
        !$acc end data
        !$acc exit data delete(c1xpvs, c2xpvs, tbpvs, myim)
        
    end subroutine run_vvm_land_wrapper
end module vvm_land_interface
