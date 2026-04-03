module vvm_land_interface
    use iso_c_binding, only: c_double, c_int, c_bool
    use index, only: jlistnum
    use param, only: my_max
    implicit none
    
    integer, parameter :: nxpvs = 7501
    real :: c1xpvs, c2xpvs, tbpvs(nxpvs)
    common/fpvscom/ c1xpvs, c2xpvs, tbpvs

contains

    subroutine run_vvm_land_wrapper(nx, ny, nsoil, dt, &
        islimsk, vegtype, soiltyp, slopetyp, &
        t1, q1, u1, v1, ps, prcp, swdn, lwdn, hgt, &
        stc, smc, slc, tskin, canopy, snwdph, &
        hflux, qflux, evap, zorl) bind(c, name="run_vvm_land_wrapper")
        
        integer(c_int), value :: nx, ny, nsoil
        real(c_double), value :: dt
        
        integer(c_int), intent(in)    :: islimsk(nx,ny), vegtype(nx,ny), soiltyp(nx,ny), slopetyp(nx,ny)
        real(c_double), intent(in)    :: t1(nx,ny), q1(nx,ny), u1(nx,ny), v1(nx,ny)
        real(c_double), intent(in)    :: ps(nx,ny), prcp(nx,ny), swdn(nx,ny), lwdn(nx,ny)
        real(c_double), intent(in)    :: hgt(nx,ny) ! VVM height (m)

        real(c_double), intent(inout) :: stc(nx,nsoil,ny), smc(nx,nsoil,ny), slc(nx,nsoil,ny)
        real(c_double), intent(inout) :: tskin(nx,ny), canopy(nx,ny), snwdph(nx,ny), zorl(nx,ny)
        real(c_double), intent(inout) :: hflux(nx,ny), qflux(nx,ny), evap(nx,ny)

        integer(c_int) :: myim(ny)
        integer(c_int) :: i, j, ncld, isot, ivegsrc, async_id, iter, lsm
        real(8)        :: xmin, xmax, xinc, x_val, t_val
        logical        :: redrag, mom4ice
        
        real(c_double) :: psi(nx,ny), prsl1(nx,ny), prslki(nx,ny)
        real(c_double) :: tg(nx,ny), z0rl(nx,ny), cd(nx,ny), cdq(nx,ny), rb(nx,ny)
        real(c_double) :: stress(nx,ny), fm(nx,ny), fh(nx,ny), ustar(nx,ny), sfcw(nx,ny)
        real(c_double) :: ddvel(nx,ny), fm10(nx,ny), fh2(nx,ny), fh10(nx,ny)
        real(c_double) :: sigmaf(nx,ny), shdmax(nx,ny), shdmin(nx,ny), tsurf(nx,ny)
        logical        :: flag_iter(nx,ny), flag_guess(nx,ny)
        
        real(c_double) :: qsurf(nx,ny), gfx(nx,ny), ep1d(nx,ny)
        real(c_double) :: sfemis(nx,ny), tgclim(nx,ny)
        real(c_double) :: snoalb(nx,ny), alb(nx,ny), albedo2(nx,ny)
        real(c_double) :: sheleg(nx,ny), tprcp(nx,ny), srflag(nx,ny), sncover(nx,ny)
        real(c_double) :: drain(nx,ny), runof(nx,ny)
        real(c_double) :: zice(nx,ny), cice(nx,ny), xtice(nx,ny), snomt(nx,ny)
        
        real(c_double) :: u10(nx,ny), v10(nx,ny), t2(nx,ny), q2(nx,ny)
        real(c_double) :: rh2(nx,ny), rh10(nx,ny)
        real(c_double) :: ro2(nx,ny)
        
        jlistnum = ny
        my_max = ny
        call set_soilveg(isot, ivegsrc)
        
        xmin = 180.0D0
        xmax = 330.0D0
        xinc = (xmax - xmin) / (nxpvs - 1)
        c2xpvs = 1.0D0 / xinc
        c1xpvs = 1.0D0 - xmin * c2xpvs
        do i = 1, nxpvs
            x_val = xmin + (i - 1) * xinc
            t_val = x_val
            tbpvs(i) = 611.2D0 * exp(17.67D0 * (t_val - 273.15D0) / (t_val - 29.65D0))
        end do
        !$acc enter data copyin(c1xpvs, c2xpvs, tbpvs)
        
        do j = 1, ny
            myim(j) = nx
        end do

        !$acc data present(islimsk, vegtype, soiltyp, slopetyp, &
        !$acc              t1, q1, u1, v1, ps, prcp, swdn, lwdn, hgt, &
        !$acc              stc, smc, slc, tskin, canopy, snwdph, hflux, qflux, evap, zorl) &
        !$acc      create(psi, prsl1, prslki, tg, z0rl, cd, cdq, rb, stress, &
        !$acc             fm, fh, ustar, sfcw, ddvel, fm10, fh2, fh10, sigmaf, &
        !$acc             shdmax, shdmin, tsurf, flag_iter, flag_guess, qsurf, &
        !$acc             gfx, ep1d, sfemis, tgclim, snoalb, alb, albedo2, &
        !$acc             sheleg, tprcp, srflag, sncover, drain, runof, zice, cice, xtice, snomt, &
        !$acc             u10, v10, t2, q2, rh2, rh10, ro2) &
        !$acc      copyin(myim)

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
                prslki(i,j) = (ps(i,j) / 100000.0D0) ** 0.286D0

                ro2(i,j) = ps(i,j) / (287.04D0 * t1(i,j) * (1.0D0 + 0.608D0 * q1(i,j)))

                flag_iter(i,j) = .true.
                flag_guess(i,j) = .false.
                
                sigmaf(i,j) = 0.8D0
                sfemis(i,j) = 0.98D0
                tgclim(i,j) = 285.0D0
                
                
                shdmin(i,j) = 0.01D0
                shdmax(i,j) = 0.99D0
                snoalb(i,j) = 0.60D0
                alb(i,j) = 0.20D0
                albedo2(i,j) = 0.0D0
                z0rl(i,j) = zorl(i,j) * 100.0D0  ! NCEP zorl to cm
                
                tprcp(i,j) = prcp(i,j) * dt / 1000.0D0  !  mm/s times dt to m
                sheleg(i,j) = snwdph(i,j) / 10.0D0
                
                ! srflag (rain or snow)
                if (t1(i,j) <= 273.16D0) then
                    srflag(i,j) = 1.0D0
                else
                    srflag(i,j) = 0.0D0
                end if
                
                tsurf(i,j) = tskin(i,j)
                tg(i,j) = tskin(i,j)
                ddvel(i,j) = 0.0D0
                qsurf(i,j) = 0.0D0
                gfx(i,j) = 0.0D0
                drain(i,j) = 0.0D0
                runof(i,j) = 0.0D0
                ep1d(i,j) = 0.0D0
                hflux(i,j) = 0.0D0
                qflux(i,j) = 0.0D0
                sncover(i,j) = 0.0D0
                snomt(i,j) = 0.0D0
                zice(i,j) = 0.0D0
                cice(i,j) = 0.0D0
                xtice(i,j) = 0.0D0
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
                    if ((iter == 1) .and. (sfcw(i,j) < 2.0D0)) then
                        flag_guess(i,j) = .true.
                    end if
                end do
            end do
            
            call sfc_ocean_gpu(myim, nx, 1, ncld, psi, u1, v1, t1, q1, &
                 tg, cd, cdq, prsl1, prslki, islimsk, ddvel, flag_iter, &
                 qsurf, gfx, qflux, hflux, ep1d, async_id)
                 
            call sfc_drv_gpu(myim, nx, nsoil, 1, ncld, psi, u1, v1, t1, q1, soiltyp, &
                 vegtype, sigmaf, sfemis, lwdn, swdn, swdn, dt, &
                 tgclim, cd, cdq, prsl1, prslki, hgt, islimsk, ddvel, slopetyp, &
                 shdmin, shdmax, snoalb, alb, flag_iter, flag_guess, isot, ivegsrc, sheleg, &
                 snwdph, tg, tprcp, srflag, smc, stc, slc, &
                 canopy, tsurf, zorl, sncover, qsurf, gfx, drain, qflux, &
                 hflux, ep1d, runof, albedo2, 1, 1, 1, async_id)
                 
            call sfc_sice_gpu(myim, nx, nsoil, 1, ncld, psi, u1, v1, t1, q1, dt, sfemis, lwdn, &
                 swdn, swdn, srflag, cd, cdq, prsl1, prslki, islimsk, ddvel, &
                 flag_iter, mom4ice, lsm, zice, cice, xtice, sheleg, &
                 tg, tprcp, stc, ep1d, snwdph, qsurf, snomt, gfx, &
                 qflux, hflux, async_id)
                 
            if (iter == 1) then
                !$acc parallel loop collapse(2) async(async_id)
                do j = 1, ny
                    do i = 1, nx
                        flag_iter(i,j) = .false.
                        flag_guess(i,j) = .false.
                        if ((islimsk(i,j) == 1) .and. (sfcw(i,j) < 2.0D0)) then
                            flag_iter(i,j) = .true.
                        end if
                    end do
                end do
            end if
        end do
        
        call sfc_diag_gpu(myim, nx, 1, ncld, psi, u1, v1, t1, q1, tg, &
                          qsurf, u10, v10, t2, q2, prslki, qflux, fm, fh, fm10, fh2, fh10, &
                          rh2, rh10, async_id)

        ! =================================================================
        ! Kinematic Flux to W/m²
        ! =================================================================
        !$acc parallel loop collapse(2) async(async_id)
        do j = 1, ny
            do i = 1, nx
                tskin(i,j) = tsurf(i,j)
                hflux(i,j) = hflux(i,j) * ro2(i,j) * 1004.6D0
                evap(i,j) = qflux(i,j) * ro2(i,j) * 2500000.0D0
            end do
        end do
        
        !$acc wait(async_id)
        !$acc end data
        !$acc exit data delete(c1xpvs, c2xpvs, tbpvs)
        
    end subroutine run_vvm_land_wrapper
end module vvm_land_interface
