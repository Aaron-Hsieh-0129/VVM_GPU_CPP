find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_vvm_test TEST_NAME VARS_TO_TEST)
    add_test(NAME Run_${TEST_NAME} 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 
                     $<TARGET_FILE:vvm> --io-tasks 1 ${TEST_DIR}/configs/${TEST_NAME}.json
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    separate_arguments(VARS_LIST UNIX_COMMAND "${VARS_TO_TEST}")
    
    add_test(NAME Verify_${TEST_NAME} 
             COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/scripts/verify_output.py 
             --baseline ${TEST_DIR}/baselines/${TEST_NAME}.h5 
             --test_out ${CMAKE_BINARY_DIR}/testing_output_${TEST_NAME}/vvm_output_000001.h5
             --vars ${VARS_LIST}
             --tol 1e-6)

    set_tests_properties(Verify_${TEST_NAME} PROPERTIES DEPENDS Run_${TEST_NAME})
endfunction()

add_vvm_test(advection_u "u th xi eta zeta")
add_vvm_test(advection_v "v th xi eta zeta")
add_vvm_test(advection_w "w th xi eta")
