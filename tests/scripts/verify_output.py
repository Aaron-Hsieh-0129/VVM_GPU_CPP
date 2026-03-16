import h5py
import numpy as np
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="VVM Regression Test Verification")
    parser.add_argument('--baseline', required=True, help="Path to golden baseline NetCDF")
    parser.add_argument('--test_out', required=True, help="Path to test output NetCDF")
    parser.add_argument('--vars', nargs='+', required=True, help="Variables to compare (e.g., th xi eta zeta)")
    parser.add_argument('--tol', type=float, default=1e-5, help="Tolerance for floating point comparison")
    args = parser.parse_args()

    try:
        f_base = h5py.File(args.baseline, 'r')
        f_test = h5py.File(args.test_out, 'r')
    except FileNotFoundError as e:
        print(f"[ERROR] Unable to find file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unable to open file: {e}")
        sys.exit(1)

    if 'Step0' not in f_base or 'Step0' not in f_test:
        print("[ERROR] 'Step0' is not in HDF5 group! Please check the output format.")
        f_base.close()
        f_test.close()
        sys.exit(1)

    step_base = f_base['Step0']
    step_test = f_test['Step0']
    
    passed = True
    for var in args.vars:
        if var not in step_base or var not in step_test:
            print(f"[FAIL] Variable {var} missing!")
            passed = False
            continue
            
        data_base = step_base[var][:]
        data_test = step_test[var][:]
        
        if data_base.shape != data_test.shape:
            print(f"[FAIL] Variable {var} has inconsistent dimension. Baseline: {data_base.shape}, Test: {data_test.shape}.")
            passed = False
            continue

        is_close = np.allclose(data_base, data_test, atol=args.tol, rtol=args.tol)
        
        if not is_close:
            max_diff = np.max(np.abs(data_base - data_test))
            print(f"[FAIL] Different variable {var}! Maximum error: {max_diff}.")
            passed = False
        else:
            print(f"[PASS] Variable {var} pass.")

    f_base.close()
    f_test.close()

    if not passed:
        sys.exit(1)

    print("All variables pass")
    sys.exit(0)

if __name__ == "__main__":
    main()
