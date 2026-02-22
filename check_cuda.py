import os
import sys
import ctypes
from pathlib import Path
import importlib.util

def check_dlls():
    pkgs = ["nvidia.cublas", "nvidia.cudnn"]
    for pkg in pkgs:
        print(f"\nChecking {pkg}...")
        spec = importlib.util.find_spec(pkg)
        if not spec:
            print(f"FAILED: Could not find spec for {pkg}")
            continue
            
        pkg_path = None
        if spec.origin and spec.origin != 'namespace':
            pkg_path = Path(spec.origin).parent
        elif spec.submodule_search_locations:
            pkg_path = Path(spec.submodule_search_locations[0])
            
        if not pkg_path:
            print(f"FAILED: Could not determine path for {pkg}")
            continue
            
        bin_dir = pkg_path / "bin"
        print(f"Package path: {pkg_path}")
        print(f"Bin dir: {bin_dir}")
        
        if not bin_dir.exists():
            print(f"FAILED: Bin directory does not exist: {bin_dir}")
            continue
            
        print(f"Adding {bin_dir} to DLL directory and PATH...")
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
        
        # List files in bin
        print("Files in bin:")
        for f in bin_dir.glob("*.dll"):
            print(f"  - {f.name}")
            # Try load it
            try:
                lib = ctypes.CDLL(str(f))
                print(f"    SUCCESS: Loaded {f.name}")
            except Exception as e:
                print(f"    FAILED: Could not load {f.name}: {e}")

if __name__ == "__main__":
    check_dlls()
