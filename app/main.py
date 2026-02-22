import sys
import os
from pathlib import Path

# Add CUDA/cuDNN DLLs to search path on Windows (MUST BE DONE BEFORE OTHER IMPORTS)
if sys.platform == "win32":
    import importlib.util
    
    # DLL directories to add
    dll_dirs = []
    
    # 1. nvidia packages (highly recommended for cuDNN 9 symbols)
    for pkg in ["nvidia.cublas", "nvidia.cudnn"]:
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            pkg_path = Path(spec.submodule_search_locations[0])
            bin_dir = pkg_path / "bin"
            if bin_dir.exists():
                dll_dirs.append(bin_dir)
                
    # 2. Torch's own DLLs as fallback
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.submodule_search_locations:
        torch_lib = Path(torch_spec.submodule_search_locations[0]) / "lib"
        if torch_lib.exists():
            dll_dirs.append(torch_lib)

    for ddir in dll_dirs:
        # Use os.add_dll_directory for Python 3.8+
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(ddir))
            except Exception:
                pass
        os.environ["PATH"] = str(ddir) + os.pathsep + os.environ["PATH"]

# Fix path to allow absolute imports from the project root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import uvicorn
import asyncio
from fastapi import FastAPI
from app.core.events import lifespan
from app.memory.database import init_db
from app.agent.orchestrator import KratosOrchestrator

# Automatically agree to the Coqui XTTS license (CPML)
os.environ["COQUI_TOS_AGREED"] = "1"

from app.core.logger import logger

app = FastAPI(title="Kratos Desk", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "alive", "name": "Kratos"}

async def run_voice_loop():
    # Wait for models to "warm up" (the lifespan event handles this in a real setup)
    await asyncio.sleep(2)
    orchestrator = KratosOrchestrator()
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        await orchestrator.stop()
        logger.info("Kratos is resting.")
    except Exception as e:
        logger.exception("Orchestrator failed: {}", e)

if __name__ == "__main__":
    # Initialize DB
    logger.info("Initializing database...")
    init_db()
    
    # Check for CLI mode or Web mode
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run the voice Loop directly
        try:
            asyncio.run(run_voice_loop())
        except KeyboardInterrupt:
            pass
