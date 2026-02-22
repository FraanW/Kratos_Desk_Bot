import sys
import os
from pathlib import Path

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
from app.core.logger import logger

# Add NVIDIA DLLs to search path on Windows
if sys.platform == "win32":
    import os
    import importlib.util
    
    for pkg in ["nvidia.cublas", "nvidia.cudnn"]:
        spec = importlib.util.find_spec(pkg)
        if spec:
            pkg_path = None
            if spec.origin and spec.origin != 'namespace':
                pkg_path = Path(spec.origin).parent
            elif spec.submodule_search_locations:
                pkg_path = Path(spec.submodule_search_locations[0])
            
            if pkg_path:
                bin_dir = pkg_path / "bin"
                if bin_dir.exists():
                    logger.info(f"Adding DLL directory to search path and PATH: {bin_dir}")
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]

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
        orchestrator.stop()
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
