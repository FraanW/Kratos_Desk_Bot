import sys
from loguru import logger
from app.config import settings

def setup_logger():
    # Remove default logger
    logger.remove()
    
    # Add console sink
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file sink
    logger.add(
        settings.DATA_DIR / "kratos.log",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        level="DEBUG"
    )

setup_logger()
