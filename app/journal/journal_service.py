from sqlalchemy.orm import Session
from app.memory.models import JournalEntry, WeeklySummary
from app.memory.embeddings import get_embedding_service
from app.memory.vector_store import get_vector_store
from app.memory.summarizer import get_summarizer
from app.core.logger import logger
from datetime import datetime

class JournalService:
    def __init__(self, db: Session):
        self.db = db
        self.embeddings = get_embedding_service()
        self.vector_store = get_vector_store()
        self.summarizer = get_summarizer()

    async def add_entry(self, text: str):
        logger.info("Adding journal entry...")
        
        # 1. Create entry in DB
        entry = JournalEntry(raw_text=text)
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        
        # 2. Generate summary and embedding async
        summary = await self.summarizer.summarize_entry(text)
        embedding = self.embeddings.embed(text)
        
        # 3. Update entry with summary
        entry.summary = summary
        self.db.commit()
        
        # 4. Add to vector store
        self.vector_store.add(entry.id, embedding)
        
        logger.info("Journal entry added (ID: {}). Summary: {}", entry.id, summary)
        
        # 5. Check for weekly compression
        await self._check_compression()
        return entry

    async def search_memory(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.embeddings.embed(query)
        entry_ids = self.vector_store.search(query_embedding, top_k)
        
        if not entry_ids:
            return []
        
        entries = self.db.query(JournalEntry).filter(JournalEntry.id.in_(entry_ids)).all()
        return [e.summary or e.raw_text[:100] for e in entries]

    async def get_latest_weekly_summary(self) -> str:
        latest = self.db.query(WeeklySummary).order_by(WeeklySummary.created_at.desc()).first()
        return latest.summary_text if latest else ""

    async def _check_compression(self):
        count = self.db.query(JournalEntry).count()
        if count > 0 and count % 25 == 0:
            logger.info("Compression threshold reached ({} entries). Generating weekly summary...", count)
            # Take last 25 entries
            entries = self.db.query(JournalEntry).order_by(JournalEntry.timestamp.desc()).limit(25).all()
            texts = [e.raw_text for e in entries]
            
            summary_text = await self.summarizer.summarize_weekly(texts)
            
            weekly = WeeklySummary(
                week_start=entries[-1].timestamp,
                summary_text=summary_text
            )
            self.db.add(weekly)
            self.db.commit()
            logger.info("Weekly summary created.")
