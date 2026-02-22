from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class JournalEntry(Base):
    __tablename__ = "journal_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    raw_text = Column(Text, nullable=False)
    summary = Column(Text)
    mood = Column(String(50))
    tags = Column(String(255))  # Comma separated

class WeeklySummary(Base):
    __tablename__ = "weekly_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    week_start = Column(DateTime, default=datetime.utcnow)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
