from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.session import Base
from datetime import datetime

class AnalysisLog(Base):
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Nullable for guest scans
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    content_type = Column(String) # IMAGE, AUDIO, VIDEO, TEXT
    risk_level = Column(String) # SAFE, SUSPICIOUS, HIGH_RISK
    confidence_score = Column(Float)
    
    summary = Column(Text)
    details = Column(Text) # JSON string or pipe-separated
    
    # Forensic Evidence
    ela_image_b64 = Column(Text, nullable=True)
    
    # Blockchain Audit
    block_hash = Column(String, nullable=True)
    
    user = relationship("User", back_populates="logs")
