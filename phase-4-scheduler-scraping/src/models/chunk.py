"""
Chunk model for RAG system
Represents a chunk of text with metadata
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a chunk of text with metadata"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional fields for chunk properties
    chunk_type: Optional[str] = Field(None, description="Type of chunk (e.g., 'structured', 'narrative')")
    section_type: Optional[str] = Field(None, description="Section type (e.g., 'basic_info', 'performance')")
    fund_name: Optional[str] = Field(None, description="Name of the mutual fund")
    source_url: Optional[str] = Field(None, description="Source URL of the data")
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the chunk"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists"""
        return key in self.metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'chunk_type': self.chunk_type,
            'section_type': self.section_type,
            'fund_name': self.fund_name,
            'source_url': self.source_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary"""
        if 'created_at' in data:
            if isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of chunk"""
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"Chunk(id={self.id}, text='{preview}', fund={self.fund_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"Chunk(id='{self.id}', fund_name='{self.fund_name}', "
                f"section_type='{self.section_type}', "
                f"text_length={len(self.text)})")
