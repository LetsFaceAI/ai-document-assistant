from pydantic import BaseModel, Field

class TextCleaningResult(BaseModel):
    original_text: str = Field(..., description="The unmodified raw text extracted from the PDF.")
    cleaned_text: str = Field(..., description="The processed text optimized for semantic chunking.")
    original_char_count: int
    cleaned_char_count: int
    applied_rules_count: int
    processing_time_ms: float = Field(..., description="Time taken to run the cleaning pipeline in milliseconds.")