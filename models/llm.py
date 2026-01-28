from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SimpleLLM:
    def __init__(self):
        """Initialize a free, lightweight LLM"""
        try:
            print("Loading LLM model... (This may take a minute first time)")
            # Small, fast model - runs on CPU
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_length=512,
                device=-1  # CPU
            )
            print("✅ LLM loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            self.generator = None
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on question and retrieved context"""
        if not self.generator:
            return f"LLM not available. Retrieved context: {context[:500]}"
        
        try:
            # Create prompt
            prompt = f"""Answer the following question based only on the context provided.

Context: {context[:1500]}

Question: {question}

Provide a clear, concise answer based on the context above."""
            
            # Generate answer
            result = self.generator(
                prompt, 
                max_length=200, 
                min_length=30,
                do_sample=False
            )
            answer = result[0]['generated_text'].strip()
            
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer. Retrieved context: {context[:500]}"

# Global instance
llm_instance = SimpleLLM()