import logging
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from data_collector import MarketDataCollector
from optimizer import ContentOptimizer
from personalizer import PersonalizationEngine

class ContentGenerator:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model=self.model_name, tokenizer=self.tokenizer)
        self.market_collector = MarketDataCollector()
        self.optimizer = ContentOptimizer()
        self.personalizer = PersonalizationEngine()

    def generate_content(self, niche: str, format: str) -> Dict[str, Any]:
        """Generates content based on market data and optimization parameters."""
        try:
            # Collect relevant market data
            data = self.market_collector.collect_data(niche)
            
            # Generate initial draft using NLP model
            prompt = f"Generate a comprehensive article about {data['keyword']} in the {niche} niche."
            generated = self.summarizer(prompt, max_length=500, min_length=250, do_sample=True)
            
            # Optimize content for SEO and readability
            optimized_content = self.optimizer.optimize(generated[0]['summary'])
            
            # Personalize content based on user data
            personalized = self.personalizer.personalize(optimized_content, data['user_profile'])
            
            return {
                "status": "success",
                "content": personalized,
                "metrics": {"word_count": len(personalized), 
                           "engagement_score": self.optimizer.readability_score(personalized)}
            }
        except Exception as e:
            logging.error(f"Error in content generation: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

# Example usage
if __name__ == "__main__":
    generator = ContentGenerator()
    result = generator.generate_content("Technology", "article")
    print(result)