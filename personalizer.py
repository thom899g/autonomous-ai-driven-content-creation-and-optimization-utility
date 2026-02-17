import logging
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class PersonalizationEngine:
    def __init__(self):
        self.user_profile_clustering = {}
        self.content_embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-mpnet-base-v2")

    def personalize(self, content: str, user_profile: Dict[str, Any]) -> str:
        """Personalizes content based on user profile and preferences."""
        try:
            # Get content embeddings
            content_embedding = self._get_embeddings(content)
            
            # Match with user's clustered profile
            matched_profile = self._find_best_match(content_embedding, user_profile['cluster_id'])
            
            # Adjust tone and language
            personalized_content = self._adjust_tone(matched_profile['preferred_tone'], content)
            
            return personalized_content
        except Exception as e:
            logging.error(f"Personalization failed: {str(e)}")
            raise

    def _get_embeddings(self, text: str) -> Dict[str, Any]:
        """Returns embeddings for the given text."""
        try:
            return self.content_embedding_model(text)[0]
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            raise

    def _find_best_match(self, content_embeddings: Dict[str, Any], cluster_id: str) -> Dict[str, Any]:
        """Finds the best matching user profile in the cluster."""
        try:
            # Compute similarity score
            similarity_score = cosine_similarity([content_embeddings], 
                                                 self.user_profile_clustering[cluster_id])
            
            # Return most similar profile
            return self.user_profile_clustering[cluster_id][similarity_score.argmax()]
        except Exception as e:
            logging.error(f