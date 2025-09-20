from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features
        print(f'Email obj: {email}')
        predicted_topic = self.model.predict(features, email.predict_type)
        topic_scores = self.model.get_topic_scores(features)
        email_scores = self.model.get_email_scores(features)
        
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "email_scores": email_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }

    def add_topic(self, topic: Dict[str, Any]):
        """Add topic to topics list"""
        if topic.topic not in self.model.topics:
            print('email_topic_inference file')
            self.model.add_topic(topic)
            return 'Topic Added'
        else:
            return 'Topic already exists'

    def store_email(self, email: Dict[str, Any]):
        """Store email in emails list"""
        print('email_topic_inference file')
        return self.model.store_email(email)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }
