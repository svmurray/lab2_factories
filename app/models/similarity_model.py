import os
import json
import math
from typing import Dict, Any, List

class EmailClassifierModel:
    """Simple rule-based email classifier model"""
    
    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())
        self.email_data = self._load_email_data()

    def _load_email_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/emails.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        result = ''
        if os.stat(data_file).st_size > 0:
            with open(data_file, 'r') as f:
                result = json.load(f)
        print(result)
        return result
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def add_topic(self, topic: Dict[str, Any]):
        print('similarity_model file')
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            print('---file open')
            contents = json.load(f)
        with open(data_file, 'w') as f:
            print(contents)
            contents[topic.topic] = {'description': topic.description}
            json.dump(contents, f)

    def store_email(self, email: Dict[str, Any]):
        print('similarity_model file')
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        with open(data_file, 'r') as f:
            print('---file open')
            print(data_file)
            if os.stat(data_file).st_size > 0:
                contents = json.load(f)
            else:
                contents = {}
        with open(data_file, 'w') as f:
            contents[email.body] = {'topic': email.topic, 'subject': email.subject}
            json.dump(contents, f)
        return 'Email Added'

    def predict_similar_email(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}

        print('in predict_similar_email - model.py')
        for email in self.email_data:
            print(email)
            score = self._calculate_email_score(features, email)

        return max(scores, key=self.email_data[scores.get]['topic'])
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores

    def _calculate_email_score(self, features: Dict[str, Any], email: Dict[str, Any]) -> float:
        """Calculate similarity score based on length difference"""
         # Get email embedding from features
        email_embedding = features.get("email_embeddings_average_embedding", 0.0)
        
        # Get topic description and create embedding (description length as embedding)
        email_body = self.email_data['body'].key
        stored_email_embedding = float(len(email_body))
        
        # Calculate similarity based on inverse distance
        # Smaller distance = higher similarity
        distance = abs(email_embedding - stored_email_embedding)
        
        # Normalize to 0-1 range using exponential decay
        # e^(-distance/scale) gives values between 0 and 1
        scale = 50.0  # Adjust this to control how quickly similarity drops with distance
        similarity = math.exp(-distance / scale)
        
        return similarity
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate similarity score based on length difference"""
        # Get email embedding from features
        email_embedding = features.get("email_embeddings_average_embedding", 0.0)
        
        # Get topic description and create embedding (description length as embedding)
        topic_description = self.topic_data[topic]['description']
        topic_embedding = float(len(topic_description))
        
        # Calculate similarity based on inverse distance
        # Smaller distance = higher similarity
        distance = abs(email_embedding - topic_embedding)
        
        # Normalize to 0-1 range using exponential decay
        # e^(-distance/scale) gives values between 0 and 1
        scale = 50.0  # Adjust this to control how quickly similarity drops with distance
        similarity = math.exp(-distance / scale)
        
        return similarity
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}
