from dataclasses import dataclass

@dataclass
class Email:
    """Dataclass representing an email with subject and body"""
    subject: str
    body: str

@dataclass
class EmailWithTopic:
    """Dataclass representing an email with subject, body, and topic"""
    subject: str
    body: str
    topic: str

@dataclass
class Topic:
"""Dataclass representing an email with topic and description"""
    topic: str
    description: str
    
