"""
Advanced AI Wikipedia Chatbot - Industry Grade Implementation

This module implements a sophisticated Wikipedia-based chatbot with:
- Enhanced NLP and query understanding
- Disambiguation handling
- Conversation memory
- Professional error handling
- Modular, maintainable architecture

Author: AI Engineer
Version: 2.0 - Industry Grade
"""

import random
import json
import torch
import wikipedia
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

@dataclass
class ChatMessage:
    """Data class for storing chat messages with metadata."""
    text: str
    sender: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class QueryProcessor:
    """Advanced query processing and normalization engine."""
    
    def __init__(self):
        # Comprehensive abbreviation mapping
        self.abbreviations = {
            # Technical terms
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'ar': 'augmented reality',
            'vr': 'virtual reality',
            'iot': 'internet of things',
            'cpu': 'central processing unit',
            'gpu': 'graphics processing unit',
            'ram': 'random access memory',
            'os': 'operating system',
            'db': 'database',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'url': 'uniform resource locator',
            'sql': 'structured query language',
            'nosql': 'not only sql',
            
            # Scientific terms
            'dna': 'deoxyribonucleic acid',
            'rna': 'ribonucleic acid',
            'ph': 'potential of hydrogen',
            'laser': 'light amplification by stimulated emission of radiation',
            'radar': 'radio detection and ranging',
            'sonar': 'sound navigation and ranging',
            
            # Organizations
            'nasa': 'national aeronautics and space administration',
            'un': 'united nations',
            'eu': 'european union',
            'who': 'world health organization',
            'fbi': 'federal bureau of investigation',
            'cia': 'central intelligence agency',
        }
        
        # Vague query indicators
        self.vague_patterns = [
            r'^(tell me about|what is|describe|explain|who is|what are)',
            r'^(help|info|information about)$',
            r'^(something|anything|nothing)$',
        ]
        
        # Non-informational query patterns
        self.non_info_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)',
            r'^(bye|goodbye|see you|farewell)',
            r'^(thanks|thank you|appreciate)',
            r'^(how are you|how do you do)',
            r'^(what can you do|who are you|what are you)',
        ]
    
    def preprocess_query(self, query: str) -> str:
        """
        Comprehensive query preprocessing for optimal Wikipedia search.
        
        Args:
            query: Raw user input
            
        Returns:
            Normalized and enhanced query string
        """
        if not query or not query.strip():
            return ""
        
        query = query.strip()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^\w\s\-\.\']', ' ', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Handle abbreviations
        query_lower = query.lower()
        for abbr, expansion in self.abbreviations.items():
            # Replace standalone abbreviations or abbreviations followed by space
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, query_lower):
                query = re.sub(pattern, expansion, query, flags=re.IGNORECASE)
                break
        
        # Handle common name formats
        query = self._normalize_names(query)
        
        return query.strip()
    
    def _normalize_names(self, query: str) -> str:
        """Normalize person names for better Wikipedia search."""
        # Handle initials like "J.K. Rowling"
        query = re.sub(r'\b([A-Z]\.)+([A-Z][a-z]+)', r'\1 \2', query)
        
        # Handle names with spaces and periods
        query = re.sub(r'\b([A-Z]\. [A-Z]\.)+', lambda m: m.group().replace('. ', ''), query)
        
        return query
    
    def is_vague_query(self, query: str) -> bool:
        """
        Detect vague or ambiguous queries that need clarification.
        
        Args:
            query: Preprocessed query
            
        Returns:
            True if query is vague and needs clarification
        """
        query_lower = query.lower()
        
        # Check against vague patterns
        for pattern in self.vague_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Single-word queries that are too generic
        words = query_lower.split()
        if len(words) == 1:
            generic_words = {'something', 'anything', 'info', 'help', 'what', 'who', 'when', 'where', 'why', 'how'}
            return words[0] in generic_words
        
        return False
    
    def is_non_informational(self, query: str) -> bool:
        """
        Detect non-informational queries (conversational, social).
        
        Args:
            query: Raw user input
            
        Returns:
            True if query is non-informational
        """
        query_lower = query.lower()
        
        for pattern in self.non_info_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def generate_clarification_question(self, query: str) -> str:
        """
        Generate appropriate clarification questions for vague queries.
        
        Args:
            query: The vague query
            
        Returns:
            Clarification question string
        """
        query_lower = query.lower()
        
        if 'tell me about' in query_lower or 'what is' in query_lower:
            return "Could you be more specific? For example, are you asking about a person, concept, event, or place?"
        
        if 'help' in query_lower or 'info' in query_lower:
            return "I can help you find information on Wikipedia. What specific topic would you like to know about?"
        
        return "I'd be happy to help! Could you tell me more specifically what you're interested in learning about?"
    
    def enhance_search_variations(self, query: str) -> List[str]:
        """
        Generate multiple search query variations for better results.
        
        Args:
            query: Preprocessed query
            
        Returns:
            List of query variations
        """
        variations = [query]
        words = query.lower().split()
        
        # Single-word enhancements
        if len(words) == 1:
            word = words[0]
            variations.extend([
                f"{word} (concept)",
                f"{word} (topic)",
                f"what is {word}",
                f"{word} in science",
                f"{word} in technology",
                f"{word} (disambiguation)"
            ])
        
        # Name enhancements (2-3 words likely to be names)
        elif 2 <= len(words) <= 3:
            full_name = ' '.join(words)
            variations.extend([
                f"{full_name} (person)",
                f"{full_name} (scientist)",
                f"{full_name} (author)",
                f"{full_name} (artist)"
            ])
        
        # Concept enhancements
        if any(word in ['theory', 'concept', 'principle', 'law'] for word in words):
            variations.append(f"{query} in physics")
            variations.append(f"{query} in science")
        
        return variations[:6]  # Limit to 6 variations

class WikipediaSearchEngine:
    """Advanced Wikipedia search with intelligent disambiguation."""
    
    def __init__(self):
        wikipedia.set_lang("en")
        self.max_summary_length = 400
        self.min_summary_length = 50
    
    def search_smart(self, query: str) -> Optional[str]:
        """
        Intelligent Wikipedia search with disambiguation handling.
        
        Args:
            query: Search query string
            
        Returns:
            Wikipedia summary or None if not found
        """
        try:
            # First attempt: Direct summary
            summary = wikipedia.summary(query, sentences=4, auto_suggest=False)
            if self._is_valid_summary(summary):
                return self._format_summary(summary)
                
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by trying most relevant options
            return self._handle_disambiguation(e.options)
            
        except wikipedia.exceptions.PageError:
            # Page not found, fall back to search
            pass
            
        except Exception as e:
            # Log error and continue to fallback
            print(f"Wikipedia direct search error: {e}")
        
        # Fallback: Search results
        return self._fallback_search(query)
    
    def _handle_disambiguation(self, options: List[str]) -> Optional[str]:
        """Handle Wikipedia disambiguation pages intelligently."""
        # Try the most relevant options first
        priority_options = []
        other_options = []
        
        for option in options[:8]:  # Limit to first 8 options
            option_lower = option.lower()
            
            # Prioritize main topics (avoid "(disambiguation)" pages)
            if not any(skip in option_lower for skip in ['disambiguation', 'redirect']):
                if any(keyword in option_lower for keyword in ['(concept)', '(topic)', 'theory', 'principle']):
                    priority_options.append(option)
                else:
                    other_options.append(option)
        
        # Try priority options first
        for option in priority_options + other_options:
            try:
                summary = wikipedia.summary(option, sentences=3, auto_suggest=False)
                if self._is_valid_summary(summary):
                    return self._format_summary(summary, source=option)
            except:
                continue
        
        return None
    
    def _fallback_search(self, query: str) -> Optional[str]:
        """Fallback search using Wikipedia search API."""
        try:
            search_results = wikipedia.search(query, results=8)
            if not search_results:
                return None
            
            # Try each search result
            for result in search_results:
                try:
                    summary = wikipedia.summary(result, sentences=3, auto_suggest=False)
                    if self._is_valid_summary(summary):
                        return self._format_summary(summary, source=result)
                except:
                    continue
                    
        except Exception as e:
            print(f"Wikipedia fallback search error: {e}")
        
        return None
    
    def _is_valid_summary(self, summary: str) -> bool:
        """Check if summary is meaningful and not too short."""
        return (
            summary and 
            len(summary.strip()) >= self.min_summary_length and
            not summary.lower().startswith(('may refer to', 'could refer to'))
        )
    
    def _format_summary(self, summary: str, source: str = None) -> str:
        """Format and clean the Wikipedia summary."""
        # Remove excessive whitespace and newlines
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Remove Wikipedia-specific formatting
        summary = re.sub(r'\([^)]*\)', lambda m: m.group() if not m.group().startswith('()') else '', summary)
        
        # Smart truncation if too long
        if len(summary) > self.max_summary_length:
            sentences = summary.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > self.max_summary_length - 20:
                    break
                truncated.append(sentence)
                current_length += len(sentence) + 2
            
            summary = '. '.join(truncated)
            if not summary.endswith('.'):
                summary += '...'
        
        return summary

class AdvancedWikipediaChatbot:
    """
    Industry-grade Wikipedia chatbot with advanced NLP capabilities.
    
    Features:
    - Intelligent query understanding and normalization
    - Disambiguation handling
    - Conversation memory
    - Professional error handling
    - Modular architecture
    """
    
    def __init__(self):
        """Initialize the chatbot with all components."""
        # Device setup for neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.wikipedia_engine = WikipediaSearchEngine()
        
        # Load ML model for conversational intents
        self._load_model()
        
        # Load conversational intents
        with open("intents.json", "r") as f:
            self.intents = json.load(f)
        
        # Conversation state
        self.conversational_tags = ["greeting", "goodbye"]
        self.conversation_history: List[ChatMessage] = []
        
        # Response cache for repeated questions
        self.response_cache: Dict[str, Tuple[str, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _load_model(self):
        """Load the trained neural network model for intent classification."""
        try:
            FILE = "data.pth"
            data = torch.load(FILE, map_location=self.device)
            
            input_size = data["input_size"]
            hidden_size = data["hidden_size"]
            output_size = data["output_size"]
            all_words = data["all_words"]
            tags = data["tags"]
            model_state = data["model_state"]
            
            self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            
            self.all_words = all_words
            self.tags = tags
            
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
            self.model = None
    
    def _classify_intent(self, msg: str) -> Tuple[str, float]:
        """
        Classify user intent using the neural network.
        
        Args:
            msg: User message
            
        Returns:
            Tuple of (intent_tag, confidence_score)
        """
        if not self.model:
            return "unknown", 0.0
        
        try:
            sentence = tokenize(msg)
            X = bag_of_words(sentence, self.all_words)
            X = torch.from_numpy(X).unsqueeze(0).to(self.device)
            
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            
            tag = self.tags[predicted.item()]
            probs = torch.softmax(output, dim=1)
            confidence = probs[0][predicted.item()].item()
            
            return tag, confidence
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "unknown", 0.0
    
    def _handle_conversational_intent(self, tag: str) -> Optional[str]:
        """
        Handle conversational intents (greeting, goodbye, etc.).
        
        Args:
            tag: Intent tag
            
        Returns:
            Appropriate response or None
        """
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return None
    
    def _get_cached_response(self, query: str) -> Optional[str]:
        """Check cache for existing response to avoid repeated API calls."""
        if query in self.response_cache:
            response, timestamp = self.response_cache[query]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.cache_ttl:
                return response
            else:
                del self.response_cache[query]
        return None
    
    def _cache_response(self, query: str, response: str):
        """Cache a response for future use."""
        self.response_cache[query] = (response, datetime.now())
    
    def get_wikipedia_response(self, msg: str) -> str:
        """
        Get intelligent Wikipedia response with advanced NLP processing.
        
        Args:
            msg: User message
            
        Returns:
            Formatted Wikipedia response
        """
        # Check cache first
        cached_response = self._get_cached_response(msg)
        if cached_response:
            return cached_response
        
        # Preprocess query
        processed_query = self.query_processor.preprocess_query(msg)
        
        if not processed_query:
            return "I didn't understand that. Could you please rephrase your question?"
        
        # Check for vague queries
        if self.query_processor.is_vague_query(msg):
            return self.query_processor.generate_clarification_question(msg)
        
        # Generate search variations
        variations = self.query_processor.enhance_search_variations(processed_query)
        
        # Try each variation
        for query in variations:
            result = self.wikipedia_engine.search_smart(query)
            if result:
                # Cache the successful response
                self._cache_response(msg, result)
                return result
        
        # Enhanced fallback message
        return (
            "I couldn't find specific information about that topic on Wikipedia. "
            "Try asking about a well-known person, concept, scientific principle, "
            "or historical event. For example: 'Albert Einstein', 'quantum physics', "
            "or 'World War II'."
        )
    
    def get_response(self, msg: str) -> str:
        """
        Main response generation method with comprehensive logic.
        
        Args:
            msg: User message
            
        Returns:
            Chatbot response
        """
        if not msg or not msg.strip():
            return "Please ask me something! I'm here to help you learn from Wikipedia."
        
        msg = msg.strip()
        
        # Check for non-informational queries
        if self.query_processor.is_non_informational(msg):
            tag, confidence = self._classify_intent(msg)
            
            if confidence > 0.75 and tag in self.conversational_tags:
                response = self._handle_conversational_intent(tag)
                if response:
                    return response
        
        # For all other queries, use Wikipedia
        return self.get_wikipedia_response(msg)

# Global chatbot instance (singleton pattern)
_chatbot_instance = None

def get_chatbot_instance() -> AdvancedWikipediaChatbot:
    """Get the singleton chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = AdvancedWikipediaChatbot()
    return _chatbot_instance

def get_response(msg: str) -> str:
    """
    Legacy interface for backward compatibility with Flask app.
    
    Args:
        msg: User message
        
    Returns:
        Chatbot response
    """
    chatbot = get_chatbot_instance()
    return chatbot.get_response(msg)

# Export main classes for testing and extension
__all__ = [
    'AdvancedWikipediaChatbot',
    'QueryProcessor', 
    'WikipediaSearchEngine',
    'ChatMessage',
    'get_response',
    'get_chatbot_instance'
]
