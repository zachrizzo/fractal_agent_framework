# fractal_framework/agents/chatbot_agent.py

from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import json

from .fractal_agent import FractalAgent
from .vector_search_agent import VectorSearchAgent
from .analyze_agent import AnalyzeAgent
from core.fractal_task import FractalTask, TaskType
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from utils.vector_store import LocalVectorStore, CodeVectorizer
import re
import ast


logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"

@dataclass
class Message:
    """Represents a chat message"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChatMemory:
    """Maintains chat history and context with enhanced features"""
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_messages: int = 50
    conversation_topics: Dict[str, float] = field(default_factory=dict)

    def _update_conversation_topics(self, message: str):
        """Update conversation topics based on message content"""
        # Simple keyword-based topic tracking
        topics = {
            "login": ["login", "authentication", "auth", "signin"],
            "code_structure": ["structure", "architecture", "design", "pattern"],
            "debugging": ["bug", "error", "issue", "problem", "fix"],
            "feature": ["feature", "functionality", "implement", "add"],
            "documentation": ["doc", "documentation", "comment", "explain"]
        }

        # Decay existing topic weights
        for topic in self.conversation_topics:
            self.conversation_topics[topic] *= 0.8

        # Update weights based on new message
        message_lower = message.lower()
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                self.conversation_topics[topic] = self.conversation_topics.get(topic, 0) + 1.0

        # Remove topics with very low weights
        self.conversation_topics = {
            k: v for k, v in self.conversation_topics.items()
            if v > 0.1
        }

    def get_active_topics(self, threshold: float = 0.5) -> List[str]:
        """Get currently active conversation topics"""
        return [
            topic for topic, weight in self.conversation_topics.items()
            if weight > threshold
        ]

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current conversation context"""
        return {
            "active_topics": self.get_active_topics(),
            "last_code_context": self.context.get("last_code_reference"),
            "last_analysis": self.context.get("last_analysis"),
            "recent_tasks": [
                msg.metadata.get("task_type")
                for msg in self.messages[-5:]
                if msg.metadata.get("task_type")
            ]
        }

    def add_message(self, message: Message):
        """Add message to history with enhanced context tracking"""
        # Add message to history
        self.messages.append(message)

        # Update context based on message content
        if message.metadata.get("code_context"):
            self.context["last_code_reference"] = message.metadata["code_context"]

        if message.metadata.get("analysis"):
            self.context["last_analysis"] = message.metadata["analysis"]

        if message.metadata.get("task_type"):
            self.context["last_task"] = message.metadata["task_type"]

        # Maintain memory size
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

        # Track conversation topics for better context
        self._update_conversation_topics(message.content)

    def get_recent_context(self, n: int = 5) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

class ChatbotAgent(FractalAgent):
    """Agent for handling chat interactions about code in any programming language"""

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        api_key: str,
        vector_store: Optional[LocalVectorStore] = None
    ):
        super().__init__(name, "chat")
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.memory = ChatMemory()
        self.vector_store = vector_store or LocalVectorStore()

    async def _process_task(
        self,
        task: FractalTask,
        context: FractalContext,
        subtask_results: List[FractalResult]
    ) -> FractalResult:
        """Process chat tasks"""
        try:
            message = task.data.get("message", "")
            if not message:
                return FractalResult(task.id, False, error="No message provided")

            # Add user message to memory with task metadata
            self.memory.add_message(Message(
                role="user",
                content=message,
                metadata={"task_type": task.type}
            ))

            # Process the message
            response = await self._handle_message(message, context)

            # Add response to memory
            self.memory.add_message(Message(
                role="assistant",
                content=response,
                metadata={"task_type": "response"}
            ))

            return FractalResult(task.id, True, result={
                "response": response,
                "context": self.memory.get_context_summary()
            })

        except Exception as e:
            logger.error(f"Error in ChatbotAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _handle_message(self, message: str, context: FractalContext) -> str:
        """Handle incoming message and generate response"""
        try:
            # Search for relevant code
            search_results = self.vector_store.search_similar(
                query_text=message,
                k=3,
                threshold=0.1  # Lower threshold for better recall
            )

            # Format code context
            code_context = self._format_code_context(search_results)

            # Generate response
            response = await self._generate_llm_response(message, code_context)

            return response

        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return "I encountered an error while processing your message. Please try again."

    def _format_code_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        """Format code search results into a readable context"""
        if not search_results:
            return "I don't have any code in my database that matches your query."

        context_parts = []

        for _, similarity, metadata in search_results:
            if 'code' in metadata:
                file_info = []
                if metadata.get('filePath'):
                    file_info.append(metadata['filePath'])
                if metadata.get('label'):
                    file_info.append(metadata['label'])

                context_parts.append("\nCode from: " + ' - '.join(file_info))
                context_parts.append(f"```\n{metadata['code'].strip()}\n```")

        if context_parts:
            return "Here's the relevant code I found:\n" + "\n".join(context_parts)
        return "While I have some code indexed, I couldn't find any relevant code for your query."

    async def _generate_llm_response(self, message: str, code_context: str) -> str:
        """Generate response using the configured LLM"""
        if self.llm_provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(message, code_context)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def _generate_openai_response(self, message: str, code_context: str) -> str:
        """Generate response using OpenAI"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # Get recent conversation history
        recent_messages = self.memory.get_recent_context(n=5)

        # Prepare system message with language-agnostic instructions
        system_message = """You are a code assistant that can help with code in any programming language.
        When explaining code:
        1. First identify the programming language and framework/libraries being used
        2. Explain the purpose and functionality of the code
        3. Break down complex parts and explain their roles
        4. Point out any notable patterns or techniques used
        5. If asked, suggest improvements while considering language best practices

        Always reference specific parts of the code when explaining."""

        # Prepare messages for OpenAI
        messages = [{"role": "system", "content": system_message}]

        # Add code context explicitly
        if code_context:
            messages.append({
                "role": "system",
                "content": code_context
            })

        # Add conversation history
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current message with specific instruction
        query_message = f"User Question: {message}\n\nPlease analyze the code above (if any) and provide a detailed response."
        messages.append({
            "role": "user",
            "content": query_message
        })

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
