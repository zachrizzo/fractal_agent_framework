# fractal_framework/agents/chatbot_agent.py

from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import json

from .enhanced_fractal_agent import EnhancedFractalAgent, AgentCapability
from core.fractal_task import FractalTask
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from utils.vector_store import LocalVectorStore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"

@dataclass
class Message:
    """Enhanced chat message representation"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)

class ChatMemory:
    """Enhanced chat memory system with context tracking"""
    def __init__(
        self,
        max_messages: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.embedding_model = SentenceTransformer(embedding_model)
        self.context_history: List[Dict[str, Any]] = []

    async def add_message(self, message: Message):
        """Add and analyze a new message"""
        message.embedding = self.embedding_model.encode(message.content)
        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get current conversation context summary"""
        recent_messages = self.messages[-5:] if self.messages else []
        return {
            "recent_messages": [
                {
                    "content": msg.content,
                    "context": msg.context,
                    "timestamp": msg.timestamp
                }
                for msg in recent_messages
            ],
            "context_history": self.context_history[-5:] if self.context_history else []
        }

class ChatbotAgent(EnhancedFractalAgent):
    """Context-aware chatbot agent"""

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        api_key: str,
        vector_store: Optional[LocalVectorStore] = None
    ):
        super().__init__(name, capabilities=[
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.CODE_GENERATION,
            AgentCapability.DOCUMENTATION
        ])

        self.llm_provider = llm_provider
        self.api_key = api_key
        self.memory = ChatMemory()
        self.vector_store = vector_store or LocalVectorStore(api_key=api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    async def _analyze_task(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task intent and required capabilities"""
        prompt = f"""Analyze this user message and context to determine the best way to handle it.

        Message: {message}

        Recent Context: {json.dumps(context.get('recent_context', {}), indent=2)}

        Determine:
        1. Primary intent (understand, execute, respond)
        2. Required capabilities
        3. Context importance (0-1)
        4. Execution strategy

        Return JSON with these fields:
        {
            "intent": "string",
            "capabilities": ["string"],
            "context_importance": float,
            "strategy": "string",
            "requires_code_context": boolean
        }"""

        try:
            response = await self._generate_llm_response(
                prompt=prompt,
                system_message="You are an expert system analyzer. Return valid JSON only."
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Task analysis error: {e}")
            return {
                "intent": "respond",
                "capabilities": ["CODE_ANALYSIS"],
                "context_importance": 0.5,
                "strategy": "direct",
                "requires_code_context": False
            }

    async def _process_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Process task with enhanced context awareness"""
        try:
            message = task.data.get("message", "")
            if not message:
                return FractalResult(task.id, False, error="No message provided")

            # Add message to memory
            user_message = Message(role="user", content=message)
            await self.memory.add_message(user_message)

            # Get current context
            context_summary = self.memory.get_context_summary()

            # Analyze task
            task_analysis = await self._analyze_task(message, context_summary)

            # Generate response
            response = await self._generate_response(message, {
                "task_analysis": task_analysis,
                "conversation_context": context_summary
            })

            return FractalResult(
                task_id=task.id,
                success=True,
                result=response
            )

        except Exception as e:
            logger.error(f"Error in task processing: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _generate_response(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate contextual response"""
        system_message = """You are an intelligent assistant that:
        1. Provides clear, direct responses
        2. Uses available context effectively
        3. Explains technical concepts clearly
        4. Maintains conversation continuity"""

        prompt = self._create_response_prompt(message, context)

        try:
            response = await self._generate_llm_response(
                prompt=prompt,
                system_message=system_message
            )

            return {
                "content": response,
                "type": context["task_analysis"]["intent"],
                "context_used": bool(context.get("code_context"))
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": "I encountered an error while generating a response. Please try again.",
                "type": "error",
                "context_used": False
            }

    def _create_response_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Create detailed prompt with context"""
        parts = [f"User Message: {message}\n"]

        if conv_context := context.get("conversation_context", {}).get("recent_messages"):
            parts.append("\nConversation Context:")
            for msg in conv_context[-3:]:  # Last 3 messages
                parts.append(f"- {msg['content']}")

        parts.append("\nProvide a response that:")
        parts.append("1. Directly addresses the user's question/request")
        parts.append("2. Incorporates relevant context")
        parts.append("3. Is clear and well-structured")

        return "\n".join(parts)

    async def _generate_llm_response(
        self,
        prompt: str,
        system_message: str,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using configured LLM"""
        if self.llm_provider == LLMProvider.OPENAI:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
