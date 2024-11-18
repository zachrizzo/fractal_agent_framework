# fractal_framework/agents/chatbot_agent.py

from typing import Dict, List, Any, Optional
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
    """Agent for handling chat interactions and coordinating with other agents"""

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
        self.code_vectorizer = CodeVectorizer()

        # Initialize sub-agents
        self.vector_search_agent = VectorSearchAgent("vector_search")
        self.analyze_agent = AnalyzeAgent("analyzer")
        self.add_sub_agent(self.vector_search_agent)
        self.add_sub_agent(self.analyze_agent)

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

            # Add user message to memory
            self.memory.add_message(Message(role="user", content=message))

            # Process the message
            response = await self._handle_message(message, context)

            # Add response to memory
            self.memory.add_message(Message(role="assistant", content=response))

            return FractalResult(task.id, True, result={
                "response": response,
                "context": self.memory.context
            })

        except Exception as e:
            logger.error(f"Error in ChatbotAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _handle_message(self, message: str, context: FractalContext) -> str:
        """Handle incoming message and generate response"""
        # First, try to find relevant code using vector search
        code_results = await self._find_relevant_code(message, context)

        # Analyze the code if found
        if code_results:
            analysis_results = await self._analyze_code(code_results, context)
        else:
            analysis_results = None

        # Generate response using LLM
        response = await self._generate_llm_response(
            message,
            code_results,
            analysis_results
        )

        return response

    async def _find_relevant_code(
        self,
        message: str,
        context: FractalContext
    ) -> List[Dict[str, Any]]:
        """Find relevant code using vector search"""
        search_task = FractalTask(
            id="search",
            type=TaskType.ANALYZE,
            data={
                "operation": "search",
                "query": message,
                "k": 3
            }
        )

        result = await self.vector_search_agent.execute_task(search_task, context)
        return result.result.get("similar_nodes", []) if result.success else []

    async def _analyze_code(
        self,
        code_results: List[Dict[str, Any]],
        context: FractalContext
    ) -> Dict[str, Any]:
        """Analyze found code"""
        analysis_tasks = []
        for code_result in code_results:
            task = FractalTask(
                id=f"analyze_{code_result['node_id']}",
                type=TaskType.ANALYZE,
                data={
                    "node_id": code_result["node_id"],
                    "analysis_type": "patterns"
                }
            )
            analysis_tasks.append(task)

        results = await asyncio.gather(*[
            self.analyze_agent.execute_task(task, context)
            for task in analysis_tasks
        ])

        return {
            "patterns": [r.result for r in results if r.success],
            "metrics": self._aggregate_metrics([r.result for r in results if r.success])
        }

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple analysis results"""
        # Implementation depends on your specific metrics
        return {}

    async def _generate_llm_response(
        self,
        message: str,
        code_results: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response using the configured LLM"""
        if self.llm_provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(
                message, code_results, analysis_results
            )
        elif self.llm_provider == LLMProvider.CLAUDE:
            return await self._generate_claude_response(
                message, code_results, analysis_results
            )
        elif self.llm_provider == LLMProvider.OLLAMA:
            return await self._generate_ollama_response(
                message, code_results, analysis_results
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def _generate_openai_response(
        self,
        message: str,
        code_results: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response using OpenAI"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # Format context from code results and analysis
        context = self._format_context(code_results, analysis_results)

        # Get recent conversation history
        recent_messages = self.memory.get_recent_context()

        # Prepare system message with better instruction
        system_message = """You are a helpful code assistant that explains code clearly and concisely.
When explaining code:
1. First give a high-level overview of what the code does
2. Point out key components and their purpose
3. Explain any important patterns or practices used
4. If relevant, suggest improvements or best practices

Be direct and specific in your explanations. If you find code related to what the user is asking about,
reference it explicitly and explain how it works."""

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": context}  # Add code context
        ]

        # Add conversation history
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current message with improved instruction
        user_message = f"User Question: {message}\n\nPlease analyze the code and explain it in relation to the user's question."
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def _format_context(
        self,
        code_results: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """Format code and analysis context for LLM prompting"""
        context_parts = ["Here's what I found in the codebase:"]

        # Add code context
        if code_results:
            for idx, result in enumerate(code_results, 1):
                code = result.get("code", "").strip()
                path = result.get("filePath", "unknown")
                label = result.get("label", "unnamed")

                context_parts.append(f"\nCode from {label} ({path}):")
                context_parts.append(f"```\n{code}\n```")

                # Add code analysis if available
                analysis = result.get("analysis", {})
                if analysis:
                    if "language" in analysis:
                        context_parts.append(f"Language: {analysis['language']}")
                    if "functions" in analysis:
                        context_parts.append(f"Functions: {', '.join(analysis['functions'])}")
                    if "patterns" in analysis:
                        context_parts.append(f"Patterns: {', '.join(analysis['patterns'])}")

                # Add React-specific analysis
                if "const" in code and "useState" in code:
                    context_parts.append("\nReact Component Analysis:")

                    # Extract state variables
                    state_matches = re.findall(r'const\s+\[(\w+),\s*set(\w+)\]\s*=\s*useState', code)
                    if state_matches:
                        context_parts.append("State Management:")
                        for state_var, _ in state_matches:
                            context_parts.append(f"- {state_var} (managed by useState)")

                    # Extract event handlers
                    handler_matches = re.findall(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{', code)
                    if handler_matches:
                        context_parts.append("Event Handlers:")
                        for handler in handler_matches:
                            context_parts.append(f"- {handler}")

                    # Extract JSX elements
                    jsx_elements = re.findall(r'<(\w+)[^>]*>', code)
                    if jsx_elements:
                        context_parts.append("UI Components:")
                        for element in set(jsx_elements):
                            context_parts.append(f"- {element}")

        # Add any additional analysis
        if analysis_results:
            context_parts.append("\nAdditional Analysis:")
            for key, value in analysis_results.items():
                if isinstance(value, (list, set)):
                    context_parts.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    context_parts.append(f"{key}: {value}")

        return "\n".join(context_parts)

    async def _generate_claude_response(
        self,
        message: str,
        code_results: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response using Claude"""
        from anthropic import Anthropic

        # Format context from code results and analysis
        context = self._format_context(code_results, analysis_results)

        # Get recent conversation history
        recent_messages = self.memory.get_recent_context()

        # Format conversation for Claude
        conversation = "\n\nHuman: You are a helpful code assistant. " + context + "\n\nAssistant: I understand. I'll help analyze and explain the code.\n\n"

        # Add conversation history
        for msg in recent_messages:
            prefix = "Human: " if msg.role == "user" else "Assistant: "
            conversation += f"{prefix}{msg.content}\n\n"

        # Add current message
        conversation += f"Human: {message}\n\nAssistant: "

        # Call Claude API
        anthropic = Anthropic(api_key=self.api_key)
        response = await anthropic.completions.create(
            model="claude-2",
            prompt=conversation,
            max_tokens_to_sample=1000,
            temperature=0.7
        )

        return response.completion

    async def _generate_ollama_response(
        self,
        message: str,
        code_results: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response using Ollama"""
        import json
        import aiohttp

        # Format context from code results and analysis
        context = self._format_context(code_results, analysis_results)

        # Get recent conversation history
        recent_messages = self.memory.get_recent_context()

        # Format conversation for Ollama
        messages = [{
            "role": "system",
            "content": "You are a helpful code assistant. " + context
        }]

        # Add conversation history
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        # Call Ollama API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama2",  # or other model name
                    "messages": messages,
                    "stream": False
                }
            ) as response:
                result = await response.json()
                return result["message"]["content"]
