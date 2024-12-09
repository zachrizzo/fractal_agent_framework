# fractal_framework/agents/chatbot_agent.py

from typing import Dict, List, Any, Optional, Tuple, Set
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
from .enhanced_fractal_agent import EnhancedFractalAgent, AgentCapability

import re
import ast

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    detected_topics: Dict[str, float] = field(default_factory=dict)

class ConversationState:
    """Tracks and analyzes conversation state and dynamics"""
    def __init__(self):
        self.topic_history: List[Dict[str, Any]] = []
        self.message_gaps: List[float] = []
        self.topic_changes: List[float] = []

    def update(self, new_topics: Dict[str, float], timestamp: datetime):
        """Update conversation state with new topics"""
        if self.topic_history:
            # Calculate time gap between messages
            last_time = self.topic_history[-1].get('timestamp', timestamp)
            if isinstance(last_time, datetime):
                time_gap = (timestamp - last_time).total_seconds()
            else:
                time_gap = 0  # Default if timestamp missing
            self.message_gaps.append(time_gap)
            logger.debug(f"Calculated time gap between messages: {time_gap} seconds")

            # Calculate topic change magnitude
            old_topics = {k: v for k, v in self.topic_history[-1].items() if k != 'timestamp'}
            topic_change = sum(abs(new_topics.get(t, 0) - old_topics.get(t, 0))
                             for t in set(new_topics) | set(old_topics))
            self.topic_changes.append(topic_change)
            logger.debug(f"Calculated topic change magnitude: {topic_change}")

            self.topic_history.append({'timestamp': timestamp, **new_topics})
        else:
            self.topic_history.append({'timestamp': timestamp, **new_topics})
            logger.debug("Initialized topic history with first message topics.")

    def get_topic_stability(self) -> float:
        """Calculate topic stability score based on recent history"""
        if not self.topic_changes:
            return 1.0
        recent_changes = self.topic_changes[-5:]
        stability = 1.0 - min(1.0, sum(recent_changes) / len(recent_changes))
        logger.debug(f"Calculated topic stability: {stability}")
        return stability

    def get_conversation_velocity(self) -> float:
        """Calculate conversation velocity based on message timing"""
        if not self.message_gaps:
            return 0.0
        recent_gaps = self.message_gaps[-5:]
        avg_gap = sum(recent_gaps) / len(recent_gaps)
        velocity = 1.0 / (1.0 + avg_gap)  # Normalize to 0-1
        logger.debug(f"Calculated conversation velocity: {velocity}")
        return velocity

@dataclass
class TopicHierarchy:
    """Represents hierarchical relationships between topics"""
    topics: Dict[str, Set[str]] = field(default_factory=lambda: {
        "backend": {"api", "database", "authentication", "performance"},
        "frontend": {"ui", "ux", "styling", "components"},
        "infrastructure": {"deployment", "scaling", "monitoring", "security"},
        "development": {"testing", "debugging", "refactoring", "documentation"},
        "architecture": {"design_patterns", "system_design", "integration", "microservices"}
    })

    def get_related_topics(self, topic: str) -> Set[str]:
        """Get all related topics including parent and siblings"""
        related = set()
        # Find parent topics
        for parent, children in self.topics.items():
            if topic in children:
                related.add(parent)
                related.update(children)  # Add siblings
        # Add direct children if topic is a parent
        if topic in self.topics:
            related.update(self.topics[topic])
        logger.debug(f"Related topics for '{topic}': {related}")
        return related

class ChatMemory:
    """Enhanced chat memory system with sophisticated topic tracking"""

    def __init__(
        self,
        max_messages: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_topic_weight: float = 0.1,
        base_decay_rate: float = 0.8
    ):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.conversation_state = ConversationState()
        self.topic_hierarchy = TopicHierarchy()
        self.min_topic_weight = min_topic_weight
        self.base_decay_rate = base_decay_rate

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.debug(f"Initialized embedding model: {embedding_model}")

        # Enhanced topic definitions with variations and related terms
        self.topic_definitions = {
            "authentication": {
                "keywords": ["auth", "login", "signin", "credentials", "jwt", "oauth"],
                "importance": 0.8
            },
            "debugging": {
                "keywords": ["bug", "error", "exception", "crash", "troubleshoot", "fix"],
                "importance": 0.9
            },
            "performance": {
                "keywords": ["optimization", "speed", "latency", "bottleneck", "efficient"],
                "importance": 0.85
            },
            # Add more topics as needed
        }

        # Initialize topic embeddings
        self.topic_embeddings = self._initialize_topic_embeddings()
        logger.debug("Initialized topic embeddings.")

    def _initialize_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for each topic based on their keywords"""
        topic_embeddings = {}
        for topic, definition in self.topic_definitions.items():
            # Combine keywords into a representative sentence
            topic_text = f"{topic} " + " ".join(definition["keywords"])
            embedding = self.embedding_model.encode(topic_text)
            topic_embeddings[topic] = embedding
            logger.debug(f"Created embedding for topic '{topic}'.")
        return topic_embeddings

    async def add_message(self, message: Message):
        """Add and analyze a new message"""
        logger.debug(f"Adding new message: {message.role} - {message.content}")

        # Generate message embedding
        message.embedding = self.embedding_model.encode(message.content)
        logger.debug("Generated embedding for the new message.")

        # Detect topics
        message.detected_topics = await self._detect_topics(message)
        logger.debug(f"Detected topics in message: {message.detected_topics}")

        # Update conversation state
        self.conversation_state.update(message.detected_topics, message.timestamp)
        logger.debug("Updated conversation state with new topics.")

        # Add message to history
        self.messages.append(message)
        logger.debug(f"Message added to memory. Total messages: {len(self.messages)}")

        # Update topic weights with decay
        await self._update_topic_weights(message)
        logger.debug("Updated topic weights with decay.")

        # Maintain memory size
        if len(self.messages) > self.max_messages:
            removed_message = self.messages.pop(0)
            logger.debug(f"Memory exceeded. Removed oldest message: {removed_message.content}")

    async def _detect_topics(self, message: Message) -> Dict[str, float]:
        """Detect topics in message using embeddings and keyword matching"""
        logger.debug("Detecting topics in the message.")
        topics: Dict[str, float] = {}

        # Calculate embedding similarity with topic embeddings
        for topic, topic_embedding in self.topic_embeddings.items():
            similarity = cosine_similarity(
                [message.embedding],
                [topic_embedding]
            )[0][0]

            if similarity > 0.3:  # Base similarity threshold
                topics[topic] = similarity
                logger.debug(f"Topic '{topic}' detected with similarity {similarity}.")

        # Enhance with keyword matching
        for topic, definition in self.topic_definitions.items():
            content_lower = message.content.lower()
            keyword_matches = sum(
                keyword in content_lower
                for keyword in definition["keywords"]
            )
            if keyword_matches > 0:
                # Combine embedding and keyword signals
                base_score = topics.get(topic, 0)
                keyword_score = keyword_matches * 0.2 * definition["importance"]
                topics[topic] = min(1.0, base_score + keyword_score)
                logger.debug(f"Keyword matching enhanced topic '{topic}' to score {topics[topic]}.")

        return topics

    async def _update_topic_weights(self, message: Message):
        """Update topic weights using a dynamic approach"""
        logger.debug("Updating topic weights with dynamic decay.")
        # Calculate dynamic decay factor
        stability = self.conversation_state.get_topic_stability()
        velocity = self.conversation_state.get_conversation_velocity()
        dynamic_decay = self.base_decay_rate + (0.2 * stability) - (0.1 * velocity)
        logger.debug(f"Dynamic decay factor calculated: {dynamic_decay}")

        # Apply decay to existing topics
        for msg in self.messages[:-1]:  # Exclude current message
            for topic in msg.detected_topics:
                old_weight = msg.detected_topics[topic]
                msg.detected_topics[topic] *= dynamic_decay
                logger.debug(f"Decayed topic '{topic}' from {old_weight} to {msg.detected_topics[topic]}.")

        # Remove topics below threshold
        for msg in self.messages:
            before = len(msg.detected_topics)
            msg.detected_topics = {
                topic: weight
                for topic, weight in msg.detected_topics.items()
                if weight > self.min_topic_weight
            }
            after = len(msg.detected_topics)
            if before != after:
                logger.debug(f"Removed {before - after} topics from a message due to low weight.")

    def get_active_topics(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get currently active topics with weights"""
        logger.debug("Retrieving active topics.")
        topic_weights: Dict[str, float] = {}

        # Consider recent messages more heavily
        for i, msg in enumerate(self.messages[-5:]):
            recency_weight = 0.6 + (0.1 * i)  # More recent messages weight more
            for topic, weight in msg.detected_topics.items():
                current_weight = topic_weights.get(topic, 0)
                topic_weights[topic] = max(
                    current_weight,
                    weight * recency_weight
                )
                logger.debug(f"Applied recency weight to topic '{topic}': {weight * recency_weight}")

        # Get related topics based on hierarchy
        enhanced_weights = topic_weights.copy()
        for topic, weight in topic_weights.items():
            related = self.topic_hierarchy.get_related_topics(topic)
            for related_topic in related:
                if related_topic not in enhanced_weights:
                    enhanced_weights[related_topic] = weight * 0.3
                    logger.debug(f"Added related topic '{related_topic}' with weight {weight * 0.3}")

        # Filter and sort topics
        active_topics = [
            (topic, weight)
            for topic, weight in enhanced_weights.items()
            if weight > threshold
        ]
        active_topics_sorted = sorted(active_topics, key=lambda x: x[1], reverse=True)
        logger.debug(f"Active topics after filtering: {active_topics_sorted}")
        return active_topics_sorted

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of current conversation context"""
        logger.debug("Generating context summary.")
        active_topics = self.get_active_topics()
        recent_messages = self.messages[-5:]

        summary = {
            "active_topics": active_topics,
            "topic_stability": self.conversation_state.get_topic_stability(),
            "conversation_velocity": self.conversation_state.get_conversation_velocity(),
            "recent_topics": [
                msg.detected_topics
                for msg in recent_messages
            ],
            "context": {
                "last_code_reference": recent_messages[-1].metadata.get("code_context")
                if recent_messages else None,
                "last_analysis": recent_messages[-1].metadata.get("analysis")
                if recent_messages else None
            }
        }
        logger.debug(f"Context summary generated: {summary}")
        return summary

class ChatbotAgent(EnhancedFractalAgent):
    """Fractal agent for handling chat interactions with dynamic sub-agent creation"""

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        api_key: str,
        vector_store: Optional[LocalVectorStore] = None
    ):
        super().__init__(name, capabilities=[
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.PATTERN_MATCHING,
            AgentCapability.CODE_GENERATION,
            AgentCapability.DOCUMENTATION
        ])
        logger.debug(f"Initializing ChatbotAgent '{name}' with LLM provider '{llm_provider.value}'.")

        self.llm_provider = llm_provider
        self.api_key = api_key
        self.memory = ChatMemory()
        self.vector_store = vector_store or LocalVectorStore()
        logger.debug("ChatMemory and VectorStore initialized.")

        self.sub_agents: List[EnhancedFractalAgent] = []
        logger.debug("Sub-agent list initialized.")

        # Initialize base embedding model for task analysis
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("Base embedding model for task analysis initialized.")

    async def _analyze_task_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze if a task is simple or requires complex processing"""
        logger.debug(f"Analyzing task complexity for message: {message}")
        prompt = f"""Analyze this task and determine its type and needs:
        Message: {message}

        Determine:
        1. Is this asking about specific code/implementation?
        2. Does it need to reference code context?
        3. Is this a general question or specific to existing code?
        4. What type of code artifacts might be relevant (functions, classes, files)?

        Return a JSON object with:
        1. needs_code_context (boolean)
        2. code_artifacts_needed (array of strings, e.g., ["login", "authentication"])
        3. context_type (string: "specific_code", "implementation", "general", or "none")
        4. reasoning (string)
        """

        try:
            response = await self._generate_llm_response(
                prompt=prompt,
                system_message="You are a task analysis expert",
                response_format={"type": "json_object"}
            )
            analysis = json.loads(response)
            logger.debug(f"Task analysis result: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze task complexity: {e}")
            # Default to safe fallback
            fallback = {
                "needs_code_context": True,
                "code_artifacts_needed": [],
                "context_type": "specific_code",
                "reasoning": "Failed to analyze task, defaulting to code search"
            }
            logger.debug(f"Using fallback task analysis: {fallback}")
            return fallback

    def _format_code_context(self, search_results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """Format code search results into a readable context"""
        logger.debug("Formatting code context from search results.")
        code_context = {}

        for _, similarity, metadata in search_results:
            if 'code' in metadata:
                file_path = metadata.get('filePath', 'unknown')
                code_context[file_path] = {
                    'code': metadata['code'].strip(),
                    'similarity': similarity,
                    'language': metadata.get('language', 'unknown'),
                    'label': metadata.get('label', '')
                }
                logger.debug(f"Added code context for file '{file_path}' with similarity {similarity}.")

        return code_context

    def _create_code_specific_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Create a detailed prompt for code-specific questions"""
        logger.debug("Creating code-specific prompt for LLM.")
        parts = [f"User Question: {message}\n"]

        # Add code context if available
        if code_context := context.get("code_context"):
            parts.append("\nRelevant Code Context:")
            for file_path, content in code_context.items():
                parts.append(f"\nFile: {file_path}")
                parts.append(f"Relevance: {content['similarity']:.2f}")
                parts.append(f"Language: {content['language']}")
                if content['label']:
                    parts.append(f"Type: {content['label']}")
                parts.append(f"Code:\n```{content['language']}\n{content['code']}\n```")
            logger.debug("Added code context to prompt.")
        else:
            parts.append("\nNo relevant code context was found.")
            logger.debug("No code context available to add to prompt.")

        # Add conversation context if relevant
        if conv_context := context.get("conversation_context"):
            active_topics = conv_context.get("active_topics", [])
            if active_topics:
                parts.append("\nActive Conversation Topics:")
                for topic, weight in active_topics:
                    parts.append(f"- {topic} (relevance: {weight:.2f})")
                logger.debug("Added conversation context to prompt.")

        parts.append("\nPlease provide a comprehensive response that:")
        parts.append("1. Directly addresses the user's question")
        parts.append("2. References specific parts of the code when relevant")
        parts.append("3. Considers the current conversation context")
        parts.append("4. Provides explanations and examples where appropriate")

        prompt = "\n".join(parts)
        logger.debug(f"Final prompt for LLM: {prompt}")
        return prompt

    def _get_system_message(self) -> str:
        """Get the system message for LLM interactions"""
        system_message = """You are an intelligent code assistant that can:
        1. Analyze and explain code in any programming language
        2. Generate code solutions
        3. Provide technical documentation
        4. Suggest optimizations and improvements

        Format your responses clearly and reference specific parts of the context when relevant."""
        logger.debug("Generated system message for LLM.")
        return system_message

    async def _generate_llm_response(
        self,
        prompt: str,
        system_message: str,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate response using the configured LLM"""
        logger.debug(f"Generating LLM response with provider '{self.llm_provider.value}'.")
        if self.llm_provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(prompt, system_message, response_format)
        else:
            error_msg = f"Unsupported LLM provider: {self.llm_provider}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def _generate_openai_response(
        self,
        prompt: str,
        system_message: str,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate response using OpenAI"""
        from openai import OpenAI

        logger.debug("Preparing to generate response using OpenAI.")
        client = OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        try:
            logger.debug("Sending request to OpenAI API.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            llm_response = response.choices[0].message.content
            logger.debug(f"Received response from OpenAI: {llm_response}")
            return llm_response

        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return "I encountered an error while generating a response. Please try again."

    async def _process_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Process a task with proper task analysis and response generation"""
        logger.debug(f"Processing task ID: {task.id}")
        try:
            message = task.data.get("message", "")
            if not message:
                error_msg = "No message provided"
                logger.error(error_msg)
                return FractalResult(task.id, False, error=error_msg)

            # Add user message to memory
            user_message = Message(
                role="user",
                content=message,
                metadata={"task_type": task.type}
            )
            await self.memory.add_message(user_message)
            logger.debug("Added user message to memory.")

            # First, analyze task complexity and requirements
            task_analysis = await self._analyze_task_complexity(message)

            # Log task analysis details
            logger.debug(f"Task analysis: {task_analysis}")

            # Build response context
            response_context = {
                "task_analysis": task_analysis,
                "code_context": None,
                "conversation_context": self.memory.get_context_summary()
            }
            logger.debug("Built response context.")

            # Perform vector search to get code context
            search_query = " ".join(task_analysis.get("code_artifacts_needed", [])) or message
            logger.debug(f"Performing vector search with query: {search_query}")
            search_results = self.vector_store.simple_search_similar(
                text=search_query,
                k=3,
                threshold=0.1
            )
            logger.debug(f"Vector search results: {search_results}")
            if search_results:
                response_context["code_context"] = self._format_code_context(search_results)
                logger.debug("Code context added to response context.")

            # Generate response
            prompt = self._create_code_specific_prompt(message, response_context)

            # Generate response
            response = await self._generate_llm_response(
                prompt=prompt,
                system_message=self._get_system_message()
            )
            logger.debug("Generated response from LLM.")

            # Add response to memory
            assistant_message = Message(
                role="assistant",
                content=response,
                metadata={
                    "task_type": "response",
                    "context_used": response_context
                }
            )
            await self.memory.add_message(assistant_message)
            logger.debug("Added assistant response to memory.")

            return FractalResult(task.id, True, result={
                "response": response,
                "analysis": task_analysis,
                "context_used": response_context
            })

        except Exception as e:
            logger.error(f"Error in ChatbotAgent _process_task: {str(e)}")
            return FractalResult(task.id, False, error=str(e))



@dataclass
class CodeAnalysisResult:
    """Structure for code analysis results"""
    language: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    patterns: List[str]
    complexity_score: float
    suggestions: List[str]
    dependencies: Dict[str, List[str]]

class CodeAnalysisAgent(EnhancedFractalAgent):
    """Agent specialized in analyzing code structure and patterns"""

    def __init__(self, name: str, llm_provider: str, api_key: str):
        super().__init__(name, capabilities=[
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.PATTERN_MATCHING
        ])
        logger.debug(f"Initializing CodeAnalysisAgent '{name}' with LLM provider '{llm_provider}'.")

        self.llm_provider = llm_provider
        self.api_key = api_key
        self.vector_store = LocalVectorStore()
        logger.debug("CodeAnalysisAgent initialized with LocalVectorStore.")

    async def execute_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Execute a task with dynamic sub-agent creation"""
        logger.debug(f"CodeAnalysisAgent executing task ID: {task.id}")
        try:
            # First, analyze the task requirements
            requirements = await self.analyze_task_requirements(task)
            logger.debug(f"Task requirements: {requirements}")

            if requirements.get("is_simple", False):
                # For simple tasks, process directly without sub-agents
                logger.debug("Task is simple. Processing directly without sub-agents.")
                return await self._process_simple_task(task, context)

            # Create necessary sub-agents based on requirements
            await self._create_required_agents(requirements, context)
            logger.debug("Required sub-agents created based on task requirements.")

            # Process subtasks if any
            subtask_results = []
            for subtask in requirements.get("subtasks", []):
                capability = subtask.get("capability")
                logger.debug(f"Processing subtask with capability: {capability}")
                subtask_agent = await context.agent_pool.get_agent(capability)
                if subtask_agent:
                    logger.debug(f"Assigned sub-agent '{subtask_agent.name}' for capability '{capability}'.")
                    result = await subtask_agent.execute_task(
                        self._create_subtask(task, subtask),
                        context.create_child(subtask.get("description", ""))
                    )
                    subtask_results.append(result)
                else:
                    logger.warning(f"No available agent found for capability '{capability}'.")

            # Process main task with results from subtasks
            response = await self._process_main_task(task, subtask_results, context)
            logger.debug("Processed main task with subtask results.")

            # Learn from execution
            await self.learn_from_execution({
                "success": True,
                "patterns": list(requirements.get("capabilities_needed", {}).keys())
            })
            logger.debug("Learned from task execution.")

            return FractalResult(task.id, True, result=response)

        except Exception as e:
            logger.error(f"Error in CodeAnalysisAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _find_relevant_code(self, description: str) -> Optional[str]:
        """Find relevant code from vector store based on description"""
        logger.debug(f"Finding relevant code for description: {description}")
        results = self.vector_store.simple_search_similar(
            text=description,
            k=1,
            threshold=0.3
        )
        if results and results[0][2].get('code'):
            code = results[0][2]['code']
            logger.debug("Relevant code found in vector store.")
            return code
        logger.debug("No relevant code found in vector store.")
        return None

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        logger.debug("Detecting programming language from code.")
        # Simple language detection based on patterns
        patterns = {
            "python": r"(import\s+\w+|def\s+\w+\(|class\s+\w+:)",
            "javascript": r"(const|let|var|function\s+\w+|class\s+\w+\s*\{)",
            "java": r"(public\s+class|private\s+\w+|protected\s+\w+)",
            "typescript": r"(interface\s+\w+|type\s+\w+|:\s*\w+)",
        }

        for language, pattern in patterns.items():
            if re.search(pattern, code):
                logger.debug(f"Detected language: {language}")
                return language
        logger.debug("Language detection failed. Defaulting to 'unknown'.")
        return "unknown"

    async def _perform_static_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """Perform static code analysis"""
        logger.debug(f"Performing static analysis for language: {language}")
        if language == "python":
            return await self._analyze_python_code(code)
        # Add handlers for other languages
        return await self._analyze_generic_code(code)

    async def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code using AST"""
        logger.debug("Analyzing Python code using AST.")
        try:
            tree = ast.parse(code)
            analysis = {
                "imports": [],
                "classes": [],
                "functions": [],
                "dependencies": {}
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    analysis["imports"].extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    analysis["imports"].append(f"{node.module}.{node.names[0].name}")
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)

            logger.debug(f"Python code analysis result: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing Python code: {e}")
            return await self._analyze_generic_code(code)

    async def _analyze_generic_code(self, code: str) -> Dict[str, Any]:
        """Analyze code using LLM when language-specific analysis isn't available"""
        logger.debug("Analyzing code generically using LLM.")
        prompt = f"""Analyze this code and identify:
        1. Imports/dependencies
        2. Classes/components
        3. Functions/methods
        4. Dependencies between components

        Code:
        ```
        {code}
        ```

        Return the analysis in JSON format with these keys:
        imports, classes, functions, dependencies"""

        analysis = await self._generate_llm_response(prompt)
        try:
            analysis_json = json.loads(analysis)
            logger.debug(f"Generic code analysis result: {analysis_json}")
            return analysis_json
        except Exception as e:
            logger.error(f"Failed to parse generic code analysis: {e}")
            return {}

    async def _analyze_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code patterns and architecture"""
        logger.debug("Analyzing code patterns and architecture using LLM.")
        prompt = f"""Analyze this {language} code and identify:
        1. Design patterns used
        2. Architectural patterns
        3. Code complexity (scale 0-1)
        4. Notable coding practices

        Code:
        ```
        {code}
        ```

        Return analysis in JSON format with keys:
        patterns, complexity, practices"""

        result = await self._generate_llm_response(prompt)
        try:
            analysis = json.loads(result)
            logger.debug(f"Pattern analysis result: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Failed to parse pattern analysis: {e}")
            return {"patterns": [], "complexity": 0.5, "practices": []}

    async def _generate_suggestions(
        self,
        code: str,
        static_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on analyses"""
        logger.debug("Generating improvement suggestions based on analyses.")
        prompt = f"""Given this code analysis:
        Static Analysis: {json.dumps(static_analysis)}
        Pattern Analysis: {json.dumps(pattern_analysis)}

        Suggest improvements for:
        1. Code structure
        2. Pattern usage
        3. Performance
        4. Maintainability

        Return a JSON array of suggestion strings."""

        suggestions = await self._generate_llm_response(prompt)
        try:
            suggestions_list = json.loads(suggestions)
            logger.debug(f"Generated suggestions: {suggestions_list}")
            return suggestions_list
        except Exception as e:
            logger.error(f"Failed to parse suggestions: {e}")
            return []

class CodeGenerationAgent(EnhancedFractalAgent):
    """Agent specialized in generating code based on requirements"""

    def __init__(self, name: str, llm_provider: str, api_key: str):
        super().__init__(name, capabilities=[
            AgentCapability.CODE_GENERATION,
            AgentCapability.PATTERN_MATCHING
        ])
        logger.debug(f"Initializing CodeGenerationAgent '{name}' with LLM provider '{llm_provider}'.")

        self.llm_provider = llm_provider
        self.api_key = api_key
        self.vector_store = LocalVectorStore()
        logger.debug("CodeGenerationAgent initialized with LocalVectorStore.")

    async def execute_task(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Execute code generation task"""
        logger.debug(f"CodeGenerationAgent executing task ID: {task.id}")
        try:
            # Extract requirements
            requirements = task.data.get("description", "")
            language = task.data.get("language", "python")
            logger.debug(f"Task requirements: {requirements}, Language: {language}")

            # Find similar code examples
            similar_code = await self._find_similar_code(requirements)
            logger.debug(f"Found similar code examples: {similar_code}")

            # Generate code plan
            plan = await self._create_code_plan(requirements, similar_code, language)
            logger.debug(f"Generated code plan: {plan}")

            # Generate the code
            generated_code = await self._generate_code(plan, language)
            logger.debug(f"Generated code: {generated_code}")

            # Verify the generated code
            verification = await self._verify_code(generated_code, requirements, language)
            logger.debug(f"Verification result: {verification}")

            if not verification["is_valid"]:
                # Try one more time with verification feedback
                logger.debug("Verification failed. Regenerating code with feedback.")
                generated_code = await self._generate_code(
                    plan,
                    language,
                    verification["feedback"]
                )
                verification = await self._verify_code(generated_code, requirements, language)
                logger.debug(f"Re-verification result: {verification}")

            result = {
                "code": generated_code,
                "language": language,
                "plan": plan,
                "verification": verification,
                "similar_examples": similar_code
            }

            logger.debug("Code generation task completed successfully.")
            return FractalResult(task.id, True, result=result)

        except Exception as e:
            logger.error(f"Error in CodeGenerationAgent: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    async def _find_similar_code(self, requirements: str) -> List[Dict[str, Any]]:
        """Find similar code examples from vector store"""
        logger.debug(f"Finding similar code for requirements: {requirements}")
        results = self.vector_store.simple_search_similar(
            text=requirements,
            k=3,
            threshold=0.2
        )
        similar_code = [
            {
                "code": r[2].get('code', ''),
                "similarity": r[1],
                "source": r[2].get('filePath', 'unknown')
            }
            for r in results if r[2].get('code')
        ]
        logger.debug(f"Similar code found: {similar_code}")
        return similar_code

    async def _create_code_plan(
        self,
        requirements: str,
        similar_code: List[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """Create a plan for code generation"""
        logger.debug("Creating code generation plan.")
        similar_code_str = "\n".join(
            f"Example {i+1}:\n```\n{example['code']}\n```"
            for i, example in enumerate(similar_code)
        )

        prompt = f"""Create a detailed plan for generating {language} code that meets these requirements:
        {requirements}

        Similar code examples:
        {similar_code_str}

        Create a plan that includes:
        1. Components/classes needed
        2. Functions/methods needed
        3. Dependencies/imports required
        4. Implementation steps
        5. Design patterns to use

        Return the plan in JSON format."""

        plan = await self._generate_llm_response(prompt)
        try:
            plan_json = json.loads(plan)
            logger.debug(f"Generated code plan: {plan_json}")
            return plan_json
        except Exception as e:
            logger.error(f"Failed to parse code plan: {e}")
            return {}

    async def _generate_code(
        self,
        plan: Dict[str, Any],
        language: str,
        feedback: Optional[str] = None
    ) -> str:
        """Generate code based on the plan"""
        logger.debug("Generating code based on the plan.")
        prompt = f"""Generate {language} code based on this plan:
        {json.dumps(plan, indent=2)}

        {"Consider this feedback from previous attempt:" + feedback if feedback else ""}

        Requirements:
        1. Follow language best practices
        2. Include documentation
        3. Use proper error handling
        4. Follow the specified design patterns

        Return only the code, no explanations."""

        code = await self._generate_llm_response(prompt)
        logger.debug(f"Generated code: {code}")
        return code

    async def _verify_code(
        self,
        code: str,
        requirements: str,
        language: str
    ) -> Dict[str, Any]:
        """Verify generated code meets requirements"""
        logger.debug("Verifying generated code against requirements.")
        prompt = f"""Verify this {language} code meets the requirements:

        Requirements:
        {requirements}

        Code:
        ```
        {code}
        ```

        Check for:
        1. Syntax correctness
        2. Requirements fulfillment
        3. Best practices
        4. Potential issues

        Return JSON with:
        1. is_valid (boolean)
        2. feedback (string)
        3. suggestions (array)"""

        verification = await self._generate_llm_response(prompt)
        try:
            verification_json = json.loads(verification)
            logger.debug(f"Verification JSON: {verification_json}")
            return verification_json
        except Exception as e:
            logger.error(f"Failed to parse verification result: {e}")
            return {
                "is_valid": False,
                "feedback": "Error in verification process",
                "suggestions": []
            }

class AgentPool:
    """Manages a pool of sub-agents with lifecycle management"""
    def __init__(self):
        self.active_agents: Dict[str, EnhancedFractalAgent] = {}
        self.idle_agents: Dict[str, EnhancedFractalAgent] = {}
        logger.debug("AgentPool initialized with empty active and idle agent pools.")

    async def get_agent(self, capability: str) -> Optional[EnhancedFractalAgent]:
        logger.debug(f"Requesting agent with capability: {capability}")
        # First check idle agents
        if capability in self.idle_agents:
            agent = self.idle_agents.pop(capability)
            self.active_agents[capability] = agent
            logger.debug(f"Reusing idle agent '{agent.name}' for capability '{capability}'.")
            return agent

        # Then check active agents
        agent = self.active_agents.get(capability)
        if agent:
            logger.debug(f"Agent '{agent.name}' is already active for capability '{capability}'.")
        else:
            logger.debug(f"No active agent found for capability '{capability}'.")
        return agent

    async def release_agent(self, agent: EnhancedFractalAgent):
        """Move agent back to idle pool"""
        capability = agent.primary_capability
        if capability in self.active_agents:
            self.active_agents.pop(capability)
            self.idle_agents[capability] = agent
            logger.debug(f"Released agent '{agent.name}' back to idle pool for capability '{capability}'.")
        else:
            logger.warning(f"Attempted to release agent '{agent.name}' which is not in active agents.")

