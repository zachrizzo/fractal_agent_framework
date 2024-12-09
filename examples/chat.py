from core.fractal_framework import FractalFramework
from core.fractal_task import FractalTask, TaskType
from core.fractal_context import FractalContext
from agents.chatbot_agent import ChatbotAgent, LLMProvider
from utils.vector_store import LocalVectorStore
import asyncio
from dotenv import load_dotenv
import os
import logging
import networkx as nx
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more verbose output
logger = logging.getLogger(__name__)

# Debug: Print available TaskType values
def debug_task_types():
    """Debug function to print available TaskType values"""
    try:
        logger.debug("Available TaskType values:")
        for task_type in TaskType:
            logger.debug(f"- {task_type.name}: {task_type.value}")
    except Exception as e:
        logger.error(f"Error inspecting TaskType: {e}")

async def initialize_framework():
    """Initialize the framework with necessary components"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    # Create framework components
    framework = FractalFramework()

    # Initialize vector store with API key
    vector_store = LocalVectorStore(api_key=api_key)

    # Create chat agent
    chat_agent = ChatbotAgent(
        name="ai_assistant",
        llm_provider=LLMProvider.OPENAI,
        api_key=api_key,
        vector_store=vector_store
    )

    # Add agent to framework
    framework.add_root_agent(chat_agent)

    return framework, vector_store, chat_agent

async def index_code_data(vector_store: LocalVectorStore, code_files: list):
    """Index code files into vector store"""
    print("Indexing code files...")

    for file_data in code_files:
        try:
            code_id = file_data['id']
            metadata = {
                'code': file_data['data']['code'],
                'filePath': file_data['data'].get('filePath', ''),
                'language': file_data['data'].get('language', 'unknown')
            }

            vector_store.add_embedding(code_id, file_data['data']['code'], metadata)
            print(f"Indexed: {code_id}")

        except Exception as e:
            print(f"Error indexing {file_data.get('id', 'unknown')}: {e}")

async def chat_session(framework: FractalFramework, chat_agent: ChatbotAgent):
    """Run interactive chat session"""
    print("\nAI Assistant Ready!")
    print("Ask questions, request code analysis, or get help with development tasks.")
    print("Type 'exit' to end the session.")
    print("-" * 50)

    # Debug: Print available task types
    debug_task_types()

    # Create initial context with empty graph
    context = FractalContext(graph=nx.DiGraph())

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break

            # Create and execute task using TaskType.ANALYZE for now
            task = FractalTask(
                id=f"chat_{abs(hash(user_input))}",
                type=TaskType.ANALYZE,  # Using ANALYZE as a fallback
                data={"message": user_input}
            )

            result = await chat_agent.execute_task(task, context)

            if result.success:
                response = result.result.get("response", "No response generated")
                print(result)
                print("\nAssistant:", response)
            else:
                error = result.error or "Unknown error occurred"
                print("\nError:", error)

        except Exception as e:
            logger.error(f"Error in chat session: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")

async def main():
    try:
        # Initialize framework
        framework, vector_store, chat_agent = await initialize_framework()

        # Example code data
        code_files = [
            {
                'id': 'example1',
                'data': {
                    'code': '''
                    def process_data(data):
                        """Process input data and return results"""
                        results = []
                        for item in data:
                            processed = item.strip().lower()
                            results.append(processed)
                        return results
                    ''',
                    'filePath': 'utils/processor.py',
                    'language': 'python'
                }
            },
            {
                "id": "example2",
                "data": {
                    "code": '''
                   const login = async (username, password) => {
                        try {
                            const response = await fetch('/api/login', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ username, password })
                            });
                            const data = await response.json();
                            return data;
                        } catch (error) {
                            console.error('Login error:', error);
                            return null;
                        }
                    };
                    ''',
                    "filePath": "services/login.js",
                    "language": "javascript"
                }
            }
        ]

        # Index code data
        await index_code_data(vector_store, code_files)

        # Start chat session with framework
        await chat_session(framework, chat_agent)

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
