# fractal_framework/examples/chat_with_storage.py

from core.fractal_framework import FractalFramework
from core.fractal_task import FractalTask, TaskType
from core.fractal_context import FractalContext
from agents.chatbot_agent import ChatbotAgent, LLMProvider
from utils.vector_store import LocalVectorStore
from utils.storage import LocalStorage, FirebaseStorage
import asyncio
from dotenv import load_dotenv
import os
import logging
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example code data for indexing
EXAMPLE_CODE_FILES = [
    {
        'id': 'data_processor',
        'data': {
            'code': r'''def process_data(data, options=None):
    """
    Process input data with optional configuration.

    Args:
        data: Input data to process
        options: Optional processing configuration

    Returns:
        Processed data results
    """
    results = []
    options = options or {}

    for item in data:
        # Apply transformations
        processed = item.strip().lower()

        # Apply optional filters
        if options.get('filter_empty') and not processed:
            continue

        # Apply optional transformations
        if options.get('uppercase'):
            processed = processed.upper()

        results.append(processed)

    return results''',
            'filePath': 'utils/data_processor.py',
            'language': 'python'
        }
    },
    {
        'id': 'auth_service',
        'data': {
            'code': r'''class AuthenticationService:
    def __init__(self, db_connection):
        self.db = db_connection
        self.active_sessions = {}

    async def login(self, username: str, password: str):
        """Authenticate user and create session"""
        try:
            # Verify credentials
            user = await self.db.users.find_one({'username': username})
            if not user or not self._verify_password(password, user['password_hash']):
                return {'success': False, 'error': 'Invalid credentials'}

            # Create session
            session_id = self._generate_session_id()
            self.active_sessions[session_id] = {
                'user_id': user['_id'],
                'created_at': datetime.now()
            }

            return {
                'success': True,
                'session_id': session_id,
                'user': {
                    'id': user['_id'],
                    'username': user['username'],
                    'role': user['role']
                }
            }
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return {'success': False, 'error': 'Authentication failed'}

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against stored hash"""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return str(uuid.uuid4())''',
            'filePath': 'services/auth_service.py',
            'language': 'python'
        }
    },
    {
        'id': 'data_validator',
        'data': {
            'code': r'''const validateUserData = (userData) => {
    const errors = {};

    // Validate username
    if (!userData.username || userData.username.length < 3) {
        errors.username = 'Username must be at least 3 characters long';
    }

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!userData.email || !emailRegex.test(userData.email)) {
        errors.email = 'Please provide a valid email address';
    }

    // Validate password
    if (!userData.password || userData.password.length < 8) {
        errors.password = 'Password must be at least 8 characters long';
    }

    // Check password strength
    const hasUpperCase = /[A-Z]/.test(userData.password);
    const hasLowerCase = /[a-z]/.test(userData.password);
    const hasNumbers = /\d/.test(userData.password);
    const hasSpecialChar = /[!@#$%^&*]/.test(userData.password);

    if (!(hasUpperCase && hasLowerCase && hasNumbers && hasSpecialChar)) {
        errors.password = 'Password must include uppercase, lowercase, numbers, and special characters';
    }

    return {
        isValid: Object.keys(errors).length === 0,
        errors
    };
};''',
            'filePath': 'utils/validators.js',
            'language': 'javascript'
        }
    }
]

def create_storage(storage_type: str) -> tuple[str, LocalStorage | FirebaseStorage]:
    """Create storage instance based on type"""
    if storage_type == "firebase":
        firebase_creds = os.getenv('FIREBASE_CREDENTIALS_PATH')
        if not firebase_creds:
            logger.warning("Firebase credentials not found, falling back to local storage")
            return "local", LocalStorage()
        try:
            storage = FirebaseStorage(credentials_path=firebase_creds)
            logger.info("Using Firebase storage")
            return "firebase", storage
        except Exception as e:
            logger.error(f"Error initializing Firebase storage: {e}")
            logger.warning("Falling back to local storage")
            return "local", LocalStorage()
    else:
        logger.info("Using local storage")
        return "local", LocalStorage()

async def index_code_data(vector_store: LocalVectorStore):
    """Index example code files into vector store"""
    print("\nIndexing example code files...")

    for file_data in EXAMPLE_CODE_FILES:
        try:
            code_id = file_data['id']
            metadata = {
                'code': file_data['data']['code'],
                'filePath': file_data['data'].get('filePath', ''),
                'language': file_data['data'].get('language', 'unknown')
            }

            vector_store.add_embedding(code_id, file_data['data']['code'], metadata)
            print(f"Indexed: {code_id} ({metadata['filePath']})")

        except Exception as e:
            print(f"Error indexing {file_data.get('id', 'unknown')}: {e}")

    print("Code indexing complete!\n")

async def initialize_framework(storage_type: str = "local"):
    """Initialize the framework with specified storage type"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    # Initialize storage
    storage_type, storage = create_storage(storage_type)

    # Create framework components
    framework = FractalFramework()
    vector_store = LocalVectorStore(api_key=api_key)

    # Index example code
    await index_code_data(vector_store)

    # Create chat agent with storage
    chat_agent = ChatbotAgent(
        name="ai_assistant",
        llm_provider=LLMProvider.OPENAI,
        api_key=api_key,
        vector_store=vector_store,
        storage=storage
    )

    # Add agent to framework
    framework.add_root_agent(chat_agent)

    return framework, vector_store, chat_agent, storage_type

async def switch_storage(chat_agent: ChatbotAgent, new_type: str) -> tuple[str, str]:
    """Switch storage type and return new type and status message"""
    try:
        new_type = new_type.lower()
        if new_type not in ["local", "firebase"]:
            return chat_agent.memory.storage.__class__.__name__.lower(), "Invalid storage type. Use 'local' or 'firebase'"

        current_type = chat_agent.memory.storage.__class__.__name__.lower()
        if "firebase" in current_type and new_type == "firebase" or \
           "local" in current_type and new_type == "local":
            return current_type, f"Already using {new_type} storage"

        # Create new storage
        new_type, new_storage = create_storage(new_type)

        # Update chat agent's storage
        chat_agent.memory.storage = new_storage

        return new_type, f"Switched to {new_type} storage"
    except Exception as e:
        logger.error(f"Error switching storage: {e}")
        return chat_agent.memory.storage.__class__.__name__.lower(), f"Error switching storage: {str(e)}"

async def chat_session(framework: FractalFramework, chat_agent: ChatbotAgent, current_storage_type: str):
    """Run interactive chat session with storage commands"""
    print("\nAI Assistant Ready!")
    print("\nAvailable commands:")
    print("  /new - Start a new conversation")
    print("  /load <id> - Load a previous conversation")
    print("  /list - List recent conversations")
    print("  /switch <type> - Switch storage type (local/firebase)")
    print("  /status - Show current storage status")
    print("\nExample questions you can ask:")
    print("  - What does the process_data function do?")
    print("  - How does the authentication service work?")
    print("  - Explain the password validation in the JavaScript code")
    print("\nType 'exit' to end the session.")
    print("-" * 50)

    # Create initial context with empty graph
    context = FractalContext(graph=nx.DiGraph())

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break

            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == '/switch' and len(parts) > 1:
                    current_storage_type, message = await switch_storage(chat_agent, parts[1])
                    print(f"\nSystem: {message}")
                    continue
                elif cmd == '/status':
                    print(f"\nSystem: Currently using {current_storage_type} storage")
                    continue

            # Create and execute task
            task = FractalTask(
                id=f"chat_{abs(hash(user_input))}",
                type=TaskType.UNDERSTAND,
                data={"message": user_input}
            )

            result = await chat_agent.execute_task(task, context)

            if result.success:
                response = result.result.get("content", "No response generated")
                print("\nAssistant:", response)
            else:
                error = result.error or "Unknown error occurred"
                print("\nError:", error)

        except Exception as e:
            logger.error(f"Error in chat session: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")

async def main():
    try:
        # Get storage type from environment or use default
        initial_storage_type = os.getenv('STORAGE_TYPE', 'local')

        # Initialize framework with specified storage
        framework, vector_store, chat_agent, current_storage_type = await initialize_framework(initial_storage_type)

        print(f"\nInitialized framework with {current_storage_type} storage")
        print("\nStorage configuration:")
        print("1. Local storage: Set STORAGE_TYPE=local in .env (or leave unset)")
        print("2. Firebase storage: ")
        print("   - Set STORAGE_TYPE=firebase in .env")
        print("   - Set FIREBASE_CREDENTIALS_PATH=/path/to/credentials.json in .env")
        print("\nYou can switch storage types during runtime using the /switch command")

        # Start chat session
        await chat_session(framework, chat_agent, current_storage_type)

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
