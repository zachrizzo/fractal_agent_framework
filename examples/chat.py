# main.py
from core.fractal_framework import FractalFramework, FractalTask
from agents.chatbot_agent import ChatbotAgent, LLMProvider, TaskType
from core.fractal_context import FractalContext
from utils.vector_store import LocalVectorStore
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env')

async def initialize_framework(
    code_data: List[Dict[str, Any]],
    llm_config: Dict[str, str],
    dimension: int = 1536  # Updated default dimension for text-embedding-3-small
) -> FractalFramework:
    """Initialize the framework with code data"""
    # Create framework
    framework = FractalFramework()

    # Initialize vector store with specified dimension and API key
    vector_store = LocalVectorStore(
        dimension=dimension,
        api_key=llm_config['api_key']  # Pass API key to vector store
    )

    # Index code data
    print("Indexing code data...")
    for data in code_data:
        if 'data' in data and 'code' in data['data']:
            code = data['data']['code']
            if code:  # Only index non-empty code
                try:
                    # Add to vector store with metadata
                    metadata = {
                        'filePath': data['data'].get('filePath', ''),
                        'label': data['data'].get('label', ''),
                        'id': data['id'],
                        'code': code  # Include original code
                    }
                    vector_store.add_embedding(data['id'], code, metadata)
                    print(f"Indexed: {data['data'].get('label', 'unnamed')}")
                except Exception as e:
                    print(f"Error indexing code: {e}")

    # Initialize chatbot agent
    chatbot = ChatbotAgent(
        name="code_assistant",
        llm_provider=LLMProvider(llm_config['provider']),
        api_key=llm_config['api_key'],
        vector_store=vector_store
    )

    # Add to framework
    framework.add_root_agent(chatbot)
    print("Framework initialized successfully!")

    return framework

async def chat_session(framework: FractalFramework):
    """Run an interactive chat session"""
    context = FractalContext(framework.graph)
    chatbot = None

    # Find the chatbot agent
    for agent in framework.root_agents:
        if isinstance(agent, ChatbotAgent):
            chatbot = agent
            break

    if not chatbot:
        print("Error: ChatbotAgent not found in framework")
        return

    print("\nWelcome to the Code Assistant!")
    print("Type 'quit' to exit")
    print("--------------------------------")

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break

            # Create a task for the chatbot
            task = FractalTask(
                id="chat_task",
                type=TaskType.ANALYZE,
                data={"message": user_input}
            )

            # Execute task
            result = await chatbot.execute_task(task, context)

            if result.success:
                print("\nAssistant:", result.result['response'])
            else:
                print("\nError:", result.error)

        except Exception as e:
            print(f"\nAn error occurred: {e}")

async def main():
    # Example code data
    code_data = [
        {
            'id': 'file-_app.js',
            'data': {
                'code': '''const login = () => {
                    const [email, setEmail] = useState();
                    const [password, setPassword] = useState();
                    const handleSubmit = () => {
                        console.log(email, password);
                    };
                    return (
                        <div>
                            <input type="text" value={email} onChange={(e) => setEmail(e.target.value)} />
                            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                            <button onClick={handleSubmit}>Login</button>
                        </div>
                    );
                }''',
                'filePath': '/path/to/file',
                'label': '_app.js'
            }
        },
        {
            'id': 'file-_product.js',
            'data': {
                'code': '''class Product {
                    constructor(name, price) {
                        this.name = name;
                        this.price = price;
                    }
                    toString() {
                        return `${this.name} - $${this.price}`;
                    }
                }''',
                'filePath': '/path/to/file',
                'label': '_product.js'
            }
        }
    ]


    # Configure LLM with your API key
    llm_config = {
        'provider': 'openai',
        'api_key': os.getenv('OPENAI_API_KEY')
    }

    # Initialize framework
    framework = await initialize_framework(code_data, llm_config)

    # Start chat session
    await chat_session(framework)

if __name__ == "__main__":
    asyncio.run(main())
