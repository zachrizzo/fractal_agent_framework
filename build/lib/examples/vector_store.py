import asyncio
from core import (
    FractalFramework,
    FractalTask,
    TaskType
)
from agents import VectorSearchAgent

async def compare_patterns():
    framework = FractalFramework()
    vector_agent = VectorSearchAgent("similarity_searcher")
    framework.add_root_agent(vector_agent)

    # Different pattern implementations
    patterns = [
        ("singleton", '''
class DatabaseConnection:
    _instance = None

    def __init__(self):
        self.connected = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def connect(self):
        if not self.connected:
            self.connected = True
            return "Connected to database"
'''),
        ("factory", '''
class VehicleFactory:
    @staticmethod
    def create_vehicle(vehicle_type):
        if vehicle_type == "car":
            return Car()
        elif vehicle_type == "truck":
            return Truck()
        else:
            raise ValueError("Unknown vehicle type")

    def get_vehicle_info(self, vehicle_type):
        vehicle = self.create_vehicle(vehicle_type)
        return vehicle.get_info()
'''),
        ("repository", '''
class ProductRepository:
    def __init__(self):
        self.products = {}

    def add(self, product_id, product):
        self.products[product_id] = product

    def get(self, product_id):
        return self.products.get(product_id)

    def update(self, product_id, product):
        if product_id in self.products:
            self.products[product_id] = product

    def delete(self, product_id):
        if product_id in self.products:
            del self.products[product_id]
'''),
        ("service", '''
class OrderService:
    def __init__(self, order_repo, product_repo):
        self.order_repository = order_repo
        self.product_repository = product_repo

    def create_order(self, order_data):
        # Validate product availability
        for item in order_data['items']:
            product = self.product_repository.get(item['product_id'])
            if not product:
                raise ValueError(f"Product {item['product_id']} not found")

        # Create order
        order_id = generate_order_id()
        self.order_repository.add(order_id, order_data)
        return order_id
''')
    ]

    # Add patterns to framework
    for pattern_id, code in patterns:
        framework.add_node(
            pattern_id,
            code=code.strip(),
            type="python"
        )

    # Compare each pattern with others
    for pattern_id, _ in patterns:
        search_task = FractalTask(
            id=f"analyze_{pattern_id}",
            type=TaskType.ANALYZE,
            data={
                "operation": "search",
                "node_id": pattern_id,
                "k": 4  # Compare with all patterns
            }
        )

        # Execute search
        result = await framework.execute_task(search_task)

        # Print detailed analysis
        print(f"\nPattern Analysis for {pattern_id}:")
        if result.success:
            similar_nodes = result.result.get("similar_nodes", [])
            for node in similar_nodes:
                similarity = node['similarity']
                print(f"\nCompared with {node['node_id']}:")
                print(f"Similarity Score: {similarity:.2f}")

                # Detailed analysis based on similarity score
                if similarity > 0.7:
                    print("Analysis: Very high similarity - likely same pattern implementation")
                elif similarity > 0.5:
                    print("Analysis: High similarity - shares major structural elements")
                elif similarity > 0.3:
                    print("Analysis: Moderate similarity - shares some common patterns")
                else:
                    print("Analysis: Low similarity - different pattern implementation")

                # Pattern-specific insights
                if node['node_id'] == 'repository' and similarity > 0.3:
                    print("Common Repository Pattern traits: CRUD operations")
                elif node['node_id'] == 'singleton' and similarity > 0.3:
                    print("Common Singleton Pattern traits: Instance management")
                elif node['node_id'] == 'factory' and similarity > 0.3:
                    print("Common Factory Pattern traits: Object creation")
                elif node['node_id'] == 'service' and similarity > 0.3:
                    print("Common Service Pattern traits: Business logic orchestration")
        else:
            print("Analysis error:", result.error)

if __name__ == "__main__":
    asyncio.run(compare_patterns())
