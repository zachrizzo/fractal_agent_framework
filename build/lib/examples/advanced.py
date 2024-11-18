import asyncio
from patterns import PatternAnalyzer, PatternRegistry
import networkx as nx

async def test_pattern_detection():
    # Initialize components
    analyzer = PatternAnalyzer(PatternRegistry())

    # Test code with multiple clear patterns
    test_code = '''
from typing import Optional, Dict, Any, Generic, TypeVar
from abc import ABC, abstractmethod
import logging

# Type variable for generic repository
T = TypeVar('T')

# Base entity for type safety
class Entity:
    """Base class for all entities"""
    def __init__(self, entity_id: str):
        self.id = entity_id

class Repository(Generic[T], ABC):
    """
    Generic Repository Interface
    Provides a standard interface for data access operations
    """
    @abstractmethod
    def add(self, entity: T) -> bool:
        """Add an entity to the repository"""
        pass

    @abstractmethod
    def get(self, entity_id: str) -> Optional[T]:
        """Retrieve an entity by ID"""
        pass

    @abstractmethod
    def update(self, entity: T) -> bool:
        """Update an existing entity"""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity by ID"""
        pass

class User(Entity):
    """User entity with basic attributes"""
    def __init__(self, user_id: str, name: str, role: str):
        super().__init__(user_id)
        self.name = name
        self.role = role

class UserRepository(Repository[User]):
    """
    User Repository Implementation
    Handles user data storage and retrieval with improved encapsulation
    """
    def __init__(self):
        self._storage: Dict[str, User] = {}
        self._logger = logging.getLogger(__name__)

    def add(self, user: User) -> bool:
        """
        Add a user to the repository
        Returns True if successful, False if user already exists
        """
        try:
            if user.id in self._storage:
                self._logger.warning(f"User {user.id} already exists")
                return False
            self._storage[user.id] = user
            self._logger.info(f"User {user.id} added successfully")
            return True
        except Exception as e:
            self._logger.error(f"Error adding user: {str(e)}")
            return False

    def get(self, user_id: str) -> Optional[User]:
        """
        Retrieve a user by ID
        Returns None if user not found
        """
        try:
            user = self._storage.get(user_id)
            if user is None:
                self._logger.info(f"User {user_id} not found")
            return user
        except Exception as e:
            self._logger.error(f"Error retrieving user: {str(e)}")
            return None

    def update(self, user: User) -> bool:
        """
        Update an existing user
        Returns True if successful, False if user not found
        """
        try:
            if user.id not in self._storage:
                self._logger.warning(f"User {user.id} not found for update")
                return False
            self._storage[user.id] = user
            self._logger.info(f"User {user.id} updated successfully")
            return True
        except Exception as e:
            self._logger.error(f"Error updating user: {str(e)}")
            return False

    def delete(self, user_id: str) -> bool:
        """
        Delete a user by ID
        Returns True if successful, False if user not found
        """
        try:
            if user_id not in self._storage:
                self._logger.warning(f"User {user_id} not found for deletion")
                return False
            del self._storage[user_id]
            self._logger.info(f"User {user_id} deleted successfully")
            return True
        except Exception as e:
            self._logger.error(f"Error deleting user: {str(e)}")
            return False

class UserManager:
    """
    Singleton pattern implementation with dependency injection
    Manages user operations through the repository
    """
    _instance = None

    def __init__(self, repository: Repository[User]):
        self._repository = repository
        self._logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls, repository: Optional[Repository[User]] = None) -> 'UserManager':
        """Get or create singleton instance with optional repository"""
        if cls._instance is None:
            if repository is None:
                repository = UserRepository()
            cls._instance = cls(repository)
        return cls._instance

class UserFactory:
    """
    Factory pattern implementation with type safety
    Creates different types of users based on role
    """
    def create_user(self, user_id: str, name: str, role: str = 'regular') -> User:
        """Create a user with the specified role"""
        try:
            return User(user_id, name, role)
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}")
            raise ValueError(f"Invalid user parameters: {str(e)}")

# Example usage:
def main():
    # Create components
    repository = UserRepository()
    manager = UserManager.get_instance(repository)
    factory = UserFactory()

    # Create and store a user
    try:
        user = factory.create_user("1", "John Doe", "admin")
        success = repository.add(user)
        if success:
            print("User created successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
'''

    # Analyze patterns
    metrics = analyzer.analyze_code(test_code)

    print("\n=== Pattern Detection Results ===")

    # Show detected patterns
    print("\nDetected Patterns:")
    for pattern, count in metrics.pattern_count.items():
        print(f"\n{pattern} Pattern:")
        print(f"- Instances: {count}")
        print(f"- Complexity: {metrics.pattern_complexity[pattern]:.2f}")
        print(f"- Quality: {metrics.pattern_quality[pattern]:.2f}/5.0")

    # Show pattern relationships
    print("\nPattern Relationships:")
    for edge in metrics.pattern_relationships.edges(data=True):
        pattern1, pattern2, data = edge
        strength = data.get('weight', 0)
        print(f"- {pattern1} -> {pattern2} (strength: {strength:.2f})")

    # Pattern coupling analysis
    print("\nPattern Coupling Analysis:")
    density = nx.density(metrics.pattern_relationships)
    print(f"- Coupling Density: {density:.2f}")

    if density > 0.7:
        print("  High coupling - Consider reducing dependencies")
    elif density > 0.4:
        print("  Moderate coupling - Monitor for potential issues")
    else:
        print("  Good coupling level - Patterns are well-separated")

    # Get improvement suggestions
    suggestions = analyzer.get_pattern_suggestions(metrics)
    if suggestions:
        print("\nImprovement Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")

if __name__ == "__main__":
    asyncio.run(test_pattern_detection())
