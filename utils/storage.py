# fractal_framework/utils/storage.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

class ConversationStorage(ABC):
    """Abstract base class for conversation storage"""

    @abstractmethod
    async def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation and return its ID"""
        pass

    @abstractmethod
    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to a conversation"""
        pass

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        pass

    @abstractmethod
    async def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent conversations"""
        pass

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        pass

class LocalStorage(ConversationStorage):
    """Local file-based storage implementation"""

    def __init__(self, storage_dir: str = "conversation_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    async def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "messages": []
        }

        self._save_conversation(conversation_id, conversation)
        return conversation_id

    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return False

            message["timestamp"] = datetime.now().isoformat()
            conversation["messages"].append(message)

            self._save_conversation(conversation_id, conversation)
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            filepath = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if not os.path.exists(filepath):
                return None

            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return None

    async def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        conversations = []
        try:
            files = sorted(
                [f for f in os.listdir(self.storage_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x)),
                reverse=True
            )

            for file in files[:limit]:
                with open(os.path.join(self.storage_dir, file), 'r') as f:
                    conversations.append(json.load(f))

            return conversations
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        try:
            filepath = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False

    def _save_conversation(self, conversation_id: str, conversation: Dict[str, Any]):
        filepath = os.path.join(self.storage_dir, f"{conversation_id}.json")
        with open(filepath, 'w') as f:
            json.dump(conversation, f, indent=2)

class FirebaseStorage(ConversationStorage):
    """Firebase-based storage implementation using subcollections for messages"""

    def __init__(self, credentials_path: Optional[str] = None):
        if not firebase_admin._apps:
            if credentials_path:
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()

        self.db = firestore.client()
        self.conversations_ref = self.db.collection('conversations')

    async def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Create conversation document without messages array
            conversation = {
                "created_at": datetime.now(),
                "metadata": metadata or {},
                "message_count": 0  # Track number of messages
            }

            doc_ref = self.conversations_ref.document()
            doc_ref.set(conversation)
            return doc_ref.id
        except Exception as e:
            print(f"Error creating conversation: {e}")
            return ""

    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        try:
            # Get conversation document reference
            conv_ref = self.conversations_ref.document(conversation_id)

            # Add timestamp to message
            message["timestamp"] = datetime.now()

            # Add message to messages subcollection
            messages_ref = conv_ref.collection('messages')
            messages_ref.add(message)

            # Update message count in conversation document
            conv_ref.update({
                "message_count": firestore.Increment(1)
            })

            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            # Get conversation document
            conv_ref = self.conversations_ref.document(conversation_id)
            conv_doc = conv_ref.get()

            if not conv_doc.exists:
                return None

            # Get conversation data
            data = conv_doc.to_dict()
            data["id"] = conv_doc.id

            # Get messages from subcollection
            messages = []
            messages_ref = conv_ref.collection('messages')
            message_docs = messages_ref.order_by('timestamp').stream()

            for doc in message_docs:
                message_data = doc.to_dict()
                message_data["id"] = doc.id
                messages.append(message_data)

            data["messages"] = messages
            return data

        except Exception as e:
            print(f"Error getting conversation: {e}")
            return None

    async def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            # Get conversations ordered by creation time
            docs = self.conversations_ref.order_by(
                "created_at", direction=firestore.Query.DESCENDING
            ).limit(limit).stream()

            conversations = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id

                # Get last message from messages subcollection
                messages_ref = doc.reference.collection('messages')
                last_message = messages_ref.order_by(
                    'timestamp', direction=firestore.Query.DESCENDING
                ).limit(1).stream()

                # Add last message to conversation data if exists
                for msg in last_message:  # Will only run once due to limit(1)
                    data["last_message"] = msg.to_dict()

                conversations.append(data)

            return conversations
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        try:
            conv_ref = self.conversations_ref.document(conversation_id)

            # Delete all messages in subcollection first
            messages_ref = conv_ref.collection('messages')
            self._delete_collection(messages_ref)

            # Then delete the conversation document
            conv_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False

    def _delete_collection(self, collection_ref, batch_size: int = 100):
        """Helper method to delete a collection"""
        docs = collection_ref.limit(batch_size).stream()
        deleted = 0

        for doc in docs:
            doc.reference.delete()
            deleted += 1

        if deleted >= batch_size:
            # If we have deleted a full batch, there might be more
            self._delete_collection(collection_ref, batch_size)
