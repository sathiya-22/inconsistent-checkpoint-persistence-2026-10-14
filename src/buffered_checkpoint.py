import abc
import json
import os
import time
from typing import Any, Dict, Optional, Protocol, runtime_checkable

# For demonstration, we'll use a simple file-based buffer.
# In a real production system, this might be Redis, a message queue, or a more robust persistent store.

class BufferStore(Protocol):
    """
    Protocol for a generic buffer storage mechanism.
    Implementations should handle transient storage of agent state.
    """
    def write(self, key: str, data: Any) -> None:
        """Writes data to the buffer for a given key."""
        ...

    def read(self, key: str) -> Optional[Any]:
        """Reads data from the buffer for a given key."""
        ...

    def delete(self, key: str) -> None:
        """Deletes data associated with a key from the buffer."""
        ...

    def clear_all(self) -> None:
        """Clears all data from the buffer."""
        ...

class FileBufferStore(BufferStore):
    """
    A simple file-based buffer store for demonstration purposes.
    Each key corresponds to a file.
    """
    def __init__(self, buffer_dir: str = "agent_buffers"):
        self.buffer_dir = buffer_dir
        os.makedirs(self.buffer_dir, exist_ok=True)

    def _get_filepath(self, key: str) -> str:
        return os.path.join(self.buffer_dir, f"{key}.json")

    def write(self, key: str, data: Any) -> None:
        filepath = self._get_filepath(key)
        with open(filepath, "w") as f:
            json.dump(data, f)
        # print(f"DEBUG: Buffered state for {key} to {filepath}")

    def read(self, key: str) -> Optional[Any]:
        filepath = self._get_filepath(key)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            # print(f"DEBUG: Read buffered state for {key} from {filepath}")
            return data
        return None

    def delete(self, key: str) -> None:
        filepath = self._get_filepath(key)
        if os.path.exists(filepath):
            os.remove(filepath)
            # print(f"DEBUG: Deleted buffered state for {key} from {filepath}")

    def clear_all(self) -> None:
        for filename in os.listdir(self.buffer_dir):
            os.remove(os.path.join(self.buffer_dir, filename))
        # print(f"DEBUG: Cleared all buffers in {self.buffer_dir}")


@runtime_checkable
class CheckpointSaver(Protocol):
    """
    A simplified protocol for checkpoint savers, mimicking LangChain's BaseCheckpointSaver.
    We only define the methods relevant to this problem.
    """
    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieves the last checkpoint."""
        ...

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Persists a new checkpoint."""
        ...

class BufferedCheckpointSaver:
    """
    Wraps an existing CheckpointSaver to add an intermediate buffering layer.

    Any state passed to `buffer_state` will be immediately persisted to the
    `BufferStore`. When `put_checkpoint` is called, the buffered state is
    cleared, assuming the underlying saver has successfully persisted it.

    If an agent run is cancelled before `put_checkpoint` is called, the
    `recover_buffered_state` method can be used to retrieve the last known
    buffered state.
    """
    def __init__(self, wrapped_saver: CheckpointSaver, buffer_store: BufferStore):
        self.wrapped_saver = wrapped_saver
        self.buffer_store = buffer_store

    def _get_buffer_key(self, config: Dict[str, Any]) -> str:
        # A simple way to derive a unique key for the buffer from the config.
        # In real systems, this might involve a more robust hashing or ID generation.
        return f"buffer_{json.dumps(config, sort_keys=True)}"

    def get_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last formal checkpoint from the wrapped saver.
        """
        return self.wrapped_saver.get(config)

    def buffer_state(self, config: Dict[str, Any], state: Dict[str, Any]) -> None:
        """
        Persists the current agent state to the transient buffer immediately.
        This should be called frequently during agent execution.
        """
        key = self._get_buffer_key(config)
        self.buffer_store.write(key, state)

    def put_checkpoint(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persists the checkpoint using the wrapped saver and then clears the
        corresponding buffered state.
        """
        result = self.wrapped_saver.put(config, checkpoint)
        key = self._get_buffer_key(config)
        self.buffer_store.delete(key) # Clear buffer after successful checkpoint
        return result

    def recover_buffered_state(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recovers the last known state from the buffer for a given configuration.
        This is useful after a failure or cancellation when no formal checkpoint
        was committed.
        """
        key = self._get_buffer_key(config)
        return self.buffer_store.read(key)

# --- Mock CheckpointSaver for demonstration ---
class MockCheckpointSaver:
    """
    A mock checkpoint saver that simulates LangChain's BaseCheckpointSaver.
    It simply stores checkpoints in memory.
    """
    def __init__(self):
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    def _get_key(self, config: Dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True)

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._get_key(config)
        # print(f"DEBUG: MockSaver.get called for {key}")
        return self._checkpoints.get(key)

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        key = self._get_key(config)
        self._checkpoints[key] = checkpoint
        # print(f"DEBUG: MockSaver.put called for {key}, checkpoint saved.")
        return checkpoint # LangGraph's put returns the saved checkpoint
