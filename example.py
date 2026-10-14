import os
import shutil
import time
from typing import Dict, Any

from src.buffered_checkpoint import MockCheckpointSaver, FileBufferStore, BufferedCheckpointSaver

# Define a simple agent state structure
AgentState = Dict[str, Any]

def simulate_agent_step(current_state: AgentState, step_num: int) -> AgentState:
    """Simulates an agent performing a step and modifying its state."""
    new_state = current_state.copy()
    new_state["steps_completed"] = step_num
    new_state["messages"].append(f"Agent processed step {step_num}")
    time.sleep(0.1) # Simulate some work
    return new_state

def cleanup_buffers(buffer_dir: str):
    if os.path.exists(buffer_dir):
        shutil.rmtree(buffer_dir)
        print(f"\n--- Cleaned up buffer directory: {buffer_dir} ---")

if __name__ == "__main__":
    BUFFER_DIR = "agent_buffers_example"
    cleanup_buffers(BUFFER_DIR)

    # 1. Setup the core components
    mock_saver = MockCheckpointSaver()
    file_buffer = FileBufferStore(buffer_dir=BUFFER_DIR)
    buffered_saver = BufferedCheckpointSaver(wrapped_saver=mock_saver, buffer_store=file_buffer)

    # Configuration for our agent run (mimics LangGraph's config)
    agent_config = {"configurable": {"thread_id": "my_agent_run_123"}}

    initial_state: AgentState = {
        "steps_completed": 0,
        "messages": ["Agent initialized."],
        "data": {"foo": "bar"}
    }

    print("--- Scenario 1: Agent run with cancellation before formal checkpoint ---")

    current_state = initial_state
    print(f"Initial state: {current_state}")

    # Simulate agent steps, buffering state frequently
    for i in range(1, 5):
        current_state = simulate_agent_step(current_state, i)
        buffered_saver.buffer_state(agent_config, current_state)
        print(f"Step {i}: Buffered state - {current_state['messages'][-1]}")
        if i == 3:
            print("\n--- Simulating a crash/cancellation after step 3 ---")
            break # Agent run stops abruptly

    # At this point, no formal checkpoint has been saved by `mock_saver`.
    # Let's try to get the last formal checkpoint:
    last_formal_checkpoint = buffered_saver.get_checkpoint(agent_config)
    print(f"\nAttempting to retrieve last formal checkpoint: {last_formal_checkpoint}")
    assert last_formal_checkpoint is None, "Should not have a formal checkpoint yet."

    # Now, let's recover from the buffer
    print("\n--- Recovering state from buffer after cancellation ---")
    recovered_state = buffered_saver.recover_buffered_state(agent_config)

    if recovered_state:
        print(f"Successfully recovered buffered state: {recovered_state}")
        # Resume agent from this state
        current_state = recovered_state
        print(f"Agent resuming from step: {current_state['steps_completed']}")
    else:
        print("No buffered state found. Starting from initial state.")
        current_state = initial_state

    # Continue the agent run from the recovered state and eventually checkpoint
    for i in range(current_state["steps_completed"] + 1, 7):
        current_state = simulate_agent_step(current_state, i)
        buffered_saver.buffer_state(agent_config, current_state)
        print(f"Step {i}: Buffered state - {current_state['messages'][-1]}")

    # Finally, a successful formal checkpoint
    print("\n--- Agent run completes, performing formal checkpoint ---")
    final_checkpoint = buffered_saver.put_checkpoint(agent_config, current_state)
    print(f"Formal checkpoint saved: {final_checkpoint}")

    # Verify the formal checkpoint can be retrieved
    retrieved_formal_checkpoint = buffered_saver.get_checkpoint(agent_config)
    print(f"Retrieved formal checkpoint: {retrieved_formal_checkpoint}")
    assert retrieved_formal_checkpoint == final_checkpoint

    # Verify the buffer for this config is now clear
    buffered_state_after_checkpoint = buffered_saver.recover_buffered_state(agent_config)
    print(f"Buffered state after formal checkpoint (should be None): {buffered_state_after_checkpoint}")
    assert buffered_state_after_checkpoint is None, "Buffer should be cleared after a successful formal checkpoint."

    print("\n--- Scenario 2: Agent run with no issues, formal checkpoint works as expected ---")
    agent_config_2 = {"configurable": {"thread_id": "my_agent_run_456"}}
    current_state_2 = initial_state.copy()
    current_state_2["messages"] = ["Agent 2 initialized."]

    for i in range(1, 4):
        current_state_2 = simulate_agent_step(current_state_2, i)
        buffered_saver.buffer_state(agent_config_2, current_state_2)
        print(f"Agent 2, Step {i}: Buffered state - {current_state_2['messages'][-1]}")

    print("\nAgent 2 completes, performing formal checkpoint.")
    final_checkpoint_2 = buffered_saver.put_checkpoint(agent_config_2, current_state_2)
    print(f"Agent 2 formal checkpoint saved: {final_checkpoint_2['steps_completed']} steps.")

    retrieved_final_checkpoint_2 = buffered_saver.get_checkpoint(agent_config_2)
    print(f"Agent 2 retrieved formal checkpoint: {retrieved_final_checkpoint_2['steps_completed']} steps.")
    assert retrieved_final_checkpoint_2 == final_checkpoint_2

    buffered_state_after_checkpoint_2 = buffered_saver.recover_buffered_state(agent_config_2)
    print(f"Agent 2 buffered state after formal checkpoint (should be None): {buffered_state_after_checkpoint_2}")
    assert buffered_state_after_checkpoint_2 is None

    cleanup_buffers(BUFFER_DIR)
