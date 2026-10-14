## Problem: Inconsistent Checkpoint Persistence in Agent Frameworks

**The Real Problem:** Developers using agentic AI frameworks like LangGraph often rely on checkpoints to save the state of an agent's execution. This is crucial for recovery, debugging, and long-running processes. However, a common issue arises when an agent run is cancelled or fails unexpectedly: any state that has been streamed but not yet formally persisted as a checkpoint is lost. This leads to data loss, unreliable agent execution, and frustration for developers trying to build robust agent systems.

**Who it Affects:**
* Developers building stateful agents with frameworks like LangGraph, AutoGen, or similar.
* Anyone needing reliable state recovery and persistence for long-running or interactive agentic workflows.
* Teams deploying agents in production environments where resilience to failures is critical.

**Why this project shape/stack was chosen:**

This prototype is a lightweight Python library designed to act as an *interceptor* or *wrapper* around existing checkpointing mechanisms. Python is the dominant language in the agentic AI ecosystem, making a Python library the most accessible and immediately useful form.

The core idea is to introduce a "pre-persistence buffer" that continuously writes streamed state to a temporary, robust storage (like a local file or an in-memory database that can be flushed) *before* the framework's official checkpointing mechanism is invoked. If the official checkpoint fails or is cancelled, the state in this buffer can be recovered and used to reconstruct the last known good state. This provides an "eventually consistent" checkpointing layer, mitigating the loss of in-flight state.

We avoid directly modifying LangGraph's internals, instead providing a pattern that can be integrated with its `BaseCheckpointSaver` or similar interfaces in other frameworks.

**Setup and Usage Instructions (Zero API Keys Required):**

1.  **Clone the repository:**
    ```bash
    git clone <this-repo-url>
    cd <this-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the example:**
    The `example.py` demonstrates how to use the `BufferedCheckpointSaver` with a simulated agent run. It shows a scenario where a "cancellation" occurs, and then how the buffered state can be recovered.

    ```bash
    python example.py
    ```

    You will see output demonstrating:
    *   An agent performing steps and buffering state.
    *   A simulated cancellation event.
    *   The recovery of the last buffered state, even though no official "checkpoint" was committed.
    *   A successful checkpoint after recovery.

**No LLM Adapter Exists:**

This problem is purely about state persistence and recovery, an infrastructure concern. It does not involve processing or reasoning about LLM input/output directly. Therefore, no LLM adapter is included or necessary. The agent's "state" in this context is generic data (e.g., a dictionary, a list of messages) that the agent framework manages.
