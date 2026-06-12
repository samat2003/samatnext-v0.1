# Security Policy

## Evaluation Safety Warning

> [!WARNING]
> - **Execution of Generated Code:** The evaluation pipeline of this repository executes model-generated Python code dynamically.
> - **Code May Be Unsafe:** Machine learning models, particularly those undergoing training or adversarial evaluation, can generate code containing bugs, infinite loops, resource exhausts, or potentially harmful behavior.
> - **Subprocess Isolation Limits:** Evaluation scripts run using subprocess isolation with timeout and resource limits. This isolation is **not** a secure sandboxing boundary and does not prevent system-level impact, network access, or filesystem access.
> - **Container Recommended:** Users must run the evaluation pipeline inside a containerized environment (e.g., Docker) or an isolated virtual machine.
> - **Sensitive Systems:** Do not run evaluation of untrusted model weights or generated code on production, sensitive, or personal development machines.
