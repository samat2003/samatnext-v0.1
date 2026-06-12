# Security Policy

## Evaluation Safety Warning

> [!WARNING]
> - **Execution of Generated Code:** The evaluation pipeline of this repository executes model-generated Python code dynamically.
> - **Code May Be Unsafe:** Machine learning models, particularly those undergoing training or adversarial evaluation, can generate code containing bugs, infinite loops, resource exhausts, or potentially harmful behavior.
> - **Subprocess Isolation Limits:** Evaluation executes model-generated Python code using subprocess isolation with timeout and resource limits. This is not a complete security boundary. Run evaluation inside a container or VM when evaluating untrusted models or code. This is not a secure sandbox.
> - **Container Recommended:** Users must run the evaluation pipeline inside a containerized environment (e.g., Docker) or an isolated virtual machine.
> - **Sensitive Systems:** Do not run evaluation of untrusted model weights or generated code on production, sensitive, or personal development machines.
