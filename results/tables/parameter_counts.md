# Model Parameter Count and Structural Specifications

| Specification / Parameter Type | SamatNext-v0.1 | Matched Transformer Baseline |
| :--- | :---: | :---: |
| **Total Parameters** | 356,082,440 | 356,082,432 |
| **Trainable Parameters** | 356,082,440 | 356,082,432 |
| **Embedding Parameters** | 116,686,848 | 116,686,848 |
| **LM Head Parameters** | 116,686,848 | 116,686,848 |
| **Attention Parameters** | 9,437,192 | 37,748,736 |
| **Mixer Parameters (Non-Attn)** | 47,185,928 | N/A (MHA only) |
| **MLP (FFN) Parameters** | 75,497,472 | 84,934,656 |
| **Normalization Parameters** | 25,344 | 25,344 |
| **Verifier Head Parameters** | 0 | N/A |
| **Embeddings Tied Status** | Untied | Untied |
| **Vocab Size** | 151,936 | 151,936 |
| **Tokenizer Length** | 151,665 | 151,665 |
| **Context Length** | 8,192 | 8,192 |
| **Hidden Size** | 768 | 768 |
| **Layer Count** | 16 | 16 |
| **FFN Intermediate Size** | 2,048 | 2,304 |
| **Number of Heads** | 12 | 12 |
| **KV Heads** | 4 | 12 |
| **Mixer Pattern / Type** | alternating | Multi-Head Attention |
| **RoPE Status** | Enabled | Enabled |
