# Language Model Project: Bigram and GPT-style Implementation

This project explores two distinct language modeling techniques: a simple Bigram Language Model (LMM) and a more sophisticated GPT-style model. Both models are implemented from scratch using PyTorch, and the GPT model is trained on the **OpenWebText** dataset, following the principles of GPT-2.

## Models

### 1. Bigram Language Model

The **Bigram Language Model** is a simple and foundational approach to language modeling. It predicts the next word in a sequence by relying on the frequency of word pairs (bigrams) within a corpus.

#### Features:
- **Token Embedding**: Maps each token to a vector of size equal to the vocabulary size.
- **Next-word Prediction**: Predicts the next word based on the preceding one using logits derived from token embeddings.
- **Cross-Entropy Loss**: During training, the model computes the cross-entropy loss between the predicted and actual tokens.

#### Bigram Model Code Overview:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, index, targets=None):
        logits = self.token_embedding(index)
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat([index, index_next], dim=-1)
        return index
```

### 2. GPT-style Language Model

The **GPT-style Language Model** is a more advanced model based on the Transformer architecture. It uses **multi-head self-attention** to capture long-range dependencies in the text, allowing for more accurate next-word prediction.

#### Features:
- **Self-Attention Mechanism**: Each token in the sequence attends to all previous tokens using scaled dot-product attention.
- **Multi-Head Attention**: Captures different attention patterns by using multiple attention heads.
- **Feed-Forward Layers**: Non-linear transformations are applied after attention layers to learn complex patterns.
- **Layer Normalization**: Stabilizes the learning process by normalizing each layer's output.
- **Text Generation**: The model can generate new sequences by sampling from the probability distribution over the vocabulary.

#### GPT Model Code Overview:

```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding(index)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.ln_head(x)
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat([index, index_next], dim=-1)
        return index
```

## Dataset

The **OpenWebText** dataset is used for training the GPT-style model. OpenWebText is a publicly available dataset that closely mimics the original web scrape used to train GPT-2, making it ideal for large-scale language modeling tasks.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Jupyter Notebook (for running the code interactively)
- Required Python libraries: `torch`, `numpy`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/language-model.git
    cd language-model
    ```

2. Install required libraries:
    ```bash
    pip install torch numpy
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```

### Usage

You can run both the Bigram Language Model and the GPT-style model by loading the respective classes in your environment.

#### Bigram Language Model:

```python
bigram_model = BigramLanguageModel(vocab_size)
logits, loss = bigram_model(index, targets)
generated_tokens = bigram_model.generate(index, max_new_tokens=10)
```

#### GPT Language Model:

```python
gpt_model = GPTLanguageModel(vocab_size)
logits, loss = gpt_model(index, targets)
generated_text = gpt_model.generate(index, max_new_tokens=50)
```

## Future Improvements

- Fine-tune both models on domain-specific datasets.
- Add support for trigrams and higher-order n-grams for the Bigram model.
- Experiment with larger models and deeper Transformer blocks.
- Optimize training with techniques like mixed precision and distributed training.
