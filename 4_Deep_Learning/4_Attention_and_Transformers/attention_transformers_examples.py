import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention using PyTorch.

    Args:
        Q (torch.Tensor): Query tensor.
        K (torch.Tensor): Key tensor.
        V (torch.Tensor): Value tensor.
        mask (torch.Tensor, optional): Mask to hide certain connections. Defaults to None.

    Returns:
        tuple: Output tensor and attention weights.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

if __name__ == "__main__":
    # Example Usage:
    # Assume a batch size of 1, sequence length of 3, and embedding dimension of 4
    batch_size = 1
    seq_len = 3
    d_model = 4

    # Simulate Query, Key, Value tensors
    # In a real Transformer, Q, K, V would be derived from the same input embedding
    # or from encoder output (for cross-attention)
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    print("Query (Q):\n", Q)
    print("\nKey (K):\n", K)
    print("\nValue (V):\n", V)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("\nOutput of Attention:\n", output)
    print("\nAttention Weights:\n", attention_weights)

    # Example with a simple mask (e.g., for decoder self-attention to prevent looking ahead)
    # For a sequence of length 3, a causal mask would be:
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    # We use 0 for masked positions and 1 for unmasked positions
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    causal_mask = causal_mask.unsqueeze(0) # Add batch dimension

    output_masked, attention_weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print("\nOutput of Masked Attention:\n", output_masked)
    print("\nMasked Attention Weights:\n", attention_weights_masked)