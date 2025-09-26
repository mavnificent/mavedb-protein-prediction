import torch
from transformers import AutoTokenizer, EsmModel
from pathlib import Path
import torch.nn as nn

weight_dir = Path(__file__).resolve().parent / Path('weights')

class ESM(nn.Module):
    def __init__(self, model_name: str="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = EsmModel.from_pretrained(model_name, 
                                              add_pooling_layer=False, 
                                              cache_dir=weight_dir)

        
    def _mean_pool(self, last_hidden_state, attention_mask):
        """
        Compute mean embeddings per sequence, ignoring CLS, SEP, and PAD tokens.
        
        Args:
            last_hidden_state: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len) with 1 for real tokens, 0 for PAD
        
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Copy mask to avoid modifying the original
        mask = attention_mask.clone()
        
        # Zero out CLS (first token) and SEP (last real token)
        seq_lengths = mask.sum(dim=1)           # sum of real tokens per sequence
        batch_indices = torch.arange(mask.size(0))
        mask[:, 0] = 0                          # CLS
        mask[batch_indices, seq_lengths - 1] = 0  # SEP
        
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)       # (batch_size, seq_len, 1)
        
        # Sum embeddings over valid positions
        sum_embs = (last_hidden_state * mask_expanded).sum(dim=1)
        
        # Count valid tokens per sequence
        counts = mask.sum(dim=1).unsqueeze(-1)
        
        # Avoid division by zero
        counts = counts.clamp(min=1)
        
        return sum_embs / counts
        
        # # Zero out CLS + SEP + PAD
        # mask = attention_mask.clone()
        # for protein, true_length in enumerate(attention_mask.sum(dim=1)):
        #     mask[protein, 0] = 0
        #     # SEP token is last token of the non-padded sequence for that protein
        #     mask[protein, true_length - 1] = 0  # SEP
            
        # # expand mask for broadcasting
        # mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size())
        # sum_embs = (last_hidden_state * mask_expanded).sum(dim=1)
        # counts = mask.sum(dim=1).unsqueeze(-1)
        # return sum_embs / counts
    
    def forward(self, seq: str | list[str], max_length: int | None=None) -> torch.Tensor:
        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            padding=True,
            truncation=bool(max_length),
            max_length=max_length
        )
        inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])