"""A small concept-conditioned GRU encoder-decoder for baby English."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD, BOS, EOS


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb, hidden, batch_first=True)

    def forward(self, src: torch.Tensor):
        x = self.drop(self.emb(src))
        out, hidden = self.gru(x)
        return out, hidden[-1]


class ConceptSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_concepts: int,
        emb: int = 128,
        hidden: int = 256,
        dropout: float = 0.2,
        use_concept: bool = True,
    ) -> None:
        super().__init__()
        self.use_concept = use_concept
        self.hidden = hidden
        self.encoder = Encoder(vocab_size, emb, hidden, dropout)
        self.drop = nn.Dropout(dropout)
        self.dec_emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.dec_gru = nn.GRUCell(emb, hidden)
        self.out = nn.Linear(2 * hidden, vocab_size)
        self.gen_gate = nn.Linear(2 * hidden + emb, 1)

        if use_concept:
            self.concept_emb = nn.Embedding(num_concepts, hidden)
            self.concept_head = nn.Linear(hidden, num_concepts)
        else:
            self.concept_emb = None
            self.concept_head = None

    def encode(self, src: torch.Tensor):
        enc_out, hidden = self.encoder(src)
        concept_logits = self.concept_head(hidden) if self.use_concept else None
        return enc_out, hidden, concept_logits

    def forward(
        self,
        src: torch.Tensor,
        dec_in: torch.Tensor,
        concepts: torch.Tensor | None = None,
    ):
        enc_out, hidden, concept_logits = self.encode(src)
        if self.use_concept:
            if concepts is None:
                concepts = concept_logits.argmax(dim=-1)
            hidden = hidden + self.concept_emb(concepts)

        src_mask = src.ne(0)
        dec_h = hidden
        outputs = []
        for step in range(dec_in.shape[1]):
            prev = dec_in[:, step]
            emb_prev = self.drop(self.dec_emb(prev))
            dec_h = self.dec_gru(emb_prev, dec_h)

            scores = torch.bmm(enc_out, dec_h.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~src_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)

            gen_probs = torch.softmax(self.out(torch.cat((dec_h, context), dim=-1)), dim=-1)
            copy_probs = torch.zeros_like(gen_probs).scatter_add_(1, src, weights)
            p_gen = torch.sigmoid(
                self.gen_gate(torch.cat((dec_h, context, emb_prev), dim=-1))
            )
            probs = p_gen * gen_probs + (1.0 - p_gen) * copy_probs
            outputs.append(torch.log(probs.clamp_min(1e-9)))

        return torch.stack(outputs, dim=1), concept_logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        concept_override: int | torch.Tensor | None = None,
    ) -> tuple[list[list[int]], torch.Tensor | None]:
        enc_out, hidden, concept_logits = self.encode(src)
        if self.use_concept:
            predicted = concept_logits.argmax(dim=-1)
            if concept_override is None:
                concepts = predicted
            elif isinstance(concept_override, torch.Tensor):
                concepts = concept_override.to(device=src.device, dtype=torch.long)
            else:
                concepts = torch.full_like(predicted, int(concept_override))
            hidden = hidden + self.concept_emb(concepts)
        else:
            predicted = None
            concepts = None

        src_mask = src.ne(0)
        batch = src.shape[0]
        prev = torch.full((batch,), 2, dtype=torch.long, device=src.device)  # BOS
        dec_h = hidden
        finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
        output = [[] for _ in range(batch)]

        for _ in range(max_len):
            emb_prev = self.drop(self.dec_emb(prev))
            dec_h = self.dec_gru(emb_prev, dec_h)
            scores = torch.bmm(enc_out, dec_h.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~src_mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
            gen_probs = torch.softmax(self.out(torch.cat((dec_h, context), dim=-1)), dim=-1)
            copy_probs = torch.zeros_like(gen_probs).scatter_add_(1, src, weights)
            p_gen = torch.sigmoid(
                self.gen_gate(torch.cat((dec_h, context, emb_prev), dim=-1))
            )
            probs = p_gen * gen_probs + (1.0 - p_gen) * copy_probs
            next_token = probs.argmax(dim=-1)
            for row in range(batch):
                if not finished[row]:
                    token = int(next_token[row].item())
                    if token == 3:  # EOS
                        finished[row] = True
                    else:
                        output[row].append(token)
            prev = next_token
            if bool(finished.all()):
                break
        return output, predicted


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
