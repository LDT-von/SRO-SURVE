"""Survival-only model with one query bank per modality."""

import torch
from torch.nn import functional as F
from torch import nn
from ..common_private_synergy.components.slots import masked_slot_read
from .duplicate_pair import duplicate_pair_loss
from ..common_private_synergy.scaled.omics_encoder import PathwayOmicsEncoder
from slotspe_proof.slot_diversity_losses import slot_read_overlap_loss


@torch.no_grad()
def read_trace(reads, messages, slots, mask):
    """Per-bag means over distinct slot pairs; kept only as validation aggregates."""
    i, j = torch.triu_indices(reads.shape[1], reads.shape[1], 1, device=reads.device)
    root = reads.sqrt()
    overlap = (root @ root.transpose(1, 2))[:, i, j].mean(1)
    result = {"read_BC": overlap, "real_tokens": mask.sum(1)}
    for name, values in [("message", messages), ("slot", slots)]:
        norm = values.norm(dim=-1)
        unit = F.normalize(values, dim=-1)
        cosine = (unit @ unit.transpose(1, 2))[:, i, j]
        valid = (norm[:, i] > 1e-12) & (norm[:, j] > 1e-12)
        result[name + "_cosine_sum"] = cosine.masked_fill(~valid, 0).sum(1)
        result[name + "_cosine_pairs"] = valid.sum(1)
    return result


class SlotEncoder(nn.Module):
    def __init__(self, input_dim, width, slots, iterations):
        super().__init__()
        self.iterations = iterations
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.1)
        self.project = nn.Linear(input_dim, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.update = nn.GRUCell(width, width)
        nn.init.zeros_(self.update.bias_ih)
        nn.init.zeros_(self.update.bias_hh)

    def forward(
        self,
        x,
        mask,
        *,
        read_rounds=(),
        penalties=None,
        pair_split=False,
        sro_margin=0.90,
        trace=None,
        axes=None,
    ):
        h = self.project(x.masked_fill(~mask[..., None], 0))
        k, v = self.key(h), self.value(h)
        slots = self.queries.expand(x.shape[0], -1, -1)
        for iteration in range(1, self.iterations + 1):
            logits = (
                self.query(self.norm(slots)) @ k.transpose(1, 2) / k.shape[-1] ** 0.5
            )
            attention, mass = masked_slot_read(logits, mask)
            if penalties is not None and iteration in read_rounds:
                if pair_split:
                    penalty = duplicate_pair_loss(logits, attention, x, mask, axes)
                else:
                    penalty, _ = slot_read_overlap_loss(
                        attention, mask, margin=sro_margin
                    )
                penalties.append(penalty)
            messages = attention @ v
            slots = self.update(messages.flatten(0, 1), slots.flatten(0, 1)).reshape_as(
                slots
            )
            if trace is not None and iteration in (1, 5, self.iterations):
                trace[iteration] = read_trace(
                    attention.detach(), messages.detach(), slots.detach(), mask
                )
        return slots, attention, mass / mask.sum(-1, keepdim=True)


class SROSurv(nn.Module):
    def __init__(self, p_dim, sizes, c):
        super().__init__()
        self.pair_split = c.get("read_loss") in ("duplicate_pair", "risk_pair")
        self.risk_pair = c.get("read_loss") == "risk_pair"
        self.read_rounds = tuple(c.get("read_loss_rounds", ()))
        selected = c.get("read_loss_modalities", ("p",)) if self.read_rounds else ()
        self.read_modalities = tuple(m for m in ("p", "g") if m in selected)
        self.sro_margin = c.get("sro_margin", 0.90)
        width = c["width"]
        self.omics_encoder = PathwayOmicsEncoder(
            sizes, width, dropout=c["omics_dropout"]
        )
        self.encoders = nn.ModuleDict(
            {
                m: SlotEncoder(d, width, c["slots_per_modality"], c["iterations"])
                for m, d in [("p", p_dim), ("g", width)]
            }
        )
        self.content = nn.ModuleDict(
            {m: nn.Linear(width, c["bins"]) for m in ("p", "g")}
        )
        self.cross = nn.Linear(c["bins"], c["bins"], bias=False)
        self.weights = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(c["slots_per_modality"])) for m in ("p", "g")}
        )

    def forward(self, p, g, p_mask, g_mask, *, penalties=None, trace=None, axes=None):
        if self.risk_pair and penalties is not None and axes is None:
            raise ValueError(
                "risk-pair training requires opposite-half training reference"
            )
        parts, terms = {}, {}
        for m, x, mask in [("p", p, p_mask), ("g", self.omics_encoder(g), g_mask)]:
            slots, attention, fraction = self.encoders[m](
                x,
                mask,
                read_rounds=self.read_rounds if m in self.read_modalities else (),
                penalties=penalties,
                pair_split=self.pair_split,
                sro_margin=self.sro_margin,
                axes=axes if m == "p" else None,
                trace=None if trace is None else trace.setdefault(m, {}),
            )
            parts[m] = (slots, attention)
            terms[m] = fraction.unsqueeze(-1) * (
                self.content[m](slots) + self.weights[m][None, :, None]
            )
        return self.predict(terms), parts, terms

    def predict(self, terms):
        p, g = terms["p"].sum(1), terms["g"].sum(1)
        return p + g + self.cross(p.tanh() * g.tanh())
