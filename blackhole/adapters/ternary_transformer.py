"""Decoder-only ternary Transformer used by fresh STATIC foundation models."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .static_0t import TernaryLinear, model_parameters, ternary_parameters

VARIANT = "STATIC-FOUNDATION-TERNARY-v1"
TOKENIZER_NAME = "STATIC-FOUNDATION-BPE-v1"
_SUPPORTED_RECIPES = {
    "causal-pretrain-v1",
    "reasoning-mid-v1",
    "software-engineering-sft-v1",
    "instruction-sft-v1",
    "business-sft-v1",
    "relationship-sft-v1",
    "browser-agent-sft-v1",
    "executive-persona-sft-v1",
    "judgment-sft-v1",
    "preference-dpo-v1",
}


@dataclass(frozen=True)
class FoundationConfig:
    vocab_size: int = 32_000
    width: int = 768
    layers: int = 16
    heads: int = 12
    ff_hidden: int = 2304
    context: int = 2048
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        return self.width // self.heads


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float()
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized.to(dtype) * self.weight.to(dtype))


def _apply_rope(value: torch.Tensor, *, theta: float) -> torch.Tensor:
    # value: [batch, heads, time, head_dim]
    dim = value.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    time = value.shape[-2]
    positions = torch.arange(time, device=value.device, dtype=torch.float32)
    frequency = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=value.device, dtype=torch.float32) / dim)
    )
    angles = torch.outer(positions, frequency)
    cos = angles.cos()[None, None, :, :].to(value.dtype)
    sin = angles.sin()[None, None, :, :].to(value.dtype)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class FoundationBlock(nn.Module):
    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.width)
        self.qkv = TernaryLinear(config.width, config.width * 3, bias=False)
        self.attn_out = TernaryLinear(config.width, config.width, bias=False)
        self.ffn_norm = RMSNorm(config.width)
        self.gate_up = TernaryLinear(config.width, config.ff_hidden * 2, bias=False)
        self.down = TernaryLinear(config.ff_hidden, config.width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, width = value.shape
        normalized = self.attn_norm(value)
        qkv = self.qkv(normalized)
        query, key, val = qkv.chunk(3, dim=-1)
        heads = self.config.heads
        head_dim = self.config.head_dim

        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, time, heads, head_dim).transpose(1, 2)

        query = _apply_rope(reshape(query), theta=self.config.rope_theta)
        key = _apply_rope(reshape(key), theta=self.config.rope_theta)
        val = reshape(val)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            is_causal=True,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        value = value + self.attn_out(attended)

        normalized = self.ffn_norm(value)
        gate, up = self.gate_up(normalized).chunk(2, dim=-1)
        value = value + self.down(F.silu(gate) * up)
        return value


class FoundationLM(nn.Module):
    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(FoundationBlock(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use language-model initialization rather than PyTorch's unit-variance embedding default."""
        base_std = 0.02
        residual_std = base_std / math.sqrt(2.0 * self.cfg.layers)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=base_std)
        for block in self.blocks:
            nn.init.normal_(block.qkv.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.gate_up.weight, mean=0.0, std=base_std)
            nn.init.normal_(block.attn_out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.down.weight, mean=0.0, std=residual_std)
            nn.init.ones_(block.attn_norm.weight)
            nn.init.ones_(block.ffn_norm.weight)
        nn.init.ones_(self.norm.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.cfg.context:
            raise ValueError(f"expected [batch, time], max context {self.cfg.context}")
        value = self.embedding(tokens)
        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        value = self.norm(value)
        # Tied token embedding/output matrix.
        return F.linear(value, self.embedding.weight)


class StaticFoundationAdapter:
    """Train a fresh decoder Transformer from random weights on chosen curriculum stages."""

    def __init__(
        self,
        *,
        name: str,
        device: str,
        config: FoundationConfig | None = None,
        seed: int = 1234,
        learning_rate: float = 3e-4,
    ) -> None:
        if name not in {"STATIC-X", "STATIC-T", "STATIC-K"}:
            raise ValueError("unknown STATIC foundation model")
        self.name = name
        self.device = device
        self.config = config or FoundationConfig()
        self.seed = seed
        self.learning_rate = float(learning_rate)
        self.weight_decay = 0.1
        self.warmup_steps = 100
        self.train_steps = 1000
        self.min_lr_ratio = 0.1
        self.dpo_beta = 0.1
        self.training_recipe_version = "causal-pretrain-v1"
        self.reference_model = None
        self._tokenizer = None
        self._reset_training_state()

    @property
    def tokenizer_path(self) -> Path:
        return Path(__file__).parent / "assets" / "static_foundation_tokenizer_v1.model"

    def _get_tokenizer(self):
        if self._tokenizer is None:
            if not self.tokenizer_path.is_file():
                raise RuntimeError(
                    f"foundation tokenizer missing: {self.tokenizer_path}"
                )
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise RuntimeError(
                    "sentencepiece runtime dependency is required for STATIC foundation models"
                ) from exc
            tokenizer = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))
            if tokenizer.get_piece_size() != self.config.vocab_size:
                raise RuntimeError(
                    f"tokenizer/model vocabulary mismatch: {tokenizer.get_piece_size()} != {self.config.vocab_size}"
                )
            self._tokenizer = tokenizer
        return self._tokenizer

    def _build_scheduler(self):
        warmup = min(max(int(self.warmup_steps), 0), max(self.train_steps - 1, 0))
        total = max(int(self.train_steps), 1)
        minimum = float(self.min_lr_ratio)

        def factor(step: int) -> float:
            if warmup and step < warmup:
                return max((step + 1) / warmup, 1.0 / warmup)
            if total <= warmup + 1:
                return 1.0
            progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum + (1.0 - minimum) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=factor)

    def _new_optimizer(self):
        kwargs = {
            "lr": self.learning_rate,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": self.weight_decay,
        }
        if self.device.startswith("cuda"):
            try:
                return torch.optim.AdamW(self.model.parameters(), fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(self.model.parameters(), **kwargs)

    def _reset_training_state(self) -> None:
        torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("BLOCKED_CUDA: PyTorch CUDA is unavailable")
            torch.cuda.manual_seed_all(self.seed)
        self.model = FoundationLM(self.config).to(self.device)
        self.reference_model = None
        self.optimizer = self._new_optimizer()
        self.scheduler = self._build_scheduler()
        self._data_generator = torch.Generator(device="cpu")
        self._data_generator.manual_seed(self.seed)

    def configure_for_training(self, config: Mapping[str, Any]) -> None:
        context_length = config.get("context_length", self.config.context)
        if context_length != self.config.context:
            raise ValueError(
                f"context_length must be exactly {self.config.context} for {VARIANT}"
            )
        seed = config.get("seed", self.seed)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        learning_rate = config.get("learning_rate", self.learning_rate)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")
        recipe = str(config.get("training_recipe_version") or "")
        if recipe not in _SUPPORTED_RECIPES:
            raise ValueError(
                "training_recipe_version must be one of "
                + ", ".join(sorted(_SUPPORTED_RECIPES))
            )
        weight_decay = config.get("weight_decay", 0.1)
        warmup_steps = config.get("warmup_steps", 100)
        min_lr_ratio = config.get("min_lr_ratio", 0.1)
        dpo_beta = config.get("dpo_beta", 0.1)
        train_steps = config.get("train_steps")
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or not 0 <= float(weight_decay) <= 1
        ):
            raise ValueError("weight_decay must be between 0 and 1")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if (
            not isinstance(min_lr_ratio, (int, float))
            or isinstance(min_lr_ratio, bool)
            or not 0 <= float(min_lr_ratio) <= 1
        ):
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if (
            not isinstance(dpo_beta, (int, float))
            or isinstance(dpo_beta, bool)
            or not math.isfinite(float(dpo_beta))
            or not 0 < float(dpo_beta) <= 2
        ):
            raise ValueError("dpo_beta must be a positive finite number no greater than 2")
        if not isinstance(train_steps, int) or isinstance(train_steps, bool) or train_steps < 1:
            raise ValueError("train_steps must be a positive integer")

        self.seed = seed
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.min_lr_ratio = float(min_lr_ratio)
        self.dpo_beta = float(dpo_beta)
        self.training_recipe_version = recipe
        self._reset_training_state()

    def spawn(self, *, seed: int, learning_rate: float):
        spawned = type(self)(
            name=self.name,
            device=self.device,
            config=self.config,
            seed=seed,
            learning_rate=learning_rate,
        )
        return spawned

    def inspect(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "variant": VARIANT,
            "device": self.device,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dpo_beta": self.dpo_beta,
            "parameters": model_parameters(self.model),
            "ternary_parameters": ternary_parameters(self.model),
            "config": asdict(self.config),
            "tokenizer": TOKENIZER_NAME,
            "training_recipe": self.training_recipe_version,
        }

    def _encode(self, text: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        return list(tokenizer.encode(text, out_type=int))

    def _special_id(self, token: str) -> int:
        tokenizer = self._get_tokenizer()
        token_id = int(tokenizer.piece_to_id(token))
        if token_id < 0 or tokenizer.id_to_piece(token_id) != token:
            raise RuntimeError(f"tokenizer lacks required special token {token}")
        return token_id

    def _causal_row(self, row: Mapping[str, Any], *, training: bool) -> tuple[list[int], list[int]]:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("causal pretraining rows require non-empty text")
        eos = self._special_id("<|eos|>")
        tokens = self._encode(text.strip()) + [eos]
        if len(tokens) < 2:
            raise ValueError("pretraining row is too short")
        maximum = self.config.context + 1
        if len(tokens) > maximum:
            max_start = len(tokens) - maximum
            start = (
                int(torch.randint(0, max_start + 1, (1,), generator=self._data_generator).item())
                if training
                else 0
            )
            tokens = tokens[start : start + maximum]
        inputs = tokens[:-1]
        labels = tokens[1:]
        return inputs, labels

    def _conversation_row(
        self,
        row: Mapping[str, Any],
        *,
        assistant_only: bool,
    ) -> tuple[list[int], list[int]]:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("conversation rows require at least user and assistant messages")

        user_id = self._special_id("<|user|>")
        assistant_id = self._special_id("<|assistant|>")
        eos_id = self._special_id("<|eos|>")
        sequence: list[int] = []
        supervised: list[bool] = []
        saw_assistant = False

        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("conversation message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError("conversation messages require user/assistant roles and non-empty content")
            prefix_id = user_id if role == "user" else assistant_id
            piece = [prefix_id] + self._encode("\n" + content.strip() + "\n")
            sequence.extend(piece)
            supervised.extend([role == "assistant"] * len(piece))
            if role == "assistant":
                saw_assistant = True

        sequence.append(eos_id)
        supervised.append(saw_assistant)
        if not saw_assistant:
            raise ValueError("conversation row has no assistant response")
        if len(sequence) > self.config.context + 1:
            raise ValueError(
                f"conversation exceeds {self.config.context}-token model context"
            )
        inputs = sequence[:-1]
        targets = sequence[1:]
        if assistant_only:
            mask = supervised[1:]
            targets = [target if keep else -100 for target, keep in zip(targets, mask)]
        return inputs, targets

    def _preference_sequence(self, prompt: str, response: str) -> tuple[list[int], list[int]]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("preference rows require a non-empty prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("preference rows require non-empty chosen/rejected responses")
        user_id = self._special_id("<|user|>")
        assistant_id = self._special_id("<|assistant|>")
        eos_id = self._special_id("<|eos|>")
        prefix = [user_id] + self._encode("\n" + prompt.strip() + "\n") + [assistant_id]
        completion = self._encode("\n" + response.strip() + "\n") + [eos_id]
        sequence = prefix + completion
        if len(sequence) > self.config.context + 1:
            raise ValueError(
                f"preference pair exceeds {self.config.context}-token model context"
            )
        inputs = sequence[:-1]
        targets = sequence[1:]
        supervised_start = max(0, len(prefix) - 1)
        targets = [
            token if index >= supervised_start else -100
            for index, token in enumerate(targets)
        ]
        return inputs, targets

    def _preference_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, list) or not batch:
            raise ValueError("preference batch must contain rows")
        pad_id = int(self._get_tokenizer().pad_id())
        prepared: list[tuple[list[int], list[int]]] = []
        max_length = 1
        for row in batch:
            if not isinstance(row, Mapping):
                raise ValueError("preference row must be an object")
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            for response in (chosen, rejected):
                row_inputs, row_targets = self._preference_sequence(prompt, response)
                prepared.append((row_inputs, row_targets))
                max_length = max(max_length, len(row_inputs))
        inputs = torch.full((len(prepared), max_length), pad_id, dtype=torch.long)
        targets = torch.full((len(prepared), max_length), -100, dtype=torch.long)
        for index, (row_inputs, row_targets) in enumerate(prepared):
            length = len(row_inputs)
            inputs[index, :length] = torch.tensor(row_inputs, dtype=torch.long)
            targets[index, :length] = torch.tensor(row_targets, dtype=torch.long)
        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    @staticmethod
    def _sequence_logps(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = model(inputs)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(targets)
        mask = targets.ne(-100)
        return -(losses * mask).sum(dim=1)

    def _dpo_observed(self, batch: Any, *, backward: bool, divisor: int = 1) -> Mapping[str, float]:
        if self.reference_model is None:
            raise RuntimeError(
                "preference-dpo-v1 requires a stage-start reference model; "
                "begin from a verified SFT/judgment checkpoint"
            )
        inputs, targets = self._preference_tensors(batch)
        self.model.train(mode=backward)
        self.reference_model.eval()
        with self._autocast():
            policy_logps = self._sequence_logps(self.model, inputs, targets)
            with torch.no_grad():
                reference_logps = self._sequence_logps(self.reference_model, inputs, targets)
            chosen_policy, rejected_policy = policy_logps[0::2], policy_logps[1::2]
            chosen_reference, rejected_reference = reference_logps[0::2], reference_logps[1::2]
            preference_logits = (
                (chosen_policy - rejected_policy)
                - (chosen_reference - rejected_reference)
            )
            loss = -F.logsigmoid(self.dpo_beta * preference_logits).mean()
        if backward:
            (loss / divisor).backward()
        loss_value = float(loss.detach())
        return {
            "train_loss": loss_value,
            "dpo_loss": loss_value,
            "preference_margin": float(preference_logits.detach().mean()),
        }

    def _tensors(self, batch: Any, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, list) or not batch:
            raise ValueError("batch must contain rows")
        pad_id = int(self._get_tokenizer().pad_id())
        if pad_id < 0:
            raise RuntimeError("foundation tokenizer has no pad token")
        prepared: list[tuple[list[int], list[int]]] = []
        max_length = 1
        for raw in batch:
            if not isinstance(raw, Mapping):
                raise ValueError("dataset row must be an object")
            if self.training_recipe_version == "causal-pretrain-v1":
                row_inputs, row_targets = self._causal_row(raw, training=training)
            elif self.training_recipe_version == "reasoning-mid-v1":
                row_inputs, row_targets = self._conversation_row(raw, assistant_only=False)
            else:
                row_inputs, row_targets = self._conversation_row(raw, assistant_only=True)
            row_inputs = row_inputs[: self.config.context]
            row_targets = row_targets[: self.config.context]
            prepared.append((row_inputs, row_targets))
            max_length = max(max_length, len(row_inputs))

        inputs = torch.full(
            (len(batch), max_length),
            pad_id,
            dtype=torch.long,
        )
        targets = torch.full(
            (len(batch), max_length),
            -100,
            dtype=torch.long,
        )
        for index, (row_inputs, row_targets) in enumerate(prepared):
            length = len(row_inputs)
            inputs[index, :length] = torch.tensor(row_inputs, dtype=torch.long)
            targets[index, :length] = torch.tensor(row_targets, dtype=torch.long)

        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def _autocast(self):
        if not self.device.startswith("cuda"):
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def begin_accumulation(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_step(self, batch: Any, *, divisor: int) -> Mapping[str, float]:
        if not isinstance(divisor, int) or divisor < 1:
            raise ValueError("gradient accumulation divisor must be a positive integer")
        if self.training_recipe_version == "preference-dpo-v1":
            return self._dpo_observed(batch, backward=True, divisor=divisor)
        self.model.train()
        inputs, targets = self._tensors(batch, training=True)
        with self._autocast():
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        (loss / divisor).backward()
        return {"train_loss": float(loss.detach())}

    def finish_accumulation(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, batch: Any) -> Mapping[str, float]:
        self.begin_accumulation()
        observed = self.accumulate_step(batch, divisor=1)
        self.finish_accumulation()
        return observed

    @torch.inference_mode()
    def evaluate(self, batch: Any) -> Mapping[str, float]:
        if self.training_recipe_version == "preference-dpo-v1":
            observed = dict(self._dpo_observed(batch, backward=False))
            return {
                "val_loss": float(observed["dpo_loss"]),
                "dpo_loss": float(observed["dpo_loss"]),
                "preference_margin": float(observed["preference_margin"]),
            }
        self.model.eval()
        inputs, targets = self._tensors(batch, training=False)
        with self._autocast():
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return {"val_loss": float(loss.detach())}

    def save(self, path: Path) -> Mapping[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        files = {
            "model": "model.pt",
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "random_state": "random_state.pt",
        }
        torch.save(
            {
                "variant": VARIANT,
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "tokenizer": TOKENIZER_NAME,
            },
            path / files["model"],
        )
        if self.reference_model is not None:
            files["reference_model"] = "reference_model.pt"
            torch.save(
                {"variant": VARIANT, "config": asdict(self.config), "state_dict": self.reference_model.state_dict()},
                path / files["reference_model"],
            )
        torch.save(self.optimizer.state_dict(), path / files["optimizer"])
        torch.save(self.scheduler.state_dict(), path / files["scheduler"])
        random_state = {
            "torch": torch.get_rng_state(),
            "data_generator": self._data_generator.get_state(),
        }
        if torch.cuda.is_available():
            random_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(random_state, path / files["random_state"])
        return {"files": files}

    def load(self, path: Path, *, weights_only: bool = False) -> None:
        payload = torch.load(path / "model.pt", map_location=self.device, weights_only=True)
        if payload.get("variant") != VARIANT or payload.get("config") != asdict(self.config):
            raise RuntimeError("checkpoint architecture differs from STATIC foundation adapter")
        self.model.load_state_dict(payload["state_dict"], strict=True)
        reference_path = path / "reference_model.pt"
        if reference_path.is_file():
            reference_payload = torch.load(reference_path, map_location=self.device, weights_only=True)
            if reference_payload.get("variant") != VARIANT or reference_payload.get("config") != asdict(self.config):
                raise RuntimeError("reference checkpoint architecture differs from STATIC foundation adapter")
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.load_state_dict(reference_payload["state_dict"], strict=True)
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)
        elif self.training_recipe_version == "preference-dpo-v1":
            self.reference_model = copy.deepcopy(self.model).eval()
            self.reference_model.requires_grad_(False)
        if weights_only:
            return
        optimizer_path = path / "optimizer.pt"
        scheduler_path = path / "scheduler.pt"
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device, weights_only=True)
            )
        if scheduler_path.is_file():
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=self.device, weights_only=True)
            )
        random_state_path = path / "random_state.pt"
        if random_state_path.is_file():
            random_state = torch.load(random_state_path, map_location="cpu", weights_only=True)
            torch.set_rng_state(random_state["torch"])
            if "data_generator" in random_state:
                self._data_generator.set_state(random_state["data_generator"])
            if self.device.startswith("cuda") and "cuda" in random_state:
                torch.cuda.set_rng_state_all(random_state["cuda"])

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt required")
        self.model.eval()
        tokenizer = self._get_tokenizer()
        eos_id = self._special_id("<|eos|>")
        assistant_id = self._special_id("<|assistant|>")
        user_id = self._special_id("<|user|>")
        prompt_ids = [user_id] + self._encode("\n" + prompt.strip() + "\n") + [assistant_id]
        output: list[int] = []
        for _ in range(max_new_tokens):
            context = (prompt_ids + output)[-self.config.context :]
            tokens = torch.tensor([context], device=self.device, dtype=torch.long)
            with self._autocast():
                logits = self.model(tokens)[0, -1].float()
            if temperature <= 0:
                selected = int(torch.argmax(logits).item())
            else:
                logits = logits / max(float(temperature), 0.05)
                limit = min(max(int(top_k), 1), self.config.vocab_size)
                values, indices = torch.topk(logits, limit)
                probs = F.softmax(values, dim=-1)
                selected = int(indices[torch.multinomial(probs, 1)].item())
            if selected == eos_id:
                break
            output.append(selected)
        return tokenizer.decode(output).strip()
