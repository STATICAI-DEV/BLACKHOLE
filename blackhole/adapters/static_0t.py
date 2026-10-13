"""Audited STATIC-0T adapter derived from the owner-supplied source archive."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

VARIANT = "STATIC-0T-STE-v1"
SOURCE_ARCHIVE_SHA256 = "fbc24587333b2f0303dd7e8ee6ada0fb7fb64760e732b34e5035fcbf5bb9e331"
SOURCE_MODEL_SHA256 = "f464f66fa2c6201d1fea209bdfe0ce0d27a90af268d260107359e72e17318355"


@dataclass(frozen=True)
class StaticConfig:
    width: int = 128
    layers: int = 8
    context: int = 256
    vocab_size: int = 256


def ternary_codes_and_scale(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    axes = tuple(range(1, weight.ndim))
    scale = weight.detach().abs().mean(dim=axes, keepdim=True).clamp_min(1e-8)
    codes = torch.round(weight.detach() / scale).clamp(-1, 1).to(torch.int8)
    return codes, scale


def fake_ternary_weight(weight: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        codes, scale = ternary_codes_and_scale(weight)
    normalized = weight / scale
    quantized = normalized + (codes.to(weight.dtype) - normalized).detach()
    return quantized * scale


class TernaryConv1d(nn.Conv1d):
    packed_inference = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.packed_inference else fake_ternary_weight(self.weight)
        return F.conv1d(
            value,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class TernaryLinear(nn.Linear):
    packed_inference = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.packed_inference else fake_ternary_weight(self.weight)
        return F.linear(value, weight, self.bias)


class StaticBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.near = TernaryConv1d(width, width * 2, 3)
        self.far = TernaryConv1d(width, width * 2, 3, dilation=dilation)
        self.select = TernaryConv1d(width, width, 1)
        self.project = TernaryConv1d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value).transpose(1, 2)
        near = self.near(F.pad(normalized, (2, 0)))
        far = self.far(F.pad(normalized, (2 * self.dilation, 0)))
        first, second = (near + far).chunk(2, dim=1)
        candidate = torch.tanh(first) * torch.sigmoid(second)
        selected = self.project(candidate) * torch.sigmoid(self.select(normalized))
        return value + selected.transpose(1, 2)


class StaticLM(nn.Module):
    def __init__(self, config: StaticConfig) -> None:
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(StaticBlock(config.width, 2**index) for index in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.head = TernaryLinear(config.width, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.cfg.context:
            raise ValueError(f"expected [batch, time], max context {self.cfg.context}")
        value = self.embedding(tokens)
        for block in self.blocks:
            value = block(value)
        return self.head(self.norm(value))


def model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def ternary_parameters(model: nn.Module) -> int:
    return sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, (TernaryLinear, TernaryConv1d))
    )


def _pack(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.int16)
    encoded = torch.where(flat == -1, 2, torch.where(flat == 1, 1, 0)).to(torch.uint8)
    padded = F.pad(encoded, (0, (-len(encoded)) % 4)).reshape(-1, 4).to(torch.int16)
    return (
        padded[:, 0]
        | (padded[:, 1] << 2)
        | (padded[:, 2] << 4)
        | (padded[:, 3] << 6)
    ).to(torch.uint8)


def _unpack(packed: torch.Tensor, shape: list[int]) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    raw = packed.to(torch.int16)
    codes = torch.stack(
        ((raw >> 0) & 3, (raw >> 2) & 3, (raw >> 4) & 3, (raw >> 6) & 3),
        dim=-1,
    ).reshape(-1)[:count]
    if bool((codes == 3).any().item()):
        raise ValueError("invalid ternary weight code 3 in packed checkpoint")
    return torch.where(codes == 2, -1, codes).reshape(shape).to(torch.float32)


def save_packed(model: StaticLM, path: Path | str) -> None:
    parts: dict[str, dict[str, Any]] = {}
    ternary_names = {
        f"{name}.weight"
        for name, layer in model.named_modules()
        if isinstance(layer, (TernaryLinear, TernaryConv1d))
    }
    for name, tensor in model.state_dict().items():
        data = tensor.detach().cpu()
        if name in ternary_names:
            codes, scale = ternary_codes_and_scale(data)
            parts[name] = {
                "kind": "ternary2",
                "shape": list(data.shape),
                "packed": _pack(codes),
                "scale": scale.squeeze().half(),
            }
        else:
            parts[name] = {"kind": "fp16", "data": data.half()}
    torch.save({"variant": VARIANT, "config": asdict(model.cfg), "parts": parts}, Path(path))


def load_packed(path: Path | str, *, device: str = "cpu") -> StaticLM:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("variant") != VARIANT:
        raise ValueError("not a STATIC-0T packed checkpoint")
    model = StaticLM(StaticConfig(**payload["config"]))
    state: dict[str, torch.Tensor] = {}
    for name, part in payload["parts"].items():
        if part["kind"] == "ternary2":
            codes = _unpack(part["packed"], part["shape"])
            scale_shape = [part["shape"][0]] + [1] * (len(part["shape"]) - 1)
            state[name] = codes * part["scale"].float().reshape(scale_shape)
        else:
            state[name] = part["data"].float()
    model.load_state_dict(state, strict=True)
    for module in model.modules():
        if isinstance(module, (TernaryLinear, TernaryConv1d)):
            module.packed_inference = True
    return model.to(device).eval()


class Static0TAdapter:
    name = "static-0t"

    def __init__(
        self,
        *,
        device: str,
        config: StaticConfig | None = None,
        seed: int = 1234,
        learning_rate: float = 0.0005,
    ) -> None:
        self.device = device
        self.config = config or StaticConfig()
        self.seed = seed
        self.learning_rate = learning_rate
        self._reset_training_state()

    def _reset_training_state(self) -> None:
        seed = self.seed
        learning_rate = self.learning_rate
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("BLOCKED_CUDA: PyTorch CUDA is unavailable")
            torch.cuda.manual_seed_all(seed)
        self.model = StaticLM(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        self._data_generator = torch.Generator(device="cpu")
        self._data_generator.manual_seed(seed)

    def configure_for_training(self, config: Mapping[str, Any]) -> None:
        context_length = config.get("context_length", self.config.context)
        if context_length != self.config.context:
            raise ValueError(f"context_length must be exactly {self.config.context} for STATIC-0T")
        seed = config.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        learning_rate = config.get("learning_rate")
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")
        self.seed = seed
        self.learning_rate = float(learning_rate)
        self._reset_training_state()

    def spawn(self, *, seed: int, learning_rate: float) -> "Static0TAdapter":
        """Create an independent model/optimizer/RNG state in the same CUDA process."""
        return Static0TAdapter(
            device=self.device,
            config=self.config,
            seed=seed,
            learning_rate=learning_rate,
        )

    def inspect(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "variant": VARIANT,
            "device": self.device,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "parameters": model_parameters(self.model),
            "ternary_parameters": ternary_parameters(self.model),
            "config": asdict(self.config),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_model_sha256": SOURCE_MODEL_SHA256,
        }

    def _tensors(self, batch: Any, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, list) or not batch:
            raise ValueError("batch must contain conversation rows")
        inputs = torch.zeros((len(batch), self.config.context), dtype=torch.long)
        targets = torch.full((len(batch), self.config.context), -100, dtype=torch.long)
        for index, row in enumerate(batch):
            messages = row.get("messages") if isinstance(row, Mapping) else None
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or [item.get("role") for item in messages if isinstance(item, Mapping)] != ["user", "assistant"]
            ):
                raise ValueError("each row must contain one user and assistant message")
            prompt = messages[0].get("content")
            answer = messages[1].get("content")
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError("user and assistant content must be non-empty text")
            payload = f"\n[USER]\n{prompt.strip()}\n[STATIC]\n{answer.strip()}\n[END]\n".encode("utf-8")
            if len(payload) < 2:
                raise ValueError("conversation is too short")
            if len(payload) > self.config.context + 1:
                maximum = len(payload) - self.config.context - 1
                start = (
                    int(torch.randint(0, maximum + 1, (1,), generator=self._data_generator).item())
                    if training
                    else 0
                )
                payload = payload[start : start + self.config.context + 1]
            values = torch.tensor(list(payload), dtype=torch.long)
            length = min(self.config.context, len(values) - 1)
            inputs[index, :length] = values[:length]
            targets[index, :length] = values[1 : length + 1]
        return inputs.to(self.device), targets.to(self.device)

    def begin_accumulation(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_step(self, batch: Any, *, divisor: int) -> Mapping[str, float]:
        if not isinstance(divisor, int) or divisor < 1:
            raise ValueError("gradient accumulation divisor must be a positive integer")
        self.model.train()
        inputs, targets = self._tensors(batch, training=True)
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
        self.model.eval()
        inputs, targets = self._tensors(batch, training=False)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return {"val_loss": float(loss.detach())}

    def save(self, path: Path) -> Mapping[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        files = {
            "model": "model.pt",
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
            "random_state": "random_state.pt",
            "packed": "static0t_packed.pt",
        }
        torch.save(
            {"variant": VARIANT, "config": asdict(self.config), "state_dict": self.model.state_dict()},
            path / files["model"],
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
        save_packed(self.model, path / files["packed"])
        return {"files": files}

    def load(self, path: Path) -> None:
        payload = torch.load(path / "model.pt", map_location=self.device, weights_only=True)
        if payload.get("variant") != VARIANT or payload.get("config") != asdict(self.config):
            raise RuntimeError("checkpoint architecture differs from STATIC-0T adapter")
        self.model.load_state_dict(payload["state_dict"], strict=True)
        optimizer_path = path / "optimizer.pt"
        scheduler_path = path / "scheduler.pt"
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))
        if scheduler_path.is_file():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device, weights_only=True))
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
        if not prompt:
            raise ValueError("prompt required")
        self.model.eval()
        token_ids = list(f"\n[USER]\n{prompt}\n[STATIC]\n".encode("utf-8"))
        output: list[int] = []
        for _ in range(max_new_tokens):
            tokens = torch.tensor([token_ids[-self.config.context :]], device=self.device)
            logits = self.model(tokens)[0, -1] / max(float(temperature), 0.05)
            limit = min(max(int(top_k), 1), self.config.vocab_size)
            values, indices = torch.topk(logits, limit)
            selected = int(indices[torch.multinomial(F.softmax(values, dim=-1), 1)].item())
            token_ids.append(selected)
            output.append(selected)
            if b"\n[END]\n" in bytes(output):
                break
        return bytes(output).decode("utf-8", errors="replace").split("\n[END]\n")[0]
