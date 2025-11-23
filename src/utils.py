from typing import Literal
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration shared across all model types."""

    max_iters: int
    batch_size: int
    num_workers: int
    learning_rate: float
    eval_interval: int
    experiment_name: str
    mask_hints: bool = False


class GPTConfig(BaseConfig):
    """Configuration for GPT-2 style transformer model."""

    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_positions: int = 64
    dropout: float = 0.1
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"


class Metric:
    def __init__(
        self,
        name: str,
        average: Literal["arithmetic", "exponential"] = "arithmetic",
        weight: float = 0.99,
    ):
        """
        Args:
            name: Name of the metric
            average: "arithmetic" for cumulative mean, "exponential" for EMA
            weight: For exponential averaging, the decay weight (typically 0.9-0.999)
                   Higher values = slower decay, more smoothing
        """
        self.name = name
        self.average = average
        self.weight = weight
        self.value = 0
        self.count = 0

        # For EMA cold start correction
        self._weight_sum = 0.0 if average == "exponential" else None

    def update(self, new_value):
        if self.average == "arithmetic":
            # Original behavior: cumulative arithmetic mean
            self.value = (self.value * self.count + new_value) / (self.count + 1)
            self.count += 1
        else:
            # Exponential moving average with bias correction
            if self.count == 0:
                # First value: initialize
                self.value = new_value
                self._weight_sum = 1.0
            else:
                # EMA update: value = weight * old + (1 - weight) * new
                self.value = self.weight * self.value + (1 - self.weight) * new_value
                self._weight_sum = self.weight * self._weight_sum + (1 - self.weight)

            self.count += 1

    @property
    def corrected_value(self):
        """Returns bias-corrected value for EMA (handles cold start)"""
        if self.average == "exponential" and self._weight_sum > 0:
            # Bias correction: divide by sum of weights
            return self.value / self._weight_sum
        return self.value

    def clear(self):
        self.value = 0
        self.count = 0
        if self.average == "exponential":
            self._weight_sum = 0.0
