"""FlagEmbedding-провайдер: настоящий BGE-M3 dense + learned sparse (вариант A).

lexical_weights — это {token_id: weight}; используем token_id как индекс sparse,
weight как значение. IDF не нужен (веса уже абсолютные) -> sparse_uses_idf=False.
На CPU тяжелее FastEmbed (fp32), поэтому use_fp16=False. Опциональная зависимость
(pip install -e ".[flag]").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .base import Embedding, EmbeddingProvider, SparseVector

logger = logging.getLogger(__name__)


class FlagProvider(EmbeddingProvider):
    name = "flag"
    sparse_uses_idf = False

    def __init__(
        self, model_name: str = "BAAI/bge-m3", dim: int = 1024, quantize: bool = True
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        # На CPU fp16 не поддерживается/медленнее -> отключаем.
        self._model = BGEM3FlagModel(model_name, use_fp16=False)
        if quantize:
            self.quantized = self._try_quantize()
        # Размерность определяем по модели (для BGE-M3 это 1024).
        self.dim = len(self._encode(["x"])[0].dense)

    def _try_quantize(self) -> bool:
        """Динамическая int8-квантизация Linear-слоёв. По замерам RSS для BGE-M3 НЕ
        снижается (fp32-копия удерживается на пике) — польза в основном по скорости CPU,
        не по памяти; default OFF (см. ADR-004). При проблеме — тихий фолбэк в fp32."""
        try:
            import torch

            inner = getattr(self._model, "model", None)
            if inner is None:
                return False
            self._model.model = torch.quantization.quantize_dynamic(
                inner, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("BGE-M3: включена int8-квантизация (dynamic).")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Квантизация BGE-M3 не удалась, работаем в fp32: %s", e)
            return False

    def _encode(self, texts: list[str]) -> list[Embedding]:
        out = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = out["dense_vecs"]
        lexical = out["lexical_weights"]
        result: list[Embedding] = []
        for d, lw in zip(dense_vecs, lexical, strict=True):
            indices = [int(k) for k in lw.keys()]
            values = [float(v) for v in lw.values()]
            result.append(
                Embedding(
                    dense=[float(x) for x in d.tolist()], sparse=SparseVector(indices, values)
                )
            )
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Embedding:
        return self._encode([text])[0]
