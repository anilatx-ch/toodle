"""Text embedding service backed by DistilBERT."""

from __future__ import annotations

import numpy as np

from src import config

try:  # pragma: no cover - runtime dependency
    import tensorflow as tf
except Exception:  # pragma: no cover - runtime dependency
    tf = None

try:  # pragma: no cover - runtime dependency
    import keras_nlp
except Exception:  # pragma: no cover - runtime dependency
    keras_nlp = None


class EmbeddingService:
    """Lazily loads DistilBERT and emits normalized CLS embeddings."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if tf is None or keras_nlp is None:
            raise RuntimeError("TensorFlow and KerasNLP are required for retrieval embeddings")

        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
            config.BERT_PRESET,
            sequence_length=config.BERT_MAX_LEN,
        )
        backbone = keras_nlp.models.DistilBertBackbone.from_preset(config.BERT_PRESET)
        text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")
        encoded = backbone(preprocessor(text_input))
        if isinstance(encoded, dict):
            encoded = encoded.get("sequence_output") or next(iter(encoded.values()))
        cls_output = tf.keras.layers.Lambda(lambda tensor: tensor[:, 0, :], name="cls_slice")(encoded)
        self._model = tf.keras.Model(inputs=text_input, outputs=cls_output)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_model()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = np.asarray(texts[start : start + batch_size], dtype=object)
            vectors = self._model(batch, training=False).numpy().astype(np.float32)
            chunks.append(vectors)
        return self._normalize(np.vstack(chunks))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_batch([text], batch_size=1)

