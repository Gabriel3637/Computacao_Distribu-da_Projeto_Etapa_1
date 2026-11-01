"""Divisão espacial, média nas sobreposições e métricas por pixel único."""
from dataclasses import dataclass
from io import BytesIO
import math
import numpy as np

MAX_ARRAY_BYTES = 48 * 1024 * 1024


def encode(array):
    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def decode(payload):
    if len(payload) > MAX_ARRAY_BYTES:
        raise ValueError("Array excede o limite de 48 MiB.")
    stream = BytesIO(payload)
    # Verifica tamanho declarado antes de alocar a matriz.
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Use o formato NPY 1.0 ou 2.0.")
    if dtype.hasobject or math.prod(shape) * dtype.itemsize > MAX_ARRAY_BYTES:
        raise ValueError("Tipo ou tamanho NPY inválido.")
    stream.seek(0)
    return np.load(stream, allow_pickle=False)


def validate(image, truth, size, overlap):
    if image.ndim != 3 or image.dtype != np.float32 or min(image.shape) < 1:
        raise ValueError("Imagem precisa ser float32 [bandas, altura, largura].")
    if not (32 <= size <= 1024 and 0 <= overlap < size):
        raise ValueError("Exige 32 <= tile_size <= 1024 e 0 <= overlap < tile_size.")
    if truth is not None and (truth.shape != image.shape[1:] or
                              not np.isin(truth, [0, 1, 255]).all()):
        raise ValueError("Gabarito incompatível: use 0, 1 e 255.")
    h, w = image.shape[1:]
    n = math.ceil(max(0, h-size)/(size-overlap))+1
    n *= math.ceil(max(0, w-size)/(size-overlap))+1
    if n > 10000:
        raise ValueError("Mais de 10.000 tiles; reduza a cena ou a sobreposição.")


@dataclass
class Patch:
    id: str
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray
    valid: np.ndarray


def positions(length, size, stride):
    if length <= size:
        return [0]
    values = list(range(0, length-size+1, stride))
    if values[-1] != length-size:
        values.append(length-size)
    return values


def split(image, size=128, overlap=32, valid=None):
    validate(image, None, size, overlap)
    _, h, w = image.shape
    if valid is None:
        valid = np.isfinite(image).all(axis=0)
    if valid.shape != (h, w):
        raise ValueError("Máscara de validade incompatível.")
    for y in positions(h, size, size-overlap):
        for x in positions(w, size, size-overlap):
            vh, vw = min(size, h-y), min(size, w-x)
            patch = np.zeros((image.shape[0], size, size), np.float32)
            good = np.zeros((size, size), bool)
            patch[:, :vh, :vw] = np.nan_to_num(image[:, y:y+vh, x:x+vw])
            good[:vh, :vw] = valid[y:y+vh, x:x+vw]
            yield Patch(f"y{y}_x{x}", x, y, vw, vh, patch, good)


class Stitcher:
    def __init__(self, shape):
        self.total = np.zeros(shape, np.float64)
        self.count = np.zeros(shape, np.uint32)
        self.seen = set()

    def add(self, patch, probability):
        if patch.id in self.seen:
            return False
        if (probability.shape != patch.valid.shape or
                not np.isfinite(probability).all() or
                np.any((probability < 0) | (probability > 1))):
            raise ValueError("Probabilidades precisam estar em [0,1], sem NaN.")
        y, x, h, w = patch.y, patch.x, patch.height, patch.width
        good = patch.valid[:h, :w]
        self.total[y:y+h, x:x+w] += probability[:h, :w] * good
        self.count[y:y+h, x:x+w] += good
        self.seen.add(patch.id)
        return True

    def finish(self, valid):
        if np.any(valid & (self.count == 0)):
            raise ValueError("Há pixels válidos sem resultado.")
        probabilities = np.divide(self.total, self.count,
                                  out=np.zeros_like(self.total), where=self.count > 0)
        mask = (probabilities >= 0.5).astype(np.uint8)
        mask[~valid] = 255
        return mask


def metrics(mask, truth=None, pixel_area_m2=0):
    good = mask != 255
    if truth is not None:
        good &= truth != 255
    predicted = int(((mask == 1) & good).sum())
    result = {"valid_pixels": int(good.sum()), "predicted_pixels": predicted}
    if truth is not None:
        tp = int(((mask == 1) & (truth == 1) & good).sum())
        fp = int(((mask == 1) & (truth == 0) & good).sum())
        fn = int(((mask == 0) & (truth == 1) & good).sum())
        tn = int(((mask == 0) & (truth == 0) & good).sum())
        ratio = lambda a, b: a / b if b else None
        result.update(tp=tp, fp=fp, fn=fn, tn=tn, true_pixels=tp+fn,
                      iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn),
                      precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                      accuracy=ratio(tp+tn, tp+fp+fn+tn))
    if pixel_area_m2 > 0:
        result["predicted_area_km2"] = predicted * pixel_area_m2 / 1e6
        if truth is not None:
            result["true_area_km2"] = result["true_pixels"] * pixel_area_m2 / 1e6
            result["area_bias_km2"] = (predicted-result["true_pixels"]) * pixel_area_m2 / 1e6
    return result
