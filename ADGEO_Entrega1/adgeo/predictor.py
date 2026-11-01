import hashlib
import time
import numpy as np

BANDS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12')


class DemoPredictor:
    """Preditor sintético para testar transporte; não identifica geleiras."""
    model_id = "demo-v1"

    def __init__(self, delay=0):
        self.delay = delay

    def __call__(self, image, valid):
        time.sleep(self.delay)
        return np.clip(image[0], 0, 1).astype(np.float32)


class TorchPredictor:
    def __init__(self, checkpoint, normalization, device="cpu"):
        import torch
        from .unet import UNet
        self.torch = torch
        self.device = torch.device(device)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = weights['modelo']
        channels = tuple(state[f"encoder.{i}.bloco.0.weight"].shape[0] for i in range(4))
        channels += (state['bottleneck.bloco.0.weight'].shape[0],)
        bands = state['encoder.0.bloco.0.weight'].shape[1]
        if bands != len(BANDS):
            raise ValueError("O checkpoint deve ter as 13 bandas do notebook.")
        cfg = weights.get('configuracao', {})
        if cfg.get('bandas', list(BANDS)) != list(BANDS):
            raise ValueError("Ordem de bandas incompatível.")
        self.model = UNet(bands, channels=channels).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        with np.load(normalization, allow_pickle=False) as values:
            self.mean = np.asarray(values['media'], dtype=np.float32)[:, None, None]
            self.std = np.asarray(values['desvio'], dtype=np.float32)[:, None, None]
        if (self.mean.shape != (13, 1, 1) or self.std.shape != (13, 1, 1) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or
                np.any(self.std <= 0)):
            raise ValueError("Normalização inválida.")
        digest = hashlib.sha256()
        for filename in (checkpoint, normalization):
            with open(filename, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
        digest.update(','.join(BANDS).encode())
        self.model_id = 'unet-' + digest.hexdigest()

    def __call__(self, image, valid):
        if image.shape[0] != 13:
            raise ValueError("U-Net exige 13 bandas na ordem documentada.")
        values = (image-self.mean)/self.std
        values[:, ~valid] = 0
        tensor = self.torch.from_numpy(np.ascontiguousarray(values))[None].to(self.device)
        with self.torch.inference_mode():
            out = self.model(tensor).softmax(dim=1)[0, 1]
        return out.cpu().numpy().astype(np.float32)
