"""Leitura de UM patch DL4GAM, com a mesma regra do notebook fornecido."""
from pathlib import Path
import numpy as np
from .predictor import BANDS


def prepare_nc(source, destination):
    import xarray as xr
    with xr.open_dataset(source, engine='netcdf4', cache=False) as ds:
        names = ds['band_data'].attrs.get('long_name')
        if not isinstance(names, (list, tuple, np.ndarray)):
            raise ValueError('long_name deve conter os nomes das bandas.')
        names = [str(n) for n in names]
        missing = set(BANDS+('CLOUDLESS_MASK', 'FILL_MASK'))-set(names)
        if missing:
            raise ValueError(f'Bandas ausentes: {sorted(missing)}')
        image = ds['band_data'].isel(band=[names.index(b) for b in BANDS])
        image = image.transpose('band', 'y', 'x').values.astype(np.float32)
        ids = ds['mask_all_g_id'].transpose('y', 'x').values
        truth = (np.isfinite(ids) & (ids >= 0)).astype(np.uint8)
        invalid = ~np.isfinite(image).all(axis=0)
        for name in ('CLOUDLESS_MASK', 'FILL_MASK'):
            quality = ds['band_data'].isel(band=names.index(name)).transpose('y', 'x').values
            invalid |= ~np.isfinite(quality) | (quality < .5)
        if 'mask_nok' in ds:
            quality = ds['mask_nok'].transpose('y', 'x').values
            invalid |= ~np.isfinite(quality) | (quality != 0)
        truth[invalid] = 255
        image[:, invalid] = 0
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, image=image, truth=truth)


def synthetic_scene(height=385, width=449):
    y, x = np.mgrid[:height, :width]
    values = (.5+.32*np.sin(x/33)+.18*np.cos(y/27)).clip(.01, .99).astype(np.float32)
    image = np.repeat(values[None], 13, axis=0)
    truth = (values >= .5).astype(np.uint8)
    truth[:7, :11] = 255
    return image, truth
