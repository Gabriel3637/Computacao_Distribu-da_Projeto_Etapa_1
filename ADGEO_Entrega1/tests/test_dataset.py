import numpy as np
import pytest
from adgeo.predictor import BANDS


def test_nc_band_order_and_invalid_pixels(tmp_path):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')
    from adgeo.dataset import prepare_nc
    names = list(reversed(BANDS))+['CLOUDLESS_MASK', 'FILL_MASK']
    data = np.ones((15, 32, 32), np.float32)
    for i, name in enumerate(names):
        if name in BANDS:
            data[i] = BANDS.index(name)+1
    data[names.index('CLOUDLESS_MASK'), 0, 0] = 0
    ids = np.full((32, 32), -1, np.float32)
    ids[4:8, 4:8] = 12
    ds = xr.Dataset({'band_data': (('band', 'y', 'x'), data),
                     'mask_all_g_id': (('y', 'x'), ids)})
    ds['band_data'].attrs['long_name'] = names
    source, output = tmp_path/'patch.nc', tmp_path/'patch.npz'
    ds.to_netcdf(source, engine='netcdf4')
    prepare_nc(source, output)
    with np.load(output, allow_pickle=False) as prepared:
        assert prepared['truth'][0, 0] == 255
        assert prepared['truth'][5, 5] == 1
        assert prepared['truth'][1, 1] == 0
        np.testing.assert_array_equal(prepared['image'][:, 1, 1], np.arange(1, 14))
        assert (prepared['image'][:, 0, 0] == 0).all()
