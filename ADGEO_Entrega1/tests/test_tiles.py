from io import BytesIO
import numpy as np
import pytest
from adgeo.tiles import split, Stitcher, validate, metrics, encode, decode


@pytest.mark.parametrize('shape,size,overlap', [((1, 19, 7), 32, 0),
    ((3, 193, 227), 64, 17), ((13, 128, 128), 128, 64), ((1, 32, 95), 32, 0)])
def test_split_join_unique_pixels(shape, size, overlap):
    rng = np.random.default_rng(42)
    image = rng.uniform(.01, .99, shape).astype(np.float32)
    valid = np.ones(shape[1:], bool)
    valid[0, 0] = False
    stitcher = Stitcher(shape[1:])
    for patch in split(image, size, overlap, valid):
        assert stitcher.add(patch, patch.image[0])
        assert not stitcher.add(patch, patch.image[0])
    expected = (image[0] >= .5).astype(np.uint8)
    expected[~valid] = 255
    assert np.array_equal(stitcher.finish(valid), expected)
    assert metrics(expected)['valid_pixels'] == np.prod(shape[1:])-1


def test_metrics_global_not_mean_per_tile():
    truth = np.array([[1, 1, 0], [0, 1, 255]], np.uint8)
    predicted = np.array([[1, 0, 1], [0, 1, 1]], np.uint8)
    r = metrics(predicted, truth, 100)
    assert (r['tp'], r['fp'], r['fn'], r['tn']) == (2, 1, 1, 1)
    assert r['iou'] == .5
    assert r['dice'] == pytest.approx(2/3)
    assert r['predicted_area_km2'] == .0003
    assert r['true_area_km2'] == .0003
    empty = metrics(np.full((2, 2), 255), np.full((2, 2), 255))
    assert empty['iou'] is None and empty['dice'] is None


def test_overlap_averages_probabilities_before_threshold():
    image = np.zeros((1, 32, 48), np.float32)
    patches = list(split(image, 32, 16))
    join = Stitcher((32, 48))
    join.add(patches[0], np.full((32, 32), .2))
    join.add(patches[1], np.full((32, 32), .9))
    result = join.finish(np.ones((32, 48), bool))
    assert (result[:, :16] == 0).all()
    assert (result[:, 16:] == 1).all()


def test_npy_round_trip_and_reject_pickle():
    array = np.arange(50, dtype=np.float32).reshape(2, 5, 5)
    assert np.array_equal(decode(encode(array)), array)
    stream = BytesIO()
    np.save(stream, np.array([{'a': 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        decode(stream.getvalue())


def test_validation_and_missing_tiles():
    image = np.zeros((1, 32, 64), np.float32)
    with pytest.raises(ValueError):
        validate(image, None, 32, 32)
    with pytest.raises(ValueError):
        validate(image, np.ones((32, 64))*2, 32, 0)
    with pytest.raises(ValueError):
        Stitcher((32, 64)).finish(np.ones((32, 64), bool))
