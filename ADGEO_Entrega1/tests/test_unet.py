import numpy as np
import pytest


def test_notebook_model_adapter_and_normalization(tmp_path):
    torch = pytest.importorskip('torch')
    from adgeo.unet import UNet
    from adgeo.predictor import TorchPredictor, BANDS
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # Mesmas classes e chaves de pesos; canais menores agilizam o teste de contrato.
    model = UNet(13, channels=(2, 4, 8, 16, 32)).eval()
    weights, norm = tmp_path/'model.pth', tmp_path/'normalizacao.npz'
    torch.save({'modelo': model.state_dict(), 'configuracao': {'bandas': list(BANDS)}}, weights)
    mean, std = np.arange(13, dtype=np.float32), np.full(13, 2, np.float32)
    np.savez(norm, media=mean, desvio=std)
    adapter = TorchPredictor(weights, norm)
    image = np.random.default_rng(42).normal(size=(13, 35, 37)).astype(np.float32)
    valid = np.ones((35, 37), bool)
    valid[:3, :3] = False
    normalized = (image-mean[:, None, None])/std[:, None, None]
    normalized[:, ~valid] = 0
    with torch.inference_mode():
        expected = model(torch.from_numpy(normalized)[None]).softmax(dim=1)[0, 1].numpy()
    np.testing.assert_allclose(adapter(image, valid), expected, rtol=1e-6, atol=1e-7)
    assert adapter.model_id.startswith('unet-')
    with pytest.raises(ValueError):
        adapter(image[:3], valid)
    np.savez(norm, media=mean, desvio=np.zeros(13))
    with pytest.raises(ValueError):
        TorchPredictor(weights, norm)


def test_unet_over_real_grpc_matches_local_tiled_inference(tmp_path):
    import asyncio
    import json
    torch = pytest.importorskip('torch')
    from adgeo.unet import UNet
    from adgeo.predictor import TorchPredictor
    from adgeo.tiles import split, Stitcher, encode, decode
    from adgeo.proto import cluster_pb2 as pb
    from adgeo.client import Client
    from scripts.demo import Cluster, stable_leader
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = UNet(13, channels=(2, 4, 8, 16, 32)).eval()
    weights, norm = tmp_path/'weights.pth', tmp_path/'normalizacao.npz'
    torch.save({'modelo': model.state_dict()}, weights)
    np.savez(norm, media=np.zeros(13, np.float32), desvio=np.ones(13, np.float32))
    predictor = TorchPredictor(weights, norm)
    image = np.random.default_rng(1).normal(size=(13, 96, 144)).astype(np.float32)
    valid = np.ones((96, 144), bool)
    stitcher = Stitcher(valid.shape)
    for patch in split(image, 64, 16):
        stitcher.add(patch, predictor(patch.image, patch.valid))
    expected = stitcher.finish(valid)
    async def run():
        cluster = Cluster(tmp_path/'cluster', delay=0, extra=[
            '--backend', 'unet', '--checkpoint', str(weights), '--normalization', str(norm)])
        cluster.start_all()
        client = Client(cluster.cfg)
        try:
            await stable_leader(client, 3, [1, 2, 3], timeout=45)
            await client.submit(pb.JobRequest(job_id='unet-cpu', image_npy=encode(image),
                tile_size=64, overlap=16, model_id=predictor.model_id))
            result = await client.wait('unet-cpu')
            assert np.array_equal(decode(result.mask_npy), expected)
            assert json.loads(result.metrics_json)['demo_backend'] is False
        finally:
            await client.close()
            cluster.close()
    asyncio.run(run())
