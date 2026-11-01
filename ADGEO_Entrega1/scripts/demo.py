"""Entrega 1: RPC real, costura e eleição, sem replicação de jobs."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import grpc
import numpy as np
from adgeo.client import Client
from adgeo.cli import save_result
from adgeo.dataset import synthetic_scene
from adgeo.proto import cluster_pb2 as pb
from adgeo.tiles import encode, decode
from .cluster_support import Cluster, stable_leader


async def run_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cluster = Cluster(directory/'peers', delay=.04)
    cluster.start_all()
    client = Client(cluster.cfg)
    image, truth = synthetic_scene(577, 641)
    expected = (image[0] >= .5).astype(np.uint8)
    expected[truth == 255] = 255
    async def job(job_id, output):
        request = pb.JobRequest(job_id=job_id, image_npy=encode(image),
            truth_npy=encode(truth), tile_size=128, overlap=32, model_id='demo-v1')
        start = time.monotonic()
        await client.submit(request)
        result = await client.wait(job_id)
        assert np.array_equal(decode(result.mask_npy), expected)
        save_result(result, directory/output)
        return dict(metrics=json.loads(result.metrics_json),
                    total_seconds=time.monotonic()-start, mask_equals_local=True)
    try:
        await stable_leader(client, 3, [1, 2, 3])
        normal = await job('entrega1-normal', 'normal')
        cluster.stop(3)
        start = time.monotonic()
        await stable_leader(client, 2, [1, 2])
        election_seconds = time.monotonic()-start
        try:
            await client.stubs[2].GetJob(pb.JobKey(job_id='entrega1-normal'), timeout=2)
            raise AssertionError('Entrega 1 não deveria replicar o estado do job.')
        except grpc.RpcError as error:
            assert error.code() == grpc.StatusCode.NOT_FOUND
        after = await job('entrega1-novo-job', 'after_election')
        assert after['metrics']['coordinator'] == 2
        summary = dict(stage=1, passed=True, initial_leader=3, final_leader=2,
            election_seconds=election_seconds, previous_job_replicated=False,
            normal=normal, after_election=after, synthetic_data=True)
        (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    finally:
        await client.close()
        cluster.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/tests/demo')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.output)), indent=2))


if __name__ == '__main__':
    main()
