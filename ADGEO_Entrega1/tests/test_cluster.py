import asyncio
import json
import grpc
import numpy as np
import pytest
from scripts.demo import Cluster, run_demo, stable_leader
from adgeo.client import Client
from adgeo.dataset import synthetic_scene
from adgeo.proto import cluster_pb2 as pb
from adgeo.tiles import encode, decode


def test_rpc_election_and_new_job_without_replication(tmp_path):
    result = asyncio.run(run_demo(tmp_path))
    assert result['passed'] and result['final_leader'] == 2
    assert result['previous_job_replicated'] is False
    assert result['normal']['metrics']['tiles'] == 42


@pytest.mark.parametrize('count', [2, 3])
def test_parallel_segmentation_with_at_least_two_nodes(tmp_path, count):
    async def run():
        cluster = Cluster(tmp_path, count=count, delay=0)
        cluster.start_all()
        client = Client(cluster.cfg)
        try:
            await stable_leader(client, count, list(range(1, count+1)))
            image, truth = synthetic_scene(64, 96)
            await client.submit(pb.JobRequest(job_id='paralelo', image_npy=encode(image),
                truth_npy=encode(truth), tile_size=32, overlap=8, model_id='demo-v1'))
            result = await client.wait('paralelo')
            assert json.loads(result.metrics_json)['dice'] == 1
            assert decode(result.mask_npy).shape == (64, 96)
        finally:
            await client.close()
            cluster.close()
    asyncio.run(run())


def test_validation_and_idempotent_submission_in_current_leader(tmp_path):
    async def run():
        cluster = Cluster(tmp_path, count=2, delay=0)
        cluster.start_all()
        client = Client(cluster.cfg)
        try:
            await stable_leader(client, 2, [1, 2])
            image, truth = synthetic_scene(64, 64)
            request = pb.JobRequest(job_id='mesmo-id', image_npy=encode(image),
                truth_npy=encode(truth), tile_size=64, overlap=16, model_id='demo-v1')
            await client.submit(request)
            first = await client.wait(request.job_id)
            await client.submit(request)
            second = await client.wait(request.job_id)
            assert first.mask_npy == second.mask_npy
            altered = pb.JobRequest.FromString(request.SerializeToString())
            altered.overlap = 0
            with pytest.raises(grpc.RpcError) as conflict:
                await client.stubs[2].SubmitJob(altered, timeout=2)
            assert conflict.value.code() == grpc.StatusCode.ALREADY_EXISTS
            with pytest.raises(grpc.RpcError) as invalid:
                await client.stubs[1].Election(pb.ElectionMessage(
                    round=pb.Round(counter=0, initiator=1), candidate_id=1), timeout=2)
            assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        finally:
            await client.close()
            cluster.close()
    asyncio.run(run())
