"""Regressões de retorno ao anel com uma rodada persistida antiga."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import grpc
from adgeo.node import Node, ForwardOutcome
from adgeo.predictor import DemoPredictor
from adgeo.proto import cluster_pb2 as pb


def make_node(directory):
    cfg = {'peers': [{'id': 1, 'address': '127.0.0.1:1'},
                     {'id': 2, 'address': '127.0.0.1:2'}], 'heartbeat_timeout': .1}
    return Node(1, cfg, directory, DemoPredictor())


def test_rejoining_node_starts_above_remote_round(tmp_path):
    async def run():
        node = make_node(tmp_path)
        heartbeat = pb.Status(node_id=2, leader_id=2,
                              round=pb.Round(counter=100, initiator=2))
        peer = SimpleNamespace(Heartbeat=AsyncMock(return_value=heartbeat),
                               Election=AsyncMock(return_value=pb.Ack(ok=True)))
        node.stubs[2] = peer
        try:
            await node.start_election()
            assert node.round.counter == 101
            assert node.leader == 0
            sent = peer.Election.await_args.args[0]
            assert sent.round.counter == 101 and sent.candidate_id == 1
        finally:
            await node.close()
    asyncio.run(run())


def test_rejected_election_does_not_elect_sender_as_single_survivor(tmp_path):
    async def run():
        node = make_node(tmp_path)
        node.stubs[2] = SimpleNamespace(
            Heartbeat=AsyncMock(return_value=pb.Status(round=pb.Round(counter=4, initiator=2))),
            Election=AsyncMock(return_value=pb.Ack(ok=False)))
        try:
            outcome = await node.forward('Election', pb.ElectionMessage())
            assert outcome == ForwardOutcome.REJECTED
            await node.start_election()
            assert node.round.counter == 5
            assert node.leader == 0
        finally:
            await node.close()
    asyncio.run(run())


def test_only_unreachable_peers_allow_single_survivor_election(tmp_path):
    async def run():
        node = make_node(tmp_path)
        node.stubs[2] = SimpleNamespace(
            Heartbeat=AsyncMock(side_effect=grpc.RpcError('offline')),
            Election=AsyncMock(side_effect=grpc.RpcError('offline')))
        try:
            await node.start_election()
            assert node.leader == 1
        finally:
            await node.close()
    asyncio.run(run())
