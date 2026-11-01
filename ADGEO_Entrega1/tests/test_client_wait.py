"""O prazo do job cobre várias tentativas de descobrir o líder."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
import adgeo.client as client_module
from adgeo.client import Client
from adgeo.proto import cluster_pb2 as pb


class Clock:
    def __init__(self):
        self.now = 0.0

    async def sleep(self, delay):
        self.now += delay


def use_clock(monkeypatch, clock):
    # Substitui apenas os objetos vistos pelo cliente, preservando o relógio
    # real do event loop e sem executar esperas de 15/120 segundos no teste.
    monkeypatch.setattr(client_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(client_module, 'asyncio', SimpleNamespace(
        sleep=clock.sleep, wait_for=asyncio.wait_for))


def test_wait_retries_discovery_timeout_then_returns_job(monkeypatch):
    async def run():
        clock = Clock()
        use_clock(monkeypatch, clock)
        client = Client.__new__(Client)
        expected = pb.JobRecord(request=pb.JobRequest(job_id='job'), state='done')
        stub = SimpleNamespace(GetJob=AsyncMock(return_value=expected))
        client.stubs = {2: stub}
        attempts = []

        async def discover(timeout):
            attempts.append(timeout)
            if len(attempts) == 1:
                clock.now += timeout
                raise TimeoutError('Nenhum líder nesta janela de descoberta.')
            return 2, pb.Status(node_id=2, leader_id=2)

        client.leader = discover
        assert await client.wait('job', timeout=40) == expected
        assert len(attempts) == 2
        assert attempts[0] == 15
        assert clock.now == pytest.approx(15.2)
        stub.GetJob.assert_awaited_once()
    asyncio.run(run())


def test_wait_stops_at_global_deadline_when_discovery_keeps_failing(monkeypatch):
    async def run():
        clock = Clock()
        use_clock(monkeypatch, clock)
        client = Client.__new__(Client)
        client.stubs = {}
        attempts = []

        async def discover(timeout):
            attempts.append(timeout)
            clock.now += min(.35, timeout)
            raise TimeoutError('Eleição ainda em andamento.')

        client.leader = discover
        with pytest.raises(TimeoutError, match='Tempo de espera de job esgotado'):
            await client.wait('job', timeout=1)
        assert len(attempts) == 2
        assert clock.now == pytest.approx(1)
        assert attempts == pytest.approx([1, .45])
    asyncio.run(run())
