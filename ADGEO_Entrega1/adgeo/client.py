import asyncio
import time
import grpc
from .proto import cluster_pb2 as pb, cluster_pb2_grpc as rpc
from .node import OPTIONS


class Client:
    def __init__(self, cfg):
        self.channels = {p['id']: grpc.aio.insecure_channel(p['address'], options=OPTIONS)
                         for p in cfg['peers']}
        self.stubs = {i: rpc.PeerStub(c) for i, c in self.channels.items()}

    async def statuses(self):
        async def get(node, stub):
            try:
                return node, await stub.Heartbeat(pb.Empty(), timeout=.6)
            except grpc.RpcError:
                return node, None
        return dict(await asyncio.gather(*(get(i, s) for i, s in self.stubs.items())))

    async def leader(self, timeout=15):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            states = await self.statuses()
            for node, state in states.items():
                if state and state.leader_id == node:
                    return node, state
            await asyncio.sleep(.2)
        raise TimeoutError('Nenhum líder foi encontrado.')

    async def submit(self, request, timeout=30):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            node, status = await self.leader(timeout=max(.1, end-time.monotonic()))
            if not request.model_id:
                request.model_id = status.model_id
            try:
                return await self.stubs[node].SubmitJob(request, timeout=10)
            except grpc.RpcError as error:
                if error.code() not in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.ABORTED,
                                        grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.DEADLINE_EXCEEDED):
                    raise
                await asyncio.sleep(.3)
        raise TimeoutError('Submissão não confirmada; reutilize o mesmo job_id ao tentar novamente.')

    async def wait(self, job_id, timeout=120):
        end = time.monotonic()+timeout

        async def pause_before_retry():
            remaining = end-time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(.2, remaining))

        while time.monotonic() < end:
            remaining = end-time.monotonic()
            try:
                # Uma janela de descoberta sem líder não esgota a espera do job.
                # O limite externo também impede que um RPC de descoberta mais
                # lento ultrapasse o prazo global restante.
                node, _ = await asyncio.wait_for(
                    self.leader(timeout=min(15, remaining)), timeout=remaining)
            except TimeoutError:
                await pause_before_retry()
                continue
            remaining = end-time.monotonic()
            if remaining <= 0:
                break
            try:
                result = await asyncio.wait_for(
                    self.stubs[node].GetJob(pb.JobKey(job_id=job_id), timeout=min(2, remaining)),
                    timeout=remaining)
                if result.state == 'failed':
                    raise RuntimeError(result.error)
                if result.state == 'done':
                    return result
            except TimeoutError:
                pass
            except grpc.RpcError as error:
                if error.code() not in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.NOT_FOUND,
                                        grpc.StatusCode.DEADLINE_EXCEEDED):
                    raise
            await pause_before_retry()
        raise TimeoutError(f'Tempo de espera de {job_id} esgotado; o job pode continuar no cluster.')

    async def close(self):
        await asyncio.gather(*(c.close() for c in self.channels.values()))
