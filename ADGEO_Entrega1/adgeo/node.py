"""Entrega 1: peer gRPC com LCR e estado de jobs local, em memória.

Hipótese: membros fixos, rede local conectada e falhas por parada. A eleição
não fornece consenso ou segurança sob partições de rede.
"""
import asyncio
from contextlib import suppress
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import grpc
import numpy as np
from .proto import cluster_pb2 as pb, cluster_pb2_grpc as rpc
from .tiles import decode, encode, validate, split, Stitcher, metrics

OPTIONS = [('grpc.max_receive_message_length', 64*1024*1024),
           ('grpc.max_send_message_length', 64*1024*1024)]


def round_key(value):
    return value.counter, value.initiator


def read_config(filename):
    cfg = json.loads(Path(filename).read_text(encoding='utf-8'))
    peers = cfg['peers']
    ids = [p['id'] for p in peers]
    addresses = [p['address'] for p in peers]
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError('IDs devem ser inteiros positivos únicos.')
    if len(set(addresses)) != len(addresses):
        raise ValueError('Endereços repetidos na configuração.')
    for key, default in [('heartbeat_interval', .3), ('heartbeat_timeout', .4),
                         ('failure_timeout', 1.5), ('election_timeout', 4),
                         ('tile_timeout', 30)]:
        cfg.setdefault(key, default)
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} precisa ser positivo e finito.')
    return cfg


class ForwardOutcome(Enum):
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    UNREACHABLE = 'unreachable'


class Node(rpc.PeerServicer):
    def __init__(self, node_id, cfg, directory, predictor):
        self.id = node_id
        self.cfg = cfg
        self.members = {p['id']: p['address'] for p in cfg['peers']}
        if node_id not in self.members:
            raise ValueError('Nó ausente da configuração.')
        ids = sorted(self.members)
        k = ids.index(node_id)
        self.successors = ids[k+1:] + ids[:k]
        self.directory = Path(directory).resolve()
        if os.name == 'nt':
            # Caminhos desta entrega podem exceder MAX_PATH ao incluir hashes.
            prefix = os.sep * 2 + '?' + os.sep
            raw = str(self.directory)
            if not raw.startswith(prefix):
                if raw.startswith(os.sep * 2):
                    raw = 'UNC' + raw[1:]
                self.directory = Path(prefix + raw)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory/'cache').mkdir(exist_ok=True)
        term_file = self.directory/'term.json'
        counter = json.loads(term_file.read_text())['counter'] if term_file.exists() else 0
        self.round = pb.Round(counter=counter)
        self.leader = 0
        self.participant = False
        self.last_election = 0.0
        self.last_leader_ok = time.monotonic()
        self.predictor = predictor
        self.compute_lock = asyncio.Lock()
        self.submit_lock = asyncio.Lock()
        self.computed = self.cache_hits = 0
        self.jobs = {}
        self.active_jobs = {}
        self.background = set()
        self.channels = {i: grpc.aio.insecure_channel(addr, options=OPTIONS)
                         for i, addr in self.members.items()}
        self.stubs = {i: rpc.PeerStub(c) for i, c in self.channels.items()}

    def log(self, event, **values):
        print(json.dumps(dict(time=time.time(), node=self.id, event=event,
                              leader=self.leader, round=list(round_key(self.round)),
                              **values), ensure_ascii=False), flush=True)

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.background.add(task)
        def finished(t):
            self.background.discard(t)
            if not t.cancelled() and t.exception():
                self.log('background_error', error=str(t.exception()))
        task.add_done_callback(finished)
        return task

    def adopt(self, value):
        self.round.CopyFrom(value)
        path = self.directory/'term.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'counter': value.counter}), encoding='utf-8')
        temporary.replace(path)
        self.participant = False
        self.leader = 0
        self.last_election = time.monotonic()

    def set_leader(self, leader):
        changed = self.leader != leader
        self.leader = leader
        self.participant = False
        self.last_leader_ok = time.monotonic()
        if changed:
            self.log('coordinator', elected=leader)

    async def forward(self, method, message):
        # Anel lógico: tenta sucessores em ordem e contorna processos parados.
        rejected = False
        for peer in self.successors:
            try:
                ack = await getattr(self.stubs[peer], method)(message, timeout=self.cfg['heartbeat_timeout'])
                if ack.ok:
                    return ForwardOutcome.ACCEPTED
                rejected = True
                self.log('message_rejected', successor=peer, method=method)
            except grpc.RpcError:
                self.log('successor_unavailable', successor=peer, method=method)
        return ForwardOutcome.REJECTED if rejected else ForwardOutcome.UNREACHABLE

    async def start_election(self):
        async def observed_counter(peer):
            try:
                status = await self.stubs[peer].Heartbeat(pb.Empty(), timeout=self.cfg['heartbeat_timeout'])
                return status.round.counter
            except grpc.RpcError:
                return 0
        # Um peer que ficou offline não precisa percorrer todas as rodadas
        # antigas, uma a uma, antes de conseguir voltar ao anel.
        observed = await asyncio.gather(*(observed_counter(i) for i in self.successors))
        counter = max([self.round.counter, *observed])+1
        self.adopt(pb.Round(counter=counter, initiator=self.id))
        self.participant = True
        self.log('election_started')
        msg = pb.ElectionMessage(round=self.round, candidate_id=self.id)
        if await self.forward('Election', msg) == ForwardOutcome.UNREACHABLE:
            self.set_leader(self.id)

    async def Heartbeat(self, request, context):
        return pb.Status(node_id=self.id, leader_id=self.leader, round=self.round,
                         model_id=self.predictor.model_id, tiles_computed=self.computed,
                         cache_hits=self.cache_hits)

    async def Election(self, request, context):
        if (request.candidate_id not in self.members or
                request.round.initiator not in self.members or request.round.counter == 0):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Eleição inválida.')
        if round_key(request.round) < round_key(self.round):
            return pb.Ack(ok=False)
        if round_key(request.round) > round_key(self.round):
            self.adopt(request.round)
        # Não reabre uma rodada já concluída por duplicação de mensagem.
        if self.leader:
            return pb.Ack(ok=True)
        candidate = request.candidate_id
        if candidate == self.id:
            self.set_leader(self.id)
            self.spawn(self.forward('Coordinator', pb.CoordinatorMessage(
                round=self.round, leader_id=self.id)))
        elif candidate > self.id:
            self.participant = True
            self.spawn(self.forward('Election', request))
        elif not self.participant:
            self.participant = True
            self.spawn(self.forward('Election', pb.ElectionMessage(
                round=self.round, candidate_id=self.id)))
        return pb.Ack(ok=True)

    async def Coordinator(self, request, context):
        if request.leader_id not in self.members:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Líder desconhecido.')
        if round_key(request.round) < round_key(self.round):
            return pb.Ack(ok=False)
        if round_key(request.round) > round_key(self.round):
            self.adopt(request.round)
        if self.leader == request.leader_id:
            return pb.Ack(ok=True)  # encerra a volta e suprime duplicatas
        self.set_leader(request.leader_id)
        self.spawn(self.forward('Coordinator', request))
        return pb.Ack(ok=True)



    def save(self, record):
        # Entrega 1: estado em memória somente no coordenador atual.
        clone = pb.JobRecord.FromString(record.SerializeToString())
        self.jobs[clone.request.job_id] = clone




    def validate_job(self, request):
        if not request.job_id or len(request.job_id) > 128:
            raise ValueError('job_id precisa ter entre 1 e 128 caracteres.')
        if request.model_id != self.predictor.model_id:
            raise ValueError('Modelo ou normalização diverge do nó.')
        image = decode(request.image_npy)
        truth = decode(request.truth_npy) if request.truth_npy else None
        validate(image, truth, request.tile_size, request.overlap)
        if not math.isfinite(request.pixel_area_m2) or request.pixel_area_m2 < 0:
            raise ValueError('Área de pixel inválida.')
        return image, truth

    async def SubmitJob(self, request, context):
        if self.leader != self.id:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Descubra o líder novamente.')
        try:
            self.validate_job(request)
        except (ValueError, KeyError, TypeError, EOFError) as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        async with self.submit_lock:
            if self.leader != self.id:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Descubra o líder novamente.')
            old = self.jobs.get(request.job_id)
            if old and old.request != request:
                await context.abort(grpc.StatusCode.ALREADY_EXISTS, 'job_id reutilizado com outra entrada.')
            record = old or pb.JobRecord(request=request, state='queued')
            self.save(record)
            self.log('job_accepted', job=request.job_id)
            return self.jobs[request.job_id]




    async def GetJob(self, request, context):
        if request.job_id not in self.jobs:
            await context.abort(grpc.StatusCode.NOT_FOUND, 'Job não encontrado neste nó.')
        return self.jobs[request.job_id]

    async def InferTile(self, request, context):
        if (request.model_id != self.predictor.model_id or
                request.leader_id != self.leader or
                round_key(request.round) != round_key(self.round)):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Modelo ou liderança desatualizada.')
        try:
            image, valid = decode(request.image_npy), decode(request.valid_npy)
            validate(image, None, image.shape[-1], 0)
            if image.shape[1] != image.shape[2] or valid.shape != image.shape[1:] or valid.dtype != bool:
                raise ValueError('Tile ou validade incompatível.')
            digest = hashlib.sha256((request.model_id+request.job_id+request.tile_id).encode())
            digest.update(request.image_npy)
            digest.update(request.valid_npy)
            path = self.directory/'cache'/(digest.hexdigest()+'.npy')
            async with self.compute_lock:
                if path.exists():
                    probability = decode(path.read_bytes())
                    self.cache_hits += 1
                else:
                    computation = asyncio.create_task(asyncio.to_thread(self.predictor, image, valid))
                    try:
                        probability = await asyncio.shield(computation)
                    except asyncio.CancelledError:
                        # Cancelar o RPC não interrompe uma thread PyTorch. Mantém
                        # a exclusão local até ela terminar, evitando inferências
                        # simultâneas no mesmo modelo/GPU após um timeout.
                        await computation
                        raise
                    temporary = path.with_suffix('.tmp')
                    temporary.write_bytes(encode(probability))
                    temporary.replace(path)
                    self.computed += 1
            return pb.TileResult(tile_id=request.tile_id, probability_npy=encode(probability),
                                 model_id=self.predictor.model_id, worker_id=self.id)
        except (ValueError, KeyError, TypeError, EOFError) as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

    async def run_job(self, record, term):
        request = record.request
        start = time.monotonic()
        try:
            image, truth = self.validate_job(request)
            valid = np.isfinite(image).all(axis=0)
            if truth is not None:
                valid &= truth != 255
            stitcher = Stitcher(image.shape[1:])
            workers = []
            for peer, stub in self.stubs.items():
                try:
                    status = await stub.Heartbeat(pb.Empty(), timeout=self.cfg['heartbeat_timeout'])
                    if status.model_id == request.model_id:
                        workers.append(peer)
                except grpc.RpcError:
                    pass
            if not workers:
                raise RuntimeError('Nenhum worker disponível com o mesmo modelo.')
            queues = {i: asyncio.Semaphore(1) for i in workers}
            failed = set()
            used = {}
            async def compute(index, patch):
                order = workers[index % len(workers):]+workers[:index % len(workers)]
                errors = []
                for peer in order:
                    if self.leader != self.id or round_key(self.round) != term:
                        raise asyncio.CancelledError()
                    if peer in failed:
                        continue
                    async with queues[peer]:
                        if peer in failed:
                            continue
                        try:
                            result = await self.stubs[peer].InferTile(pb.Tile(
                                job_id=request.job_id, tile_id=patch.id,
                                x=patch.x, y=patch.y, valid_width=patch.width,
                                valid_height=patch.height, image_npy=encode(patch.image),
                                valid_npy=encode(patch.valid), model_id=request.model_id,
                                round=self.round, leader_id=self.id), timeout=self.cfg['tile_timeout'])
                            if result.tile_id != patch.id or result.model_id != request.model_id:
                                raise ValueError('Resposta de tile incompatível.')
                            stitcher.add(patch, decode(result.probability_npy))
                            used[peer] = used.get(peer, 0)+1
                            return
                        except grpc.RpcError as error:
                            errors.append(f'{peer}: {error.code().name}')
                            failed.add(peer)
                            self.log('tile_retry', job=request.job_id, tile=patch.id, worker=peer)
                raise RuntimeError('Falha em todos os workers: '+', '.join(errors))
            # Janela limitada evita manter milhares de arrays/tarefas em memória.
            patches = iter(enumerate(split(image, request.tile_size, request.overlap, valid)))
            while True:
                batch = []
                for _ in range(max(1, len(workers)*2)):
                    item = next(patches, None)
                    if item is None:
                        break
                    batch.append(compute(*item))
                if not batch:
                    break
                async with asyncio.TaskGroup() as group:
                    for computation in batch:
                        group.create_task(computation)
            if self.leader != self.id or round_key(self.round) != term:
                return
            mask = stitcher.finish(valid)
            measured = metrics(mask, truth, request.pixel_area_m2)
            measured.update(tiles=len(stitcher.seen), workers=used,
                            model_id=request.model_id, coordinator=self.id,
                            elapsed_seconds=time.monotonic()-start,
                            demo_backend=request.model_id == 'demo-v1')
            result = pb.JobRecord(request=request, state='done', mask_npy=encode(mask),
                                  metrics_json=json.dumps(measured, allow_nan=False))
            self.save(result)
            self.log('job_done', job=request.job_id, **measured)
        except asyncio.CancelledError:
            self.log('job_interrupted', job=request.job_id)
            raise
        except Exception as error:
            if self.leader == self.id and round_key(self.round) == term:
                result = pb.JobRecord(request=request, state='failed', error=str(error))
                self.save(result)
                self.log('job_failed', job=request.job_id, error=str(error))

    async def monitor(self):
        await asyncio.sleep(.4)
        while True:
            now = time.monotonic()
            if self.leader and self.leader != self.id:
                try:
                    status = await self.stubs[self.leader].Heartbeat(
                        pb.Empty(), timeout=self.cfg['heartbeat_timeout'])
                    if status.leader_id == self.leader and round_key(status.round) == round_key(self.round):
                        self.last_leader_ok = time.monotonic()
                except grpc.RpcError:
                    pass
                if now-self.last_leader_ok > self.cfg['failure_timeout']:
                    self.log('leader_suspected')
                    await self.start_election()
            elif not self.leader and now-self.last_election > self.cfg['election_timeout']:
                await self.start_election()
            for job, task in list(self.active_jobs.items()):
                if task.done():
                    self.active_jobs.pop(job)
                elif self.leader != self.id:
                    task.cancel()
            if self.leader == self.id:
                for job, record in list(self.jobs.items()):
                    if record.state == 'queued' and job not in self.active_jobs:
                        self.active_jobs[job] = self.spawn(self.run_job(record, round_key(self.round)))
            await asyncio.sleep(self.cfg['heartbeat_interval'])

    async def close(self):
        for task in list(self.background):
            task.cancel()
        await asyncio.gather(*self.background, return_exceptions=True)
        await asyncio.gather(*(c.close() for c in self.channels.values()))


async def serve(node_id, cfg, directory, predictor):
    node = Node(node_id, cfg, directory, predictor)
    server = grpc.aio.server(options=OPTIONS)
    rpc.add_PeerServicer_to_server(node, server)
    port = node.members[node_id].rsplit(':', 1)[1]
    if not server.add_insecure_port('0.0.0.0:'+port):
        raise RuntimeError(f'Porta {port} indisponível.')
    await server.start()
    node.log('server_started', address=node.members[node_id], model_id=predictor.model_id)
    monitor = asyncio.create_task(node.monitor())
    try:
        await server.wait_for_termination()
    finally:
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        await node.close()
        await server.stop(0)
