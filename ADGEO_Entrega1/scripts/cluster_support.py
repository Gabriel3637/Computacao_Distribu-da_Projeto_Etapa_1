from pathlib import Path
import asyncio
import json
import os
import socket
import shutil
import subprocess
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[1]

class Cluster:
    def __init__(self, directory, count=3, delay=.06, extra=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_directory = self.directory/('_state_'+uuid.uuid4().hex)
        self.state_directory.mkdir()
        self._created_state_directory = self.state_directory.resolve()
        # Reserva portas até obter todas, evitando repetição dentro da seleção.
        sockets = []
        for _ in range(count):
            s = socket.socket()
            s.bind(('127.0.0.1', 0))
            sockets.append(s)
        self.cfg = dict(peers=[dict(id=i+1, address=f'127.0.0.1:{s.getsockname()[1]}')
                               for i, s in enumerate(sockets)],
                        heartbeat_interval=.15, heartbeat_timeout=.25,
                        failure_timeout=.9, election_timeout=2.5, tile_timeout=8)
        for s in sockets:
            s.close()
        self.config_path = self.directory/'cluster.json'
        self.config_path.write_text(json.dumps(self.cfg, indent=2), encoding='utf-8')
        self.processes = {}
        self.files = []
        self.delay = delay
        self.extra = extra or []

    def start(self, node):
        log = (self.directory/f'node-{node}.log').open(
            'a' if node in self.processes else 'w', encoding='utf-8')
        self.files.append(log)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        self.processes[node] = subprocess.Popen([
            sys.executable, '-m', 'adgeo', 'node', '--id', str(node),
            '--config', str(self.config_path), '--data-dir', str(self.state_directory/f'node-{node}'),
            '--demo-delay', str(self.delay), *self.extra], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, creationflags=flags)

    def start_all(self):
        for p in self.cfg['peers']:
            self.start(p['id'])

    def stop(self, node):
        p = self.processes[node]
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)

    def close(self):
        for node in self.processes:
            self.stop(node)
        for file in self.files:
            file.close()
        # Só remove o diretório temporário criado por esta instância. Os logs,
        # resultados e demais arquivos escolhidos pelo usuário são preservados.
        target = self.state_directory.resolve()
        if (target != self._created_state_directory or target.parent != self.directory
                or not target.name.startswith('_state_')):
            raise RuntimeError('Diretório temporário fora da área esperada.')
        # Só aplica o namespace de caminhos longos depois de validar o alvo.
        # Sem ele, rmtree pode localizar a pasta e falhar nos hashes dos arquivos
        # filhos ao ultrapassarem MAX_PATH no Windows.
        if os.name == 'nt':
            prefix = os.sep * 2 + '?' + os.sep
            raw = str(target)
            if not raw.startswith(prefix):
                if raw.startswith(os.sep * 2):
                    raw = 'UNC' + raw[1:]
                target = Path(prefix + raw)
        for attempt in range(6):
            try:
                if target.exists():
                    shutil.rmtree(target)
                break
            except PermissionError:
                # No Windows, o antivírus pode manter o arquivo recém-gravado
                # aberto por um instante após o término de todos os processos.
                if attempt == 5:
                    print(f'Estado temporário preservado por bloqueio do Windows: {target}', file=sys.stderr)
                else:
                    time.sleep(.2)

async def stable_leader(client, expected, alive, timeout=20):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        states = await client.statuses()
        if all(states[i] and states[i].leader_id == expected for i in alive):
            return
        await asyncio.sleep(.1)
    raise AssertionError(f'Não convergiu para líder {expected}.')
