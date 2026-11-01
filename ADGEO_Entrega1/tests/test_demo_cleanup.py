"""Limpeza do estado temporário, inclusive acima de MAX_PATH no Windows."""
import os
from pathlib import Path
import pytest
from scripts.demo import Cluster


def writable_path(path):
    """Permite criar a estrutura longa que o teste precisa depois remover."""
    resolved = path.resolve()
    if os.name != 'nt':
        return resolved
    raw = str(resolved)
    prefix = os.sep * 2 + '?' + os.sep
    if raw.startswith(prefix):
        return resolved
    if raw.startswith(os.sep * 2):
        raw = 'UNC' + raw[1:]
    return Path(prefix + raw)


def test_cleanup_removes_long_directory_and_file(tmp_path):
    cluster = Cluster(tmp_path/'demo', count=1)
    original_state = cluster.state_directory
    nested = original_state/('nested_'+'a'*80)/('nested_'+'b'*80)/'jobs'
    filename = nested/(('c'*64)+'.'+('d'*32)+'.tmp')
    assert len(str(filename.resolve())) > 300
    long_folder = writable_path(nested)
    long_folder.mkdir(parents=True)
    long_file = writable_path(filename)
    long_file.write_bytes(b'estado temporario')
    sentinel = cluster.directory/'evidence.txt'
    sentinel.write_text('preservar', encoding='utf-8')
    cluster.close()
    assert not writable_path(original_state).exists()
    assert sentinel.read_text(encoding='utf-8') == 'preservar'
    assert cluster.config_path.exists()


def test_cleanup_rejects_directories_not_created_by_this_instance(tmp_path):
    cluster = Cluster(tmp_path/'demo', count=1)
    original_state = cluster.state_directory
    others = (cluster.directory/'_state_outra_instancia', tmp_path/'_state_externo')
    try:
        for other in others:
            other.mkdir()
            sentinel = other/'nao_apagar.txt'
            sentinel.write_text('preservar', encoding='utf-8')
            cluster.state_directory = other
            with pytest.raises(RuntimeError, match='fora da área esperada'):
                cluster.close()
            assert sentinel.read_text(encoding='utf-8') == 'preservar'
    finally:
        cluster.state_directory = original_state
        cluster.close()
