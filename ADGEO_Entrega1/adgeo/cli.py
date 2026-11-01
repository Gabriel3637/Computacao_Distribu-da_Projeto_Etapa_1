import argparse
import asyncio
import json
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from .node import read_config, serve
from .predictor import DemoPredictor, TorchPredictor
from .client import Client
from .proto import cluster_pb2 as pb
from .tiles import encode, decode
from .dataset import prepare_nc, synthetic_scene


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mask = decode(result.mask_npy)
    np.save(directory/'mask.npy', mask, allow_pickle=False)
    view = np.zeros((*mask.shape, 3), np.uint8)
    view[mask == 1] = (65, 175, 235)
    view[mask == 255] = (150, 150, 150)
    Image.fromarray(view).save(directory/'mask.png')
    (directory/'metrics.json').write_text(result.metrics_json, encoding='utf-8')
    (directory/'job_id.txt').write_text(result.request.job_id, encoding='utf-8')


async def remote(args):
    client = Client(read_config(args.config))
    try:
        if args.command == 'status':
            for node, status in (await client.statuses()).items():
                print(json.dumps({'node': node, 'online': status is not None,
                                  'leader': status.leader_id if status else None,
                                  'model_id': status.model_id if status else None,
                                  'tiles': status.tiles_computed if status else None,
                                  'cache_hits': status.cache_hits if status else None}))
        else:
            if args.command == 'submit':
                with np.load(args.input, allow_pickle=False) as data:
                    request = pb.JobRequest(job_id=args.job_id or uuid.uuid4().hex,
                        image_npy=encode(data['image'].astype(np.float32)),
                        truth_npy=encode(data['truth']) if 'truth' in data else b'',
                        tile_size=args.tile_size, overlap=args.overlap,
                        pixel_area_m2=args.pixel_area_m2)
                await client.submit(request)
                job_id = request.job_id
                print('Job confirmado:', job_id, flush=True)
            else:
                job_id = args.job_id
            result = await client.wait(job_id, timeout=args.timeout)
            save_result(result, args.output)
            print(json.dumps(json.loads(result.metrics_json), indent=2, ensure_ascii=False))
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description='ADGEO - gRPC, tiles e eleição LCR')
    commands = parser.add_subparsers(dest='command', required=True)
    node = commands.add_parser('node', help='Executa um peer (coordenador ou worker)')
    node.add_argument('--id', type=int, required=True)
    node.add_argument('--config', default='config/cluster.local.json')
    node.add_argument('--data-dir')
    node.add_argument('--backend', choices=['demo', 'unet'], default='demo')
    node.add_argument('--checkpoint')
    node.add_argument('--normalization')
    node.add_argument('--device', default='cpu')
    node.add_argument('--demo-delay', type=float, default=0)
    status = commands.add_parser('status')
    status.add_argument('--config', default='config/cluster.local.json')
    for command in ('submit', 'result'):
        p = commands.add_parser(command)
        p.add_argument('--config', default='config/cluster.local.json')
        p.add_argument('--job-id', required=command == 'result')
        p.add_argument('--output', default='output/result')
        p.add_argument('--timeout', type=float, default=120)
        if command == 'submit':
            p.add_argument('--input', required=True, help='NPZ com image [C,H,W] e truth opcional')
            p.add_argument('--tile-size', type=int, default=128)
            p.add_argument('--overlap', type=int, default=32)
            p.add_argument('--pixel-area-m2', type=float, default=0)
    data = commands.add_parser('make-demo')
    data.add_argument('--output', default='data/scene_demo.npz')
    prepare = commands.add_parser('prepare-nc')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'node':
        if args.backend == 'unet':
            if not args.checkpoint or not args.normalization:
                parser.error('U-Net exige --checkpoint e --normalization.')
            predictor = TorchPredictor(args.checkpoint, args.normalization, args.device)
        else:
            predictor = DemoPredictor(args.demo_delay)
        try:
            asyncio.run(serve(args.id, read_config(args.config), args.data_dir or f'data/node-{args.id}', predictor))
        except KeyboardInterrupt:
            pass
    elif args.command == 'make-demo':
        image, truth = synthetic_scene()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, image=image, truth=truth)
        print(args.output)
    elif args.command == 'prepare-nc':
        prepare_nc(args.input, args.output)
        print(args.output)
    else:
        asyncio.run(remote(args))
