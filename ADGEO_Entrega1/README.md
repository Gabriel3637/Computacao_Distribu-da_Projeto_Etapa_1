# ADGEO - Entrega 1

Versão independente da primeira etapa: inferência de tiles por **gRPC** e
**eleição de líder Chang-Roberts em anel**, implementada manualmente.

O relatório desta etapa está em `output/pdf/ADGEO_Entrega1.pdf` e tem
fonte LaTeX em `output/pdf/artigo_sbc`. As evidências são exclusivas desta versão,
em `output/tests/`.

## Instalar e executar tudo

Abra o PowerShell dentro desta pasta:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.demo --output output/tests/demo
```

A demonstração inicia três processos reais. O líder 3 segmenta uma cena em
42 tiles, é encerrado e o líder 2 é eleito. Um **novo job** é submetido ao eleito
e sua máscara é comparada com a referência local. O teste também confirma que
o novo líder não recebeu o histórico do job anterior.

Os jobs ficam em memória no líder atual. Se ele cair durante um trabalho, o
cliente deverá reenviar a entrada ao novo líder. A retomada automática do mesmo
job, a replicação e a persistência de jobs estão na entrega 2. O contador da
eleição e o cache local de inferência continuam existindo nesta versão.

## Execução manual

Em três terminais separados, nesta pasta:

```powershell
.\.venv\Scripts\python.exe -m adgeo node --id 1
```

```powershell
.\.venv\Scripts\python.exe -m adgeo node --id 2
```

```powershell
.\.venv\Scripts\python.exe -m adgeo node --id 3
```

Em um quarto terminal:

```powershell
.\.venv\Scripts\python.exe -m adgeo status
.\.venv\Scripts\python.exe -m adgeo make-demo
.\.venv\Scripts\python.exe -m adgeo submit --input data/scene_demo.npz --job-id cena-001 --output output/cena-001
```

Saídas: `mask.npy`, `mask.png`, `metrics.json` e `job_id.txt`. Azul indica classe
geleira, preto fundo e cinza pixels ignorados. O backend `demo` usa uma regra
sintética para testar o sistema; não identifica geleiras de verdade.

## Duas ou mais máquinas

Copie o projeto e instale-o em cada máquina. Copie a configuração local para
`config/cluster.lan.json`; troque os IPs locais pelos IPs reais e mantenha a mesma
lista em todas as máquinas. Inicie somente o ID correspondente a cada máquina,
usando `--config config/cluster.lan.json`. Permita as portas TCP escolhidas na
rede privada. O cliente também deve usar essa configuração.

Os membros são estáticos e seus IDs são únicos. O maior ID ativo vence a eleição.
Cada nó tenta os próximos sucessores configurados se um deles não responder.
Heartbeats e timeouts detectam suspeitas; não oferecem consenso sob partições.
Os RPCs usam rede confiável, sem TLS ou autenticação neste protótipo.

## U-Net e NetCDF

O cluster realiza **inferência**, usando o mesmo modelo em todos os workers.
O treinamento permanece no notebook original. Para instalar o suporte opcional:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install xarray netCDF4
.\.venv\Scripts\python.exe -m pytest -q --junitxml=output/tests/pytest.xml
```

Para CUDA, instale a distribuição do PyTorch apropriada ao equipamento.
Baixe `unet_melhor.pth` e `normalizacao.npz` da mesma execução do notebook e
coloque em `models/` em todas as máquinas. Eles não foram fornecidos nos anexos.

```powershell
.\.venv\Scripts\python.exe -m adgeo node --id 1 --backend unet --checkpoint models/unet_melhor.pth --normalization models/normalizacao.npz --device cpu
.\.venv\Scripts\python.exe -m adgeo prepare-nc --input dados/um_patch.nc --output data/um_patch.npz
.\.venv\Scripts\python.exe -m adgeo submit --input data/um_patch.npz --job-id patch-001 --output output/patch-001
```

Use `--device cuda` para GPU. Ordem das 13 bandas:
`B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12`. O adaptador mantém as classes
do notebook, usa `eval()`/`inference_mode()`, normaliza por banda e compara o
hash dos pesos e da normalização. Os testes opcionais usam pesos artificiais;
isso não constitui avaliação científica com os pesos treinados.

## Tiles, máscaras e métricas

Entrada NPZ: `image` float32 `[C,H,W]` e `truth` opcional `[H,W]` com 0,1,255.
O padrão é tile 128, overlap 32; ambos podem ser alterados no cliente. A costura
calcula a média das probabilidades e aplica o limiar 0,5 depois. Cada pixel da
máscara final entra uma vez em TP/FP/FN e nos cálculos de IoU e Dice.
Denominador zero produz `null`; rótulos 255 e pixels inválidos são ignorados.
Arquivos NetCDF diferentes não são costurados sem coordenadas/CRS compatíveis.

Informe `--pixel-area-m2` apenas quando souber a área constante do pixel em m².
Sem esse dado, o sistema informa pixels, sem inventar áreas geográficas.
Os RPCs unários limitam cada mensagem a 64 MiB, arrays a 48 MiB e jobs a 10.000 tiles.

## Contrato e evidências

RPCs desta etapa: `Heartbeat`, `Election`, `Coordinator`, `InferTile`, `SubmitJob`
e `GetJob`. Não há `ReplicateJob` no contrato desta versão.

```powershell
.\.venv\Scripts\python.exe -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. adgeo/proto/cluster.proto
```

`output/tests/pytest.xml` contém os resultados da suíte. `output/tests/demo/summary.json`
registra eleição e máscaras; `demo/peers/node-N.log` contém os logs reais.
Sem instalar PyTorch/xarray, os testes opcionais aparecem como ignorados.
Os testes registrados foram executados em processos distintos na mesma máquina.

O artigo foi revisado tecnicamente. As responsabilidades seguem a divisão de frentes informada pelo grupo, com
teoria e implementação compartilhadas por dupla. O grupo deve revisar a redação
e assegurar sua autoria, conforme o enunciado.
