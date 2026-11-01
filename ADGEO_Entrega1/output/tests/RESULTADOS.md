# Entrega 1: resultados executados

**22 testes aprovados**, sem falhas, erros ou ignorados. Duração da suíte: 12,67 s.

Execução em processos distintos na mesma máquina. A U-Net foi testada em CPU.
Imagens e pesos artificiais validam a engenharia, sem medir qualidade com geleiras reais.

## Testes

- `test_wait_retries_discovery_timeout_then_returns_job`: aprovado.
- `test_wait_stops_at_global_deadline_when_discovery_keeps_failing`: aprovado.
- `test_rpc_election_and_new_job_without_replication`: aprovado.
- `test_parallel_segmentation_with_at_least_two_nodes[2]`: aprovado.
- `test_parallel_segmentation_with_at_least_two_nodes[3]`: aprovado.
- `test_validation_and_idempotent_submission_in_current_leader`: aprovado.
- `test_nc_band_order_and_invalid_pixels`: aprovado.
- `test_cleanup_removes_long_directory_and_file`: aprovado.
- `test_cleanup_rejects_directories_not_created_by_this_instance`: aprovado.
- `test_rejoining_node_starts_above_remote_round`: aprovado.
- `test_rejected_election_does_not_elect_sender_as_single_survivor`: aprovado.
- `test_only_unreachable_peers_allow_single_survivor_election`: aprovado.
- `test_split_join_unique_pixels[shape0-32-0]`: aprovado.
- `test_split_join_unique_pixels[shape1-64-17]`: aprovado.
- `test_split_join_unique_pixels[shape2-128-64]`: aprovado.
- `test_split_join_unique_pixels[shape3-32-0]`: aprovado.
- `test_metrics_global_not_mean_per_tile`: aprovado.
- `test_overlap_averages_probabilities_before_threshold`: aprovado.
- `test_npy_round_trip_and_reject_pickle`: aprovado.
- `test_validation_and_missing_tiles`: aprovado.
- `test_notebook_model_adapter_and_normalization`: aprovado.
- `test_unet_over_real_grpc_matches_local_tiled_inference`: aprovado.

## Demonstração da primeira etapa

- 42 tiles, máscaras iguais à referência local antes e depois da eleição.
- Líder inicial 3, líder final 2; convergência: 2,23 s.
- O job anterior não foi replicado: o eleito processou um novo job.
- Esta versão não recupera automaticamente o estado do líder morto.

Logs, configurações e máscaras estão nas subpastas de `output/tests/demo`.
O arquivo `pytest.xml` documenta a suíte. Os resultados do benchmark estão em `benchmark_unet/benchmark.json` na entrega 2.
