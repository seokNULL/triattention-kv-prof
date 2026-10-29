# vLLM runner (AIME24 / AIME25 / MATH-500)

`scripts/cli.py run-one`(HF transformers 경로)과 같은 모델·데이터셋·프롬프트·샘플링·seed·채점기를 쓰되,
생성을 **vLLM의 paged KV cache**로 하는 실행기입니다. `--method triattention`이면
`triattention/vllm/`의 TriAttention runtime plugin이 켜지고, `--method fullkv`이면 꺼집니다.

| 파일 | 역할 |
|---|---|
| `quick_test.py` | 짧은 문제 8개로 1~2분 안에 끝나는 동작 확인 (압축 발생 횟수 출력) |
| `run_benchmark.py` | benchmark: 모델 1개 × 데이터셋 1개 × 방법 1개 실행 + 채점 |
| `run_all.sh` | 3개 모델 × 3개 데이터셋 × (fullkv, triattention) 일괄 실행 |
| `env_check.py` | 버전, plugin 등록, 모델/데이터/통계 파일 존재 여부 확인 |
| `kv_dump.py` | quick 문제 1개를 생성하면서 paged KV cache와 block table을 파일로 dump |
| `view_kv_dump.py` | dump를 `layer head token block slot K/V 값...` 행 형태로 출력하거나 CSV/numpy로 저장 |
| `analyze_kv_dump.py` | dump에서 KV 유사도와 (가상의) hole 비율을 계산하는 예시 분석 |
| `data/quick.jsonl` | `quick_test.py`, `kv_dump.py`용 문제 8개 (정답은 정수) |

모델·데이터셋 목록, 기본 budget, 통계 파일 선택, 데이터 경로는 `scripts/cli.py`의 것을 그대로 가져다 씁니다.

## 1. 환경: `kv-prof`를 복제해서 vLLM 설치

TriAttention plugin은 vLLM 내부를 직접 패치하므로 버전이 맞아야 합니다.

- `GPUModelRunner._prepare_inputs(scheduler_output, num_scheduled_tokens)` 패치 → **vLLM ≥ 0.12**
- `BlockTable.compute_slot_mapping(req_indices, positions)` / `commit_slot_mapping` 호출 → **vLLM < 0.19** (0.19에서 API 변경)
- 확인한 버전: **vLLM 0.18.1** (모든 패치 지점이 설치되는 것까지 확인. GPU 실행은 미확인)

vLLM 0.18.1은 torch 2.10, transformers ≥ 4.56을 요구합니다. 이를 `kv-prof`에 설치하면 HF 경로의
r1kv/triattention(transformers 4.55 필요)이 깨지므로, **복제본에 설치**합니다. `kv-prof`는 그대로 남습니다.

```bash
conda create -n kv-prof-vllm --clone kv-prof
conda activate kv-prof-vllm
pip install "vllm==0.18.1"
cd ~/work/triattention-kv-prof
pip install -e . --no-deps          # 이 환경에 TriAttention plugin entry point 등록
pip install -r requirements.txt     # 채점기 의존성 (timeout-decorator, multiprocess 등)
python vllm_runner/env_check.py     # 마지막 줄이 "== result: OK" 이면 준비 완료
```

모델(`experiments/models/`)과 데이터(`data/`)는 저장소 안의 파일이라 두 환경이 같이 씁니다.
아직 없다면 `python scripts/prepare_assets.py`로 받으세요.

## 2. 빠른 동작 확인: `quick_test.py`

짧은 산수·대수 문제 8개를 한 번에 넣어 1~2분 안에 끝납니다. 최대 길이 4096 토큰, budget 512로
생성 도중 압축이 일어나도록 맞춰 두었습니다.

```bash
conda activate kv-prof-vllm
python vllm_runner/quick_test.py --model Qwen3-8B --method triattention
python vllm_runner/quick_test.py --model Qwen3-8B --method fullkv          # 비교용
python vllm_runner/quick_test.py --model Qwen3-8B --budget 256 --pruning-mode per_layer_per_head
```

마지막에 문제별 정답/예측/토큰 수 표와 요약이 출력됩니다.

```
 #    gold        pred  ok  tokens  finish
 0     391         391   O    1834  stop
 ...
{ "accuracy": 0.875, "output_tokens": ..., "compression_events": 23, "reclaimed_blocks": 184, ... }
```

- `compression_events`: TriAttention이 실제로 KV를 압축한 횟수. triattention인데 0이면 압축이 안 된 것입니다
  (출력이 budget + divide_length보다 짧았거나 plugin이 꺼져 있음).
- `reclaimed_blocks`: 압축 후 반환된 KV block 수.
- 결과: `experiments/vllm_outputs/quick/<model>/<method>/` (`results.jsonl`, `summary.json`, `triattention.log`)
- 옵션: `--budget`, `--max-length`, `--pruning-mode`, `--divide-length`, `--window-size`,
  `--protect-prefill`/`--no-protect-prefill`, `--block-size`, `--gpu-memory-utilization`

## 3. KV cache dump와 분석: `kv_dump.py`, `analyze_kv_dump.py`

fullkv에는 eviction이 없으므로 hole이 생기지 않습니다. 그래서 (1) 실제 paged KV를 block 구조 그대로 dump하고,
(2) 그 위에서 유사도를 측정하고, (3) 원하는 accumulate/evict 정책을 offline으로 적용해 hole을 시뮬레이션합니다.

```bash
conda activate kv-prof-vllm
python vllm_runner/kv_dump.py --model Qwen3-8B --problem 0                    # fullkv, 전체 layer
python vllm_runner/kv_dump.py --model Qwen3-8B --problem 0 --layers 0,18,35 --snapshot-every 1024
python vllm_runner/analyze_kv_dump.py experiments/kv_dumps/Qwen3-8B/fullkv/problem0 --plot 18:0
```

`kv_dump.py`는 요청 1개만 생성합니다(끝난 뒤 그 block을 다른 요청이 덮어쓰지 않도록). 엔진을 같은 프로세스에서 띄우고
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) model runner의 `execute_model`을 감싸서 매 step의 KV 길이와 block table을 기록한 뒤,
생성이 끝나면 그 요청의 K/V를 paged cache에서 꺼냅니다.

| 파일 | 내용 |
|---|---|
| `kv.pt` | `key`, `value`: `[layer, token, kv_head, head_dim]` bf16, **논리 토큰 순서** (key는 RoPE 적용 후). `block_ids`, `block_size`: 토큰 t는 물리 block `block_ids[t // block_size]`의 slot `t % block_size`에 있었음 |
| `meta.json` | 모델 형태, prompt/출력 길이, KV 텐서 layout, 생성 결과 |
| `tokens.json` | KV 위치별 token id와 token 문자열 (`kv_token_ids[t]`가 KV 위치 t의 토큰) |
| `trace.jsonl` | step별 KV 길이와 block table (바뀔 때만 기록) |
| `snapshots/` | `--snapshot-every N`일 때 생성 도중 N 토큰마다의 dump |

크기: Qwen3-8B 기준 토큰당 36 layer × 8 head × 128 × 2(K,V) × 2B = 144KB, 4096 토큰이면 약 590MB입니다.
`--layers`로 줄일 수 있습니다.

```python
import torch
d = torch.load("experiments/kv_dumps/Qwen3-8B/fullkv/problem0/kv.pt")
k = d["key"][li]            # layer d["layers"][li]: [T, kv_heads, head_dim]
block_of_token = [d["block_ids"][t // d["block_size"]] for t in range(d["kv_len"])]
```

dump 내용을 행 단위 배열로 보기: `view_kv_dump.py`

```bash
D=experiments/kv_dumps/Qwen3-8B/fullkv/problem0
python vllm_runner/view_kv_dump.py $D --layer 18 --head 0 --tokens 0:8          # 터미널 표 (앞 8개 dim)
python vllm_runner/view_kv_dump.py $D --layer 0,18 --kv K --dims 0 --tokens 100:104   # 128개 dim 전부
python vllm_runner/view_kv_dump.py $D --layer 18 --tokens all --csv kv_l18.csv  # CSV (모든 dim)
python vllm_runner/view_kv_dump.py $D --layer 18 --npz kv_l18.npz               # numpy float32 배열
```

```
layer head token block slot kv piece                d0        d1  ...
   18    0    15   211   15  K Ġthe            0.0123   -1.2344  ...
   18    0    16   487    0  K Ġsum            ...
```

- `--layer`, `--head`, `--tokens`: `18`, `0,18,35`, `0:16`, `all` 형식
- `--kv K|V|KV`, `--dims N`(0이면 전부), `--precision`
- CSV 열: `layer, head, token, block, slot, kv, piece, d0 … d127` (선택한 행의 모든 dim)
- npz: `key`, `value` = `[layer, token, head, dim]` float32와 `layers`, `heads`, `tokens`, `block`, `slot`

`analyze_kv_dump.py`는 출발점이 되는 예시 분석입니다. layer별로 다음을 출력합니다.

- `adjK`/`adjV`: 인접 토큰의 K/V cosine
- `blockK`/`randK`: 같은 vLLM block 안의 토큰 쌍 vs 무작위 토큰 쌍의 K cosine
- `bestK`/`V@bestK`: 생성된 각 토큰이 앞선 토큰들 중 가장 비슷한 것과의 K cosine, 그 토큰과의 V cosine
  (append 시점에 accumulate/append/evict를 판단할 때 보게 되는 값)
- 가상의 hole: `bestK ≥ τ` 이고 `V@bestK ≥ τ`인 토큰을 합쳐 없앤다고 할 때
  - `head-slot`: (토큰, kv_head) 단위로 빈 칸 비율 (head별 eviction)
  - `token-slot`: 한 layer에서 모든 kv_head가 비어 slot 전체가 빈 비율 (vLLM slot 단위)
  - `free blocks`: block의 모든 slot이 비어 block을 통째로 반환할 수 있는 수

비교 대상을 "앞선 모든 토큰"으로 잡아서, 실제 순차 정책이 없앨 수 있는 양의 상한입니다.
`--method triattention`으로 dump하면 마지막 압축 이후의 압축된 cache가 저장됩니다(위치가 연속이 아님).

## 4. benchmark 실행: `run_benchmark.py`

```bash
conda activate kv-prof-vllm

# 짧게 확인 (3문제, 샘플 1개)
python vllm_runner/run_benchmark.py --model Qwen3-8B --dataset aime24 --method fullkv \
    --num-samples 1 --max-examples 3
python vllm_runner/run_benchmark.py --model Qwen3-8B --dataset aime24 --method triattention \
    --budget 2048 --num-samples 1 --max-examples 3

# 전체: 3 모델 × 3 데이터셋 × 2 방법 (추가 인자는 모두 run_benchmark.py로 전달)
bash vllm_runner/run_all.sh
bash vllm_runner/run_all.sh --num-samples 1 --max-examples 3
MODELS="Qwen3-8B" DATASETS="aime24 math500" METHODS=triattention \
    bash vllm_runner/run_all.sh --budget 1024
```

실행 중에는 vLLM 진행 막대, 주기적인 KV cache 사용률 로그, TriAttention 압축 로그
(`TriAttention compression applied ... before=... after=... reclaimed_blocks=...`), chunk별 처리량이 출력됩니다.

TriAttention plugin은 `triattention.*` 이름의 logger를 쓰는데, vLLM은 `vllm` logger에만 출력 handler를 붙입니다.
그래서 그대로 두면 plugin 로그가 **보이지 않습니다**. 두 스크립트는 실행 폴더에 `vllm_logging.json`을 만들고
`VLLM_LOGGING_CONFIG_PATH`로 지정해, plugin 로그를 콘솔과 `triattention.log`에 남깁니다
(직접 `VLLM_LOGGING_CONFIG_PATH`를 지정하면 그 설정을 그대로 씁니다).

## 5. benchmark 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--budget` | Qwen3/DS-Qwen 2048, DS-Llama 512 | KV budget (triattention) |
| `--num-samples` | aime 8, math500 1 | 문제당 샘플 수 (pass@1 평균) |
| `--max-examples` | 전체 | 앞에서부터 N문제 |
| `--max-length` | aime 32768, math500 8192 | 프롬프트 + 생성 토큰 수 |
| `--pruning-mode` | `per_head` | `per_head`, `per_layer_per_head`, `per_layer` |
| `--protect-prefill` / `--no-protect-prefill` | 보호 | 프롬프트 토큰 eviction 금지 (HF 경로 기본과 동일) |
| `--divide-length` | 128 | 몇 토큰마다 압축할지 |
| `--window-size` | 128 | 항상 남기는 최근 토큰 수 |
| `--stats-path` | 저장소 calibration 파일 | 통계 파일 직접 지정 |
| `--temperature`, `--top-p`, `--seed` | 0.6, 0.95, 888 | `runner_defaults.yaml` 값 |
| `--top-k` | 모델 `generation_config`의 값, 없으면 50 | HF `generate()`와 같게 맞춤 |
| `--block-size` | 16 | vLLM KV block 크기 (hole 연구용) |
| `--gpu-memory-utilization` | 0.9 | |
| `--tensor-parallel-size` | 1 | |
| `--max-num-seqs`, `--max-num-batched-tokens` | vLLM 기본 | 동시 요청 수, prefill chunk 크기 |
| `--chunk-size` | 64 | `generate()` 한 번에 넣는 요청 수. 끝난 chunk는 저장되어 재실행 시 이어서 함 |
| `--run-tag` | 없음 | 결과 폴더 이름 뒤에 붙는 꼬리표 |
| `--skip-eval` | 채점함 | 생성만 하고 채점 생략 |

엔진은 두 방법 모두 `enforce_eager=True`, `enable_prefix_caching=False`로 돕니다
(TriAttention 필수 조건이며, fullkv도 같은 조건으로 맞춰 공정하게 비교).

## 6. benchmark 결과

```
experiments/vllm_outputs/<dataset>/<model>/sample<N>/<method>/<tag>/
├── run_config.json                 # 실제로 쓴 설정 전체
├── generations/generations.jsonl   # (문제, 샘플)마다 1줄: output, 토큰 수, finish_reason ...
├── throughput.json                 # 출력 토큰 수, 생성 시간, tok/s, 압축 횟수, 반환 block 수
├── triattention.log                # plugin 로그 (압축 이벤트)
└── eval/<dataset>/..._metrics.json # 채점 결과 (acc = pass@1)
```

`<tag>`는 `full` 또는 `budget_<B>`이고, `per_head`가 아닌 pruning mode와 `--run-tag`가 뒤에 붙습니다.
같은 설정으로 다시 실행하면 이미 생성된 (문제, 샘플)은 건너뜁니다.

## 7. HF 경로와의 차이

- 프롬프트 토큰은 `scripts/worker.py`와 똑같이 HF tokenizer로 만들어 넘깁니다. 샘플별 seed 공식도 같습니다.
  다만 vLLM과 HF는 sampling 커널이 달라 같은 seed라도 생성 결과는 같지 않습니다.
- vLLM plugin에는 r1kv/snapkv 구현이 없어 `fullkv`, `triattention`만 지원합니다.
- `round_window` 같은 HF 전용 TriAttention 옵션은 vLLM runtime에 없습니다.
