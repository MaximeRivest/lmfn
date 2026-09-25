# banking77 distillation — measured results (2026-09-24)

Teacher: DeepSeek V4.1 Flash (low reasoning, Prime Inference), 1,894 labels, $0.65.
Student: Qwen3.5-0.8B, full fine-tune on one RTX 3090 (prime-rl SFT).
Quality: the same 200 random test questions no model trained on.
Speed: batch jobs on local RTX 3090s, vLLM 0.29.

| model / serving | accuracy | agrees w/ teacher | rows/s | 100M rows |
|---|---|---|---|---|
| teacher, DeepSeek V4.1 Flash (thinks) | 82% (answered rows) | — | API | $324/M rows |
| Qwen3.5-0.8B untrained | 23% | 42% | — | — |
| student, long prompt, chat (vLLM defaults) | 72% | 84% | 28 → 86 | 325 h |
| student, long prompt, batched ids | 72.5% | | 103 | 270 h |
| student, short prompt (45 tokens), generating, 1 GPU | 71.5% | | 583 | 48 h |
| **student + 77-way classifier head, 1 GPU** | **80.5%** | **91%** | **708** | **39 h** |
| **same, 2 GPUs (one server each)** | 80.5% | 91% | **1,411** | **20 h** |
| TypeSafe Jev (`jev-latest` = 1.13), zero-shot, one 77-key choice | 78.5% (80.0% on the 175 DeepSeek answered) | 88% | **685** (measured, 0 errors) | **~41 h / $431** |

Classifier head: the student's 248k-word output layer replaced by a 77-way
linear layer on the last prompt token (initialized from the student's own
output rows for each intent's first token), trained on the same teacher labels
(backbone frozen, 30 s), temperature-scaled on 15% held out. Served by vLLM
with `--runner pooling --convert classify`; one forward pass per row, no
generation, probabilities for all 77 intents.

Confidence: top-3 accuracy 92%. Sorting by confidence and keeping the most
confident 80% of rows gives 86.9% accuracy, 50% gives 92% (the rest could go
to the teacher). Still overconfident after temperature scaling (ECE 0.15;
mean confidence 0.95 vs 0.80 accuracy): the calibration set is teacher
labels, which the head fits better than the human labels it is scored on.

Why the head is more accurate than generating: it chooses among exactly the
77 intents with one decision, instead of spelling a name token by token
(17 intents start with the token "card").

Limit: the GPU is saturated on prompt processing (~35k tokens/s per 3090); the
next gain is fewer prompt tokens or a smaller/faster backbone, not the head.

## TypeSafe Jev (2026-09-24)

Zero-shot, no training: one `choice` judgment with the 77 intent names as keys
(lm15 dev `judgments`), the customer message as the state. 200/200 answered,
median latency 0.12 s. Accuracy 78.5%, top-3 91%; most confident 80% of rows
90.6%, 50% 95%. Confidence closer to accuracy than the head's (mean 0.89 vs
0.785, ECE ~0.08 vs 0.15), and it spreads mass over genuinely ambiguous pairs
(`card_arrival` 0.78 / `card_delivery_estimate` 0.19).

Cost: ~1,026 input tokens per row at $0.042 per million (output free) =
$4.3 per million rows.

Throughput, measured (the documented 1,200 requests/minute is not what the
key sees): 10,000 requests at 128 in flight ran at 685 req/s with no error;
at 256 in flight about half returned HTTP 429. So ~685 rows/s per key,
100M rows in ~41 h. Predictions: `jev_test200_predictions.json`.

## Latency (one row, including HTTP)

| | one request at a time | under load |
|---|---|---|
| Jev (API over the internet) | p50 127 ms, p90 164 ms | p50 130 ms, p99 355 ms at 685 rows/s |
| student + head (vLLM on a local 3090) | p50 28 ms, p90 29 ms | p50 111 ms, p99 271 ms at 551 rows/s (one row per request) |
