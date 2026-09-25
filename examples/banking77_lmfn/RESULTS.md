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

## Full distillation from Jev's distributions (2026-09-25)

Base Qwen3.5-0.8B (original weights, not the DeepSeek-distilled student), the
248k-word output layer dropped, a 77-way head on the last prompt token
(initialized from the base model's rows for each intent's first token).
Every weight trained (752M parameters, no freezing, no LoRA). Loss:
KL(Jev || student) on Jev's full 77-way distribution (`jev_label.py`,
`full_distill.py`). AdamW, lr 3e-5 body / 3e-4 head, batch 32, 3 epochs,
cosine; one RTX 3090 each; bf16 autocast over fp32 weights.

Teacher labels: Jev on all 9,993 banking77 train questions in **16.1 s**
(621 rows/s, 128 in flight, 0 errors, 0 retries), ~$0.43. Jev's pick matches
the human label on 76.8% of them.

| student | train rows | accuracy | top-3 | agrees w/ Jev | ECE | mean conf. | 80% most confident | 50% |
|---|---|---|---|---|---|---|---|---|
| Jev itself (teacher), same 200 | — | 78.5% | 91% | — | 0.105 | 0.89 | 90.6% | 95% |
| DeepSeek student + frozen-body head (earlier) | 1,894 | 80.5% | 92% | — | 0.15 | 0.95 | 86.9% | 92% |
| full distill from Jev | 1,894 (DeepSeek's texts) | 77.0% | 91% | 90.5% | **0.061** | 0.83 | 88.1% | 95% |
| **full distill from Jev** | **9,493** | **80.0%** | **94.5%** | **92.5%** | 0.102 | 0.87 | **90.0%** | **96%** |

Held-out KL to Jev (500 train rows never trained on): 0.255 (1,894 rows),
0.105 (9,493 rows). On 200 test questions one question is 0.5 point; the
standard error of an accuracy near 80% is about 2.8 points, so 77-80.5% are
not reliably different; top-3, KL and agreement move more clearly.

Wall time, measured (model already downloaded, Python env installed):

| phase | 1,894 rows | 9,493 rows |
|---|---|---|
| Jev labels (all 9,993 rows, once) | 16 s | 16 s |
| tokenize + load model | 8 s | 9 s |
| train, 3 epochs | 97 s (180 steps) | 312 s (891 steps) |
| of which one-time GPU kernel compile, first epoch | ~38 s | ~36 s |
| evaluate 200 + save bf16 weights | 3 s | 3 s |
| **total, labels to saved model** | **~2 min** | **~5.7 min** |

Peak GPU memory 15.6 / 16.8 GB. Serving speed not measured: same backbone and
prompt as the earlier head (708 rows/s per 3090 in vLLM), but the saved
`model.pt` still has to be exported in vLLM's classify format.
