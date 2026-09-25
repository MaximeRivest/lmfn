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

## ModernBERT students from the same Jev distributions (2026-09-25)

`modernbert_distill.py`: ModernBERT-base (150M) and -large (396M), whole model
trained, 77-way classification head, input = the message alone. Same Jev
labels, same 9,493 train / 500 validation rows, same KL loss and the same 200
test questions as the Qwen3.5-0.8B full distillation above. AdamW, lr 5e-5
(base) / 3e-5 (large), batch 32, cosine, flash-attention 2, bf16 autocast.
One RTX 3090 each.

| student | params | passes | accuracy | top-3 | agrees w/ Jev | KL to Jev (test) | ECE | 80% most conf. | 50% | labels→saved model |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B, full (above) | 752M | 3 | 80.0% | 94.5% | 92.5% | 0.081 | 0.102 | 90.0% | 96% | 324 s |
| **ModernBERT-base** | 150M | 3 | 79.0% | 93.5% | 92.5% | 0.123 | **0.076** | 88.7% | 97% | **81 s** |
| ModernBERT-base | 150M | 6 | 78.5% | 93.5% | 92.0% | 0.095 | 0.105 | 88.7% | 97% | 159 s |
| ModernBERT-large | 396M | 3 | 77.5% | 94.0% | 92.0% | 0.090 | 0.120 | 88.7% | 97% | 155 s |
| ModernBERT-large | 396M | 6 | 78.5% | 93.5% | **95.5%** | **0.066** | 0.098 | 89.4% | 96% | 308 s |

(labels→saved model excludes the 16 s of Jev labelling, shared by all.)
Validation KL flattens between passes 5 and 6; more passes bring the student
closer to Jev without raising accuracy against human labels. All accuracies
are within noise of each other on 200 questions (±2.8 points).

Serving, plain PyTorch bf16, one 3090, 19,986 real messages sorted by length:

| | rows/s | 100M rows, one GPU | one row, p50 |
|---|---|---|---|
| Qwen3.5-0.8B head, **vLLM** classify (earlier) | 708 | 39 h | 28 ms (incl. HTTP) |
| **ModernBERT-base**, plain PyTorch | **6,508** | **4.3 h** | 13.7 ms (no HTTP) |
| ModernBERT-large, plain PyTorch | 3,031 | 9.2 h | 16.8 ms (no HTTP) |

The comparison favours Qwen's serving (optimized vLLM vs a plain PyTorch
loop); the latencies are not like for like (one includes an HTTP round trip).
Peak training memory: 4.2 GB (base), 10.3 GB (large), 16.8 GB (Qwen).

## Tiny encoders: Ettin 17M / 32M / 68M (2026-09-25)

`jhu-clsp/ettin-encoder-*` (ModernBERT architecture and recipe, small sizes),
same script (`modernbert_distill.py <model id> 6`), lr 1e-4, 6 passes, same
Jev labels, split and 200 test questions.

| student | params | accuracy | top-3 | agrees w/ Jev | KL to Jev | ECE | 50% most conf. | labels→saved model | train memory | rows/s, one 3090 | 100M rows | one row p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ettin-17M | 17M | 77.5% | 92.5% | 92.0% | 0.146 | 0.088 | 96% | **46 s** | 0.5 GB | **42,664** | **0.7 h** | 5.3 ms |
| Ettin-32M | 32M | 77.5% | 94.0% | 91.0% | 0.114 | 0.084 | 96% | 63 s | 0.9 GB | 25,955 | 1.1 h | 6.9 ms |
| Ettin-68M | 68M | 78.5% | 93.0% | 93.0% | 0.104 | 0.094 | 96% | 113 s | 2.1 GB | 11,885 | 2.3 h | 12.3 ms |
| ModernBERT-base (above) | 150M | 79.0% | 93.5% | 92.5% | 0.123 | 0.076 | 97% | 81 s (3 passes) | 4.2 GB | 6,508 | 4.3 h | 13.7 ms |
| Qwen3.5-0.8B (above, vLLM) | 752M | 80.0% | 94.5% | 92.5% | 0.081 | 0.102 | 96% | 324 s | 16.8 GB | 708 | 39 h | 28 ms (HTTP) |

rows/s is the model alone on pre-tokenized input (same harness for every
encoder). One Python thread tokenizes ~22,000 rows/s, so at 17M the tokenizer,
not the GPU, is the limit unless tokenization is parallelized across cores.
Accuracy differences stay within the ±2.8-point noise of 200 questions; KL to
Jev shows the bigger models follow the teacher's distribution more closely.

## BERT-tiny, 4.4M parameters (2026-09-25)

`google/bert_uncased_L-2_H-128_A-2` (2 layers, width 128; Google, 2020), same
script, labels, split and test.

| setting | accuracy | top-3 | agrees w/ Jev | KL to Jev | ECE | 80% / 50% most conf. | labels→saved model | rows/s (model only) | one row p50 |
|---|---|---|---|---|---|---|---|---|---|
| lr 3e-4, 10 passes | 71.5% | 89.5% | 83.5% | 0.416 | 0.102 | 81.9% / 93% | 31 s | 72,899 | 1.5 ms |
| **lr 1e-3, 30 passes** | **75.0%** | **93.0%** | **89.0%** | **0.248** | 0.123 | 87.5% / 96% | 84 s | **81,757** | 1.3 ms |
| Ettin-17M (above) | 77.5% | 92.5% | 92.0% | 0.146 | 0.088 | — / 96% | 46 s | 42,664 | 5.3 ms |

Validation KL flattens at ~0.27 from pass ~17 on (lr 1e-3): the model is at
its capacity, not under-trained. Peak training memory 0.1 GB. At these speeds
single-thread tokenization (~21,000 rows/s) is the bottleneck, so real
end-to-end throughput needs tokenization spread over several cores.

## OpenRouter token logprobs as 77-way soft targets (2026-09-25)

`or_logprob_targets.py`: the model answers with one intent name (temperature
0, no reasoning); the class distribution is rebuilt from the top-k
alternatives along the answered path (intent names span 2-6 tokens). Same 200
test questions; compared with the human labels and with Jev.

What each model offers on OpenRouter (checked per provider, 2026-09-25):

| model | logprobs | reasoning off? | alternatives per token | usable |
|---|---|---|---|---|
| xiaomi/mimo-v2.6-pro | no provider returns them | — | — | no |
| qwen/qwen3.8-max-0902 | Alibaba only | no: mandatory (121 reasoning tokens even at "minimal"); answer then 1.000 certain | 5 | no |
| z-ai/glm-5.3 | ~20 providers | "minimal" gives 0 reasoning on most rows, but 22/200 rows still reasoned, 4 answered nothing | 20 (Together); 5 (Fireworks); **Modal and Parasail return wrong data** (the same alternatives repeated at every position) | poorly |
| **moonshotai/kimi-k3** | 9 providers | **yes** (`reasoning.enabled=false`), 0 reasoning tokens on all 200 | 20 (Parasail) | **yes** |

| teacher (200 test questions) | accuracy | top-3 | agrees w/ Jev | KL(Jev‖it) | mean conf. | ECE | rows with spread (<0.95) | coverage (mean / min) | cost per 1,000 rows | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev (for reference) | 78.5% | 91% | — | 0 | 0.89 | 0.105 | — | 1.0 | $0.04 | 0.12 s |
| **Kimi K3 (Parasail)** | **82.0%** | 90.5% | 87.5% | 0.41 | 0.91 | 0.093 | 33% | **0.976** / 0.657 | **$1.54** | 0.48 s |
| GLM-5.3 (Together) | 71.0% | 79.5% | 78.0% | 1.52 | 0.88 | 0.179 | 26% | 0.918 / 0.0 | $0.20 | 0.26 s |

Coverage = share of probability that lands on a single intent; the rest is
formatting/prose tokens ("lost") or top-20 truncation. Top-3 is capped by the
top-20 alternatives seen along one path, so tails beyond them are zero.
DeepSeek V4.1 Flash (earlier hard labels) scored 82.9% on the 175 questions it
answered; Kimi answered all 200.

## Claude Opus 5.5, hard labels (2026-09-25)

`or_hard_labels.py anthropic/claude-opus-5.5 low`: same instruction and 200
test questions, answer constrained to the 77 intents by a JSON-schema enum.
Reasoning cannot be disabled on OpenRouter for this model ("mandatory");
at effort "low" it used 2 reasoning tokens per question on average.

| | accuracy | agrees w/ Jev | median latency | cost per 1,000 rows |
|---|---|---|---|---|
| **Claude Opus 5.5 (low)** | **92.0%** | 84.0% | 2.4 s | $8.12 |
| Kimi K3 (logprobs run) | 82.0% | 87.5% | 0.48 s | $1.54 |
| Jev | 78.5% | — | 0.12 s | $0.04 |

200/200 valid answers. Standard error near 92% is ±1.9 points, so the gap
to Kimi (10 points) is real. banking77 is public (2020), so part of this
may be memorized test data; a fresh held-out set would settle it.

## Soft (Jev's distribution) vs hard (Jev's top pick) targets (2026-09-25)

`soft_vs_hard.py`: one change only, the target: Jev's 77-way distribution or
a one-hot on its top choice. Same encoder, data, split, loss, passes; 3 seeds
each. Evaluated on the FULL banking77 test set (3,076 questions; Jev labelled
it in 5 s, 79.4% vs human). "Calibrated" = one temperature fitted on the 500
validation rows' human labels, applied to both alike. Mean ± std over seeds.

| student / data | target | accuracy | top-3 | agrees w/ Jev | ECE raw | ECE calibrated | mean conf. | 50% most conf. | 80% most conf. | NLL (calibrated) |
|---|---|---|---|---|---|---|---|---|---|---|
| ModernBERT-base / 9,493 | soft | 78.8 ±0.1 | **91.7** | 91.6 | **0.067** | **0.016** | 0.85 | **96.3** | **87.8** | **0.80** |
| | hard | 78.6 ±0.3 | 90.6 | 91.0 | 0.152 | 0.036 | 0.94 | 92.8 | 86.8 | 0.95 |
| Ettin-17M / 9,493 | soft | 77.3 ±0.3 | **91.4** | 89.8 | **0.077** | **0.019** | 0.85 | **95.9** | **86.6** | **0.83** |
| | hard | 77.2 ±0.1 | 88.4 | 88.8 | 0.183 | 0.039 | 0.95 | 91.1 | 85.1 | 1.04 |
| Ettin-17M / 1,894 | soft | **67.5** ±0.5 | **84.8** | 76.2 | **0.060** | **0.022** | 0.73 | **89.3** | **76.5** | **1.24** |
| | hard | 65.3 ±0.7 | 80.3 | 73.6 | 0.198 | 0.036 | 0.85 | 85.2 | 74.2 | 1.52 |

Top-1 accuracy: no difference with ~9.5k rows (+0.2 points, within seed and
sampling noise); +2.2 points with 1.9k rows. Everything about the rest of
the distribution improves clearly and at every size: top-3 +1 to +4.5
points, calibration error halved raw and still halved after temperature
scaling, and the most-confident-half accuracy +3.5 to +4.8 points (the
"send unsure rows to a bigger model" use). Training time identical.

## No teacher: the dataset's own human labels (2026-09-25)

`soft_vs_hard.py MODEL ... human`: same students, rows, split, settings and
full 3,076-question test set as the soft/hard comparison; the target is the
banking77 human label (one-hot). 3 seeds.

| student / data | target | accuracy | top-3 | agrees w/ Jev | ECE raw | ECE calibrated | 50% most conf. | 80% most conf. | NLL (cal.) |
|---|---|---|---|---|---|---|---|---|---|
| ModernBERT-base / 9,493 | **human** | **93.2** ±0.3 | **98.0** | 79.7 | **0.017** | **0.011** | **99.5** | **98.9** | **0.26** |
| | Jev soft | 78.8 | 91.7 | 91.6 | 0.067 | 0.016 | 96.3 | 87.8 | 0.80 |
| Ettin-17M / 9,493 | **human** | **91.5** ±0.2 | **97.1** | 78.0 | 0.047 | **0.011** | **99.7** | **98.4** | **0.33** |
| | Jev soft | 77.3 | 91.4 | 89.8 | 0.077 | 0.019 | 95.9 | 86.6 | 0.83 |
| Ettin-17M / 1,894 | **human** | **76.0** ±1.1 | **89.3** | 66.9 | 0.092 | 0.020 | 96.2 | 86.3 | 0.92 |
| | Jev soft | 67.5 | 84.8 | 76.2 | 0.060 | 0.022 | 89.3 | 76.5 | 1.24 |

The test labels come from the same annotators and guidelines as the training
labels, so a human-trained student also learns their conventions on
ambiguous pairs (e.g. card_arrival vs card_delivery_estimate); teachers are
scored against conventions they never saw. Accuracy against these labels
therefore measures fit to this labelling scheme, not only understanding.
For reference, Opus 5.5 scored 92.0% on the 200-question subset (±1.9).

## Training Ettin-17M on a laptop CPU (2026-09-25)

`cpu_train.py`: the 91.5% recipe (human labels, 9,493 rows, 6 passes, batch
32, AdamW 1e-4, warmup + cosine, clip 1.0) on XPSwhite: Intel i7-1065G7
(4 cores / 8 threads, AVX-512 + VNNI, no bf16 hardware), 31 GB RAM,
PyTorch 2.13 CPU, while the desktop was in use. The Iris Plus G7 GPU (Gen11)
is not supported by PyTorch XPU and had no compute driver installed.

Settings, 40 timed steps each:

| setting | s/step | projected training |
|---|---|---|
| 4 threads, fp32, random batches | 0.64 | 19 min |
| 8 threads (hyperthreads) | 1.64 | 49 min |
| 4 threads, bf16 autocast (emulated) | 1.40 | 42 min |
| **4 threads, fp32, length-grouped batches** | **0.31** | **9 min** |
| 3 threads, grouped | 0.30 | 9 min |

Full run (4 threads, grouped, seed 0): training 674 s (passes 101 → 122 s
as the laptop heated), 688 s wall in total. **Accuracy 91.0%**, top-3 97.0%
on all 3,076 test questions (GPU, same recipe, 3 seeds: 91.3-91.6%).
Inference on the laptop: 14 ms per message, ~400 messages/s in batches.
The random-batch (exact-order) run was stopped after 4 of 6 passes
(~250-285 s per pass).

## OpenRouter models that allow distillation (2026-09-25)

OpenRouter's own flag: with `provider.enforce_distillable_text = true` a
request only reaches models whose authors allow training on their output.
Probing all 368 text models: **112 allowed** (Qwen incl. the closed
Qwen3.8 Max, DeepSeek, Kimi, NVIDIA Nemotron, Mistral, Meta Llama,
Microsoft Phi-4); 248 refused, including every Claude, GPT, Gemini, GLM,
MiMo, gemma and gpt-oss model, whatever their weights' license.

`or_compare.py`: 20 allowed models, same 200 test questions and
instruction, answer constrained to the 77 intents (JSON-schema enum), the
distillation flag on every request, reasoning off (or minimal where it is
mandatory). Accuracy ±2.8 points (1 s.e.).

| model | accuracy | reasoning used | median latency | cost / 1,000 rows |
|---|---|---|---|---|
| qwen/qwen3.8-2.4t-a95b | 83.5% | 117 tokens | 2.15 s | $0.937 |
| moonshotai/kimi-k3 | 83.5% | none | 1.1 s | $3.145 |
| nvidia/nemotron-3-ultra-550b-a55b | 81.5% | none | 1.18 s | $0.106 |
| qwen/qwen3.8-max-prime | 81.0% | 86 tokens | 3.31 s | $3.015 |
| mistralai/mistral-small-2603 | 81.0% * | none | 0.63 s | $0.021 |
| deepseek/deepseek-v4-pro-0813 | 80.5% | none | 0.67 s | $0.270 |
| qwen/qwen3.8-flash | 79.5% | none | 1.24 s | $0.036 |
| moonshotai/kimi-k2.6 | 78.5% | none | 0.62 s | $0.138 |
| qwen/qwen3.8-max-0902 | 78.5% | 93 tokens | 4.84 s | $1.550 |
| qwen/qwen3.8-omni-flash | 78.0% | none | 1.47 s | $0.148 |
| deepseek/deepseek-v4-flash-0731 | 78.0% | none | 0.93 s | $0.026 |
| deepseek/deepseek-v4.1-flash | 77.5% | none | 0.34 s | $0.057 |
| mistralai/mistral-medium-3-5 | 77.5% | none | 0.58 s | $0.825 |
| qwen/qwen3.8-27b | 77.0% | none | 0.5 s | $0.097 |
| meta-llama/llama-4-maverick | 77.0% | none | 0.63 s | $0.131 |
| meta-llama/llama-3.3-70b-instruct | 76.0% | none | 0.42 s | $0.330 |
| qwen/qwen3.5-9b | 73.5% | none | 0.71 s | $0.046 |
| nvidia/nemotron-3-super-120b-a12b | 73.0% | none | 1.15 s | $0.046 |
| nvidia/nemotron-3.5-lightning | 69.0% | none | 2.92 s | $0.061 |
| microsoft/phi-4 | 65.5% | none | 0.53 s | $0.032 |

\* answered 116 of 200 (rate-limited); accuracy on those.
Reference, not allowed: Claude Opus 5.5 92.0%. Students on human labels
(full test set): ModernBERT-base 93.2%, Ettin-17M 91.5%.

## Majority vote of the distillable models (2026-09-25)

Same 200 answers as above, no new calls; ties go to the most accurate model.

| ensemble | accuracy |
|---|---|
| best single (Kimi K3 / Qwen3.8 2.4T) | 83.5% |
| vote of six (Kimi K3, Qwen3.8 2.4T, DeepSeek V4 Pro, Kimi K2.6, DeepSeek V4.1 Flash, Nemotron 3 Ultra) | 83.0% |
| vote of top 3 / top 4 + Qwen3.8 Max prime / all seven | 83.0% / 83.0% / 82.5% |
| best 5-model subset, chosen on these same questions (optimistic) | 84.5% |
| "oracle": at least one of the six is right | 88.0% |

Vote minus Kimi K3: -0.5 points, 95% bootstrap interval [-2.0, +1.0] (better
on 1 question, worse on 2): no gain. The models make the same mistakes.
Agreement is a strong confidence signal instead: all six agree on 154/200
questions (77%), and are right on 92.2% of them; on the 46 others, 52%.

## Highest reasoning level (2026-09-25)

`EFFORT=xhigh or_compare.py ...`: the same six, reasoning at "xhigh" (all
accepted it), same prompt, 200 questions, distillation flag on. Paired
comparison with the reasoning-off run; 95% bootstrap interval.

| model | off | xhigh | change | fixed / broke | thinking tokens | median latency | $ per 1,000 |
|---|---|---|---|---|---|---|---|
| **Kimi K2.6** | 78.5% | **85.0%** | **+6.5 [+3.0, +10.5]** | 14 / 1 | 591 | 8.1 s | 0.14 → 1.83 |
| **DeepSeek V4.1 Flash** | 77.5% | 82.0% | **+4.5 [+0.5, +8.5]** | 13 / 4 | 198 | 0.8 s | 0.06 → 0.26 |
| Qwen3.8 2.4T | 83.5% | 83.5% | 0.0 [-3.0, +3.0] | 5 / 5 | 158 | 2.1 s | 0.94 → 1.09 |
| DeepSeek V4 Pro | 80.5% | 80.0% | -0.5 [-4.0, +3.5] | 7 / 8 | 369 | 1.6 s | 0.27 → 1.69 |
| Kimi K3 | 83.5% | 82.0% | -1.5 [-3.5, 0.0] | 0 / 3 | 119 | 2.4 s | 3.15 → 5.90 |
| Nemotron 3 Ultra | 81.5% | (70.0%) | failed run | 4 / 27 | 0 | 1.6 s | 0.11 → 0.10 |
| vote of six | 83.0% | 84.5% | | | | | |

Nemotron's xhigh run is not a model result: routed to DeepInfra, it
reported no reasoning on any row and left 19 of 200 unanswered (77.3% on the
181 it answered). Even at the top setting the models think briefly
(120-590 tokens). Thinking lifts the weaker models to the level of the
strongest, not above: the ceiling stays near 83-85%, against Opus 92% and
human-label students 91-93%. Run cost ≈ $2.20.

## Escalating when unsure: logprobs vs agreement (2026-09-25)

Logprobs through OpenRouter with the distillation flag on (probe of every
allowed model that lists them): most DeepSeek, Kimi K2.x/K3, Nemotron,
Llama and open Qwen models return 20 alternatives; Alibaba-hosted Qwen
returns 5; Qwen3.8 2.4T / Max / Flash, Kimi K2.7 Code and several "thinking"
models return none. Some provider routes return impossible data (an
"alternative" at probability 1.0: DeepSeek V4 Flash via Parasail, DeepSeek
V3-0324 via GMICloud, Qwen3 Coder via Google) and must be avoided.

`or_logprob_targets.py MODEL any 20 200 off|none`: plain-text answer (no
enum), 77-way distribution rebuilt from the top-20 alternatives; confidence
= its top probability. "Keep at 90%" = the largest share of rows, taken
most-confident first, whose accuracy stays at or above 90%; the rest escalate.

| model | accuracy | AUROC (conf. → right) | acc. of top 80% / 50% | keep at 90% | keep at 95% | $ / 1,000 |
|---|---|---|---|---|---|---|
| **Kimi K3** | 81.5% | **0.86** | **92.5% / 95%** | **85.5%** | **70%** | 1.54 |
| **Kimi K2.6** | 81.5% | 0.80 | 90% / 95% | 80% | 54.5% | **0.12** |
| DeepSeek V4.1 Flash | 76.0% | 0.81 | 86.3% / 92% | 73% | 11.5% | **0.05** |
| Nemotron 3 Super | 73.5% | 0.83 | 82.5% / 93% | 62% | 41.5% | 0.04 |
| DeepSeek V4 Pro 0813 | 71.0% | 0.84 | 82.5% / 91% | 52.5% | 40.5% | 0.35 |
| Llama 4 Maverick | 73.0% | 0.80 | 82.5% / 91% | 52.5% | 11.5% | 0.09 |
| Qwen3.8 27B | 62.5% | 0.87 | 75% / 87% | 45.5% | 30% | 0.06 |
| Qwen3.5 397B | 65.5% | 0.83 | 73.8% / 94% | 56% | 37% | 0.11 |
| DeepSeek V4 Flash 0731 | 67.5% | 0.77 | 75.6% / 87% | 20% | 10.5% | 0.05 |
| Nemotron 3.5 Lightning | 64.5% | 0.78 | 72.5% / 83% | 37% | 30.5% | 0.03 |

Without the enum constraint several models lose accuracy (DeepSeek V4 Pro
80.5 → 71.0, Qwen3.8 27B 77.0 → 62.5): logprobs and a constrained answer do
not come together here. Kimi K2.6 and K3 keep theirs.

Agreement instead (enum answers from the earlier runs, any model):

| pair / trio | share where they agree | accuracy there | $ / 1,000 |
|---|---|---|---|
| DeepSeek V4 Flash 0731 + Qwen3.8 Flash | 84.5% | 86.4% | 0.06 |
| DeepSeek V4.1 Flash + Qwen3.8 Flash | 84.0% | 86.3% | 0.09 |
| Qwen3.8 Flash + Nemotron 3 Ultra | 88.5% | 86.4% | 0.14 |
| DeepSeek V4.1 Flash + Qwen3.8 Flash + Nemotron 3 Ultra | 80.5% | 88.2% | 0.20 |
| all six (above) | 77% | 92.2% | 4.65 |

At the same share kept (~80-85%), one Kimi model's logprobs give 90-92%
accuracy on the kept rows; two cheap models agreeing give ~86%. Logprobs also
let the threshold be moved; agreement gives one fixed split.

## Assistant prefill among distillable models (2026-09-25)

Test: the request ends with an assistant message "1, …, 10. Now backwards:
10, 9," — a model that honors the prefill continues with "8"; one that
ignores it starts over at "1". Distillation flag on, reasoning off.

Default routing, all 108 allowed models: 38 continue, 45 start over, 18
unclear (empty or odd answers), 4 refused. Support depends on the provider,
not only the model; per provider for the models that matter:

| model | providers that honor prefill |
|---|---|
| Kimi K3 | none of 15 |
| Kimi K2.6 | Baidu, StreamLake (2 of 18) |
| DeepSeek V4 Flash 0731 | 21 of 29 (not AtlasCloud, BaseTen, Cloudflare, DigitalOcean, Parasail, Together, Wafer) |
| DeepSeek V4 Pro 0813 | CoreWeave, DeepInfra, DeepSeek, Fireworks, GMICloud, NextBit, Phala, StreamLake, Venice (9 of 21) |
| DeepSeek V4.1 Flash | clean: DeepSeek, OpenInference, Phala; with stray text first: CoreWeave, DeepInfra, Fireworks, Makora, Together |
| Nemotron 3 Ultra | all 3 (BaseTen, DeepInfra, Venice) |
| Nemotron 3 Super | DeepInfra |
| Qwen3.8 2.4T / Max prime / Flash | none |
| Mistral (own API) | yes, the prefill is echoed back; Mistral Small 4 untestable (rate-limited) |

Pin a working provider with provider.order + allow_fallbacks=false.

## Reasoning prefill among distillable models (2026-09-25)

Test: a planted thought ("the secret code is ZEBRA-42"), asked "what is the
secret code? … else UNKNOWN". Without the prefill every model answers
UNKNOWN. Three ways to plant it: A = OpenRouter `reasoning` field on a final
assistant message; B = "<think>\n…" as the assistant content prefill; C = the
reasoning field plus a content prefill "The code is". Pinned providers where
content prefill worked, reasoning effort low.

One try each over 12 model/provider pairs: only 5 combinations used the
thought at all. Repeated 5 times:

| model @ provider | method | used the planted thought |
|---|---|---|
| Kimi K3 @ Parasail | B (<think> in content) | 4/5 |
| Nemotron 3 Ultra @ BaseTen | C (field + content prefill) | 4/5 |
| Kimi K3 @ Parasail | A (field) | 1/5 |
| DeepSeek V4 Flash 0731 @ DeepInfra | A | 2/5, output corrupted (stray tokens) |
| DeepSeek V4 Flash 0731 @ Fireworks | B | 0/5 (empty answers) |

Never used: Nemotron 3 Ultra @ DeepInfra, DeepSeek V4 Pro 0813 (DeepSeek,
DeepInfra), DeepSeek V4.1 Flash @ DeepSeek (field refused), Kimi K2.6
(StreamLake, Baidu), Qwen3.8 2.4T @ Together; Nemotron 3 Super @ DeepInfra
refused every request. Kimi K3 does not continue a content prefill (above),
so there the planted text is read as context rather than continued. Not
dependable for production through OpenRouter; open weights served locally
(vLLM, own chat template) give full control over the thinking.

## Dataclass + enum prefill as the confidence frame (2026-09-25)

`or_dataclass_logprobs.py`: the prompt declares `class Intent(Enum)` (77
members) and `@dataclass class Classification: intent: Intent`; the message
is `message = '…'`; the assistant is prefilled with
`result = Classification(intent=Intent.` and stops at ")", so the model
writes only the member name, and the 77-way distribution is rebuilt from the
top-20 logprobs of those tokens. Baseline = the plain prompt, same model and
same pinned provider, same 200 questions, distillation flag on.

Routes giving both prefill and sane logprobs: Kimi K3 (Parasail, Makora,
Morph, Phala, Sail Research), Kimi K2.6 (CoreWeave, Inceptron, Parasail),
DeepSeek V4 Pro 0813 (Parasail, Cloudflare), DeepSeek V4 Flash 0731
(Cloudflare); none for Nemotron. DeepSeek's own API breaks in this frame
("???" answers) and its plain-mode logprobs are uninformative (confidence
0.995 on average, AUROC 0.51).

| model @ provider | frame | accuracy | AUROC | ECE | acc. top 80% / 50% | keep at 90% | keep at 95% | $ / 1,000 |
|---|---|---|---|---|---|---|---|---|
| Kimi K3 @ Parasail | dataclass | 84.0% | 0.834 | 0.095 | 92.5% / 97% | 85% | 74% | 1.30 |
| | plain | 81.5% | 0.849 | 0.099 | 93.1% / 95% | 86% | 66.5% | 1.54 |
| Kimi K2.6 @ CoreWeave | dataclass | 80.0% | **0.852** | 0.142 | 90.6% / 94% | 81.5% | 42% | 0.47 |
| | plain | 81.5% | 0.797 | 0.130 | 89.4% / 95% | 78.5% | 56.5% | 0.12 |
| DeepSeek V4 Pro 0813 @ Parasail | dataclass | 76.0% | 0.780 | 0.176 | 83.8% / 89% | 58% | 31% | 0.27 |
| | plain | 78.0% | 0.760 | 0.172 | 83.1% / 92% | 63% | 34.5% | 0.38 |
| DeepSeek V4 Flash 0731 @ Cloudflare | dataclass | 70.0% | 0.755 | 0.196 | 77.5% / 85% | 33.5% | 11% | 0.08 |
| | plain | 71.5% | 0.767 | 0.213 | 78.1% / 89% | 57.5% | 14% | 0.12 |

Paired differences (dataclass − plain, 95% bootstrap): accuracy Kimi K3
+2.5 [0.0, +5.5], K2.6 −1.5 [−4.5, +1.5], V4 Pro −2.0 [−6.5, +2.5], V4 Flash
−1.5 [−6.5, +3.5]; AUROC K3 −0.015 [−0.08, +0.05], K2.6 +0.055 [+0.005,
+0.11], V4 Pro +0.02 [−0.06, +0.09], V4 Flash −0.01 [−0.09, +0.05].
The frame's confidence tracks correctness (AUROC 0.76-0.85) about as well as
the plain frame's; only K2.6's AUROC improves beyond noise. Its practical
gains: the output is exactly one member name (no parsing) and it opens
prefill on routes where the counting-test prefill failed (Kimi K3). The code
prompt is longer (~1,300 tokens), which raises cost on cheap models.
