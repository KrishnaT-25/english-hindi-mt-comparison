# English to Hindi Translation Results

All four systems were scored on the same 500 fixed IITB test examples.

| Model | BLEU | chrF | Mean ms/sentence | Training seconds | Total parameters | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer | 0.36 | 10.56 | 385.7 | 1128.8 | 13,525,824 | 13,525,824 |
| mt5 | 0.06 | 6.99 | 4462.4 | ≈6829.7 | 300,520,832 | 344,064 |
| llama | 4.11 | 19.55 | 3011.2 | N/A | 1,235,814,400 | 0 |
| gemma | 9.83 | 35.59 | 4711.9 | N/A | 999,885,952 | 0 |

## Output diagnostics

| Model | Empty | Truncated at 128 tokens | No Devanagari characters |
| --- | ---: | ---: | ---: |
| transformer | 0 | 54 | 25 |
| mt5 | 0 | 310 | 0 |
| llama | 0 | 43 | 188 |
| gemma | 0 | 3 | 2 |

Highest measured chrF: **gemma**. This is a quality ranking on this test set, not an architecture-only effect.

## Model comparison

The best encoder-decoder system here is transformer (chrF 10.56); the best prompt-only decoder system is gemma (chrF 35.59). This comparison reflects the actual trained checkpoints and prompts, not a controlled architecture ablation.

transformer has the lowest mean recorded inference time (385.7 ms/sentence at batch size 1). The table also gives total and adapted parameter counts. The two prompt-only systems have no adaptation time; the trained systems used different procedures and data exposure.

## Temperature study

The following 100-example validation runs use one seed. Greedy decoding disables sampling; the other runs use `top_k=0` and `top_p=1`.

| Model | Decoding | BLEU | chrF |
| --- | --- | ---: | ---: |
| transformer | Greedy | 0.25 | 9.82 |
| transformer | Temperature 0.3 | 0.33 | 11.60 |
| transformer | Temperature 0.7 | 0.23 | 12.06 |
| mt5 | Greedy | 0.11 | 6.08 |
| mt5 | Temperature 0.3 | 0.27 | 8.85 |
| mt5 | Temperature 0.7 | 0.12 | 8.56 |
| llama | Greedy | 2.69 | 18.64 |
| llama | Temperature 0.3 | 2.33 | 16.65 |
| llama | Temperature 0.7 | 3.11 | 16.85 |
| gemma | Greedy | 7.59 | 34.99 |
| gemma | Temperature 0.3 | 7.25 | 33.57 |
| gemma | Temperature 0.7 | 6.95 | 32.74 |

On this fixed validation subset, transformer: t0.7 gives the highest validation chrF (12.06; change from greedy +2.24). mt5: t0.3 gives the highest validation chrF (8.85; change from greedy +2.77). llama: greedy gives the highest validation chrF (18.64; change from greedy +0.00). gemma: greedy gives the highest validation chrF (34.99; change from greedy +0.00).
These 100-example, one-seed results are exploratory and were not used to select test checkpoints. See [temperature output changes and examples](qualitative/temperature_examples.md).

## Limitations

The systems differ in pretraining, model size, tokenization, precision, and adaptation. The scratch model was trained on a small subset. Decoder-only systems used prompting, so their training time is not applicable.

mT5 used its validation-selected epoch-2 adapter. Epoch 3 was interrupted by a CUDA error; its unfinished updates were not scored. The reported adaptation time is an estimate from file timestamps.

The primary comparison uses the first 500 IDs of the frozen seed-42 test selection. The Transformer, mT5, and Llama also have supplemental 1,000-example prediction files, but those larger runs are not ranked against Gemma's smaller run.

## Error and responsible use review

See the [manual review of observed errors](qualitative/manual_review.md) and [actual translation examples](qualitative/review_examples.md) for names, numbers, negation, unsupported additions, gendered wording, cultural meaning, and sensitive text. Consequential translations require human review. The separate temperature experiment is exploratory.
