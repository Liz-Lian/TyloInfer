# Utility commands

Additional commands beyond [build](build.md) and [serve](serve.md).

## `tyloinfer wt2-reference`

Generate official WT2 reference logits from original HF weights. The output
feeds KL evaluation through `mfq-eval kl --kl-base`.

The manifest records a SHA-256 identity of the ordered token vocabulary.
Metal `mfq-perplexity` checks this identity against the candidate tokenizer
before evaluation. Older manifests without this identity must be regenerated;
matching vocabulary sizes alone do not establish compatible token IDs.

| Option | Meaning | Default |
|---|---|---|
| `--model PATH` | Original HF checkpoint. | Required |
| `--dataset PATH` | WT2 evaluation corpus. | Required |
| `--output PATH` | Reference logits file. | Required |
| `--context-size N` | Evaluation context size. | `512` |
| `--chunks N` | Corpus chunks. | `8` |
| `--backend {auto,metal,cuda}` | Execution backend. | `auto` |
| `--device NAME` | Device; must match the selected backend. | `mps` or `cuda:0` |
| `--resident` | Keep the model resident instead of layerwise streaming. | Off |

```shell
uv run tyloinfer wt2-reference --model model-hf --dataset wikitext-corpus \
  --output reference.bin
```

## `tyloinfer voice-runtime-check`

Verify that the optional MiniCPM-o voice output dependencies import on this
host: ONNX Runtime, the tokenizer and audio stacks, and the CosyVoice modules.
The command prints the first missing dependency and exits nonzero on failure.

```shell
uv sync --extra minicpmo45
uv run tyloinfer voice-runtime-check
```

`tyloinfer inspect` is reserved for future use and is not implemented yet.
