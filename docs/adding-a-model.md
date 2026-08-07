# Adding a Model

Models are defined through YAML config files. A config defines model metadata, selects one backend, and provides the backend-specific arguments used during initialization and generation.

The benchmark currently supports these backends:

| Backend | Preferred | Runtime |
| --- | --- | --- |
| `api` | yes | OpenAI-compatible chat completions API |
| `vllm` | yes | `vllm.LLM` |
| `hf` | compatibility | `transformers` |
| `random` | baseline only | random baseline |

## Top-Level Fields

Every config has the following top-level schema:

```yaml
name: Model Display Name
org: Organization
model_id: provider-or-path/model-id
model_family: model_family
model_type: open
model_size: 7B
backend: api
model_args: {}
generation: {}
preprocessing:
  steps: []
postprocessing:
  steps: []
```

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `model_id` | yes | `string` | Backend model identifier. For API backends this is sent as the `model` argument. For local backends this is passed to the model loader. |
| `backend` | yes | `api` / `vllm` / `hf` / `random` | Backend implementation used for inference. |
| `name` | no | `string` / `null` | Display name stored in run metadata and used as `<model_name>` in leaderboard output paths. If omitted, the evaluator falls back to `model_id`. |
| `org` | no | `string` / `null` | Organization stored in run metadata and used as `<org>` in leaderboard output paths. If omitted, the evaluator derives it from `model_id` when possible. |
| `model_family` | no | `string` / `null` | Reporting metadata, for example `qwen3_5`, `gemma`, or `llava`. |
| `model_type` | no | `string` / `null` | Reporting metadata, for example `open`, `closed`, `random`, or a project-specific grouping. |
| `model_size` | no | `string` / `null` | Reporting metadata, for example `7B`, `27B`, or `397B`. |
| `model_args` | no | `mapping` | Backend initialization and execution options. Defaults to `{}`. |
| `generation` | no | `mapping` | Generation parameters. Interpretation depends on the backend. Defaults to `{}`. |
| `preprocessing` | no | `mapping` | Prompt preprocessing pipeline. Defaults to no preprocessing. |
| `postprocessing` | no | `mapping` | Raw-output postprocessing pipeline. Defaults to no postprocessing. |

Leaderboard artifacts are written to:

```text
results/leaderboard/<org>/<model_name>/<dataset_revision>/<split>/<evaluation_mode>/
```

## API Backend

Use `backend: api` for models available through an OpenAI-compatible chat completions endpoint. The backend sends a single user message containing text and, when available, an image encoded as a data URL.

Example:

```yaml
backend: api
model_args:
  api_key_env: OPENAI_API_KEY
  base_url: https://api.openai.com/v1
  timeout: 120
  max_retries: 10
  num_threads: 8
generation:
  max_completion_tokens: 64
  temperature: 0.0
  top_p: 1.0
```

### `model_args`

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `api_key_env` | no | `string` | `OPENAI_API_KEY` | Environment variable that contains the API key. The value can be provided by the shell or `.env`. |
| `base_url` | no | `string` / `null` | OpenAI SDK default | API base URL passed to `openai.OpenAI`. Set this for OpenRouter or internal servers. |
| `timeout` | no | `number` / `null` | OpenAI SDK default | Request timeout passed to `openai.OpenAI`. |
| `max_retries` | no | `integer` | `5` | Retry attempts for timeout, connection, rate-limit, and server errors. |
| `num_threads` | no | `integer` | `8` | Number of concurrent API requests used by `generate_batch`. |

### `generation`

All `generation` keys are forwarded to:

```python
client.chat.completions.create(model=model_id, messages=messages, **generation)
```

Use the parameter names expected by the provider.

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `max_completion_tokens` | no | `integer` | provider default | Common token limit for newer OpenAI-compatible APIs. |
| `max_tokens` | no | `integer` | provider default | Token limit used by some OpenAI-compatible providers. |
| `temperature` | no | `number` | provider default | Sampling temperature. Use `0.0` for deterministic decoding when supported. |
| `top_p` | no | `number` | provider default | Nucleus sampling parameter. |
| `extra_body` | no | `mapping` | provider default | Provider-specific request body extension, when supported by the SDK/provider. |

The table lists common fields only. The backend does not validate or restrict `generation`.

## vLLM Backend

Use `backend: vllm` for local GPU inference through `vllm.LLM`. The backend builds either prompt-based requests for `llm.generate(...)` or chat requests for `llm.chat(...)`.

Example:

```yaml
backend: vllm
model_args:
  dtype: bfloat16
  trust_remote_code: true
  max_model_len: 8192
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  generate_batch_size: 16
generation:
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  repetition_penalty: 1.0
```

### `model_args`

Most `model_args` keys are passed directly to:

```python
LLM(model=model_id, **model_args)
```

The backend consumes the following specific keys before constructing `LLM`.

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `use_chat` | no | `boolean` | `false` | If `true`, use `llm.chat(...)`. Otherwise, use prompt-based `llm.generate(...)`. |
| `generate_batch_size` | no | `integer` | `0`, or `1` for Mistral tokenizer mode | Batch size used when calling vLLM. `0` means all records in the current task batch. |
| `tokenizer_kwargs` | no | `mapping` | `{}` | Extra arguments passed to `AutoProcessor.from_pretrained(...)` or `AutoTokenizer.from_pretrained(...)`. |

The following keys are passed to vLLM:

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `dtype` | no | `string` | vLLM default | Model dtype, for example `bfloat16`, `float16`, or `auto`. |
| `trust_remote_code` | no | `boolean` | vLLM default | Passed to vLLM and also used when loading the processor/tokenizer. |
| `max_model_len` | no | `integer` | vLLM default | Maximum model context length. |
| `tensor_parallel_size` | no | `integer` | vLLM default | Number of GPUs used for tensor parallelism. |
| `gpu_memory_utilization` | no | `number` | vLLM default | Fraction of GPU memory vLLM may use. |
| `enforce_eager` | no | `boolean` | vLLM default | vLLM execution-mode option. |
| `tokenizer_mode` | no | `string` | vLLM default | Tokenizer mode. If set to `mistral`, defaults `generate_batch_size` to `1` and sets `fix_mistral_regex` in tokenizer kwargs. |

### `generation`

The vLLM backend converts `generation` into `vllm.SamplingParams`.

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | no | `integer` | `64` | Mapped to `SamplingParams.max_tokens`. |
| `temperature` | no | `number` | `0.0` | Mapped to `SamplingParams.temperature`. |
| `top_p` | no | `number` | `1.0` | Mapped to `SamplingParams.top_p`. |
| `top_k` | no | `integer` | `0` | Mapped to `SamplingParams.top_k`. |
| `repetition_penalty` | no | `number` | `1.0` | Mapped to `SamplingParams.repetition_penalty`. |
| `skip_special_tokens` | no | `boolean` | `true` | Mapped to `SamplingParams.skip_special_tokens`. |
| `chat_template_kwargs` | no | `mapping` | `{}` | Forwarded to the tokenizer/processor chat template and to `llm.chat(...)` when `use_chat` is enabled. |

## Hugging Face Backend

Use `backend: hf` as a compatibility path for local `transformers` inference. The backend loads `AutoProcessor` and then tries `AutoModelForImageTextToText`, `AutoModelForVision2Seq`, and `AutoModelForCausalLM`.

Example:

```yaml
backend: hf
model_args:
  trust_remote_code: true
  dtype: float16
  device_map: auto
generation:
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
```

### `model_args`

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `trust_remote_code` | no | `boolean` | `true` | Passed to `AutoProcessor.from_pretrained(...)` and model loading. |
| `device_map` | no | `string` / `mapping` | `auto` | Device placement passed to model loading. |
| `dtype` | no | `string` / `null` | `float16` | Converted to a `torch` dtype and passed as `torch_dtype`. Use `auto`, `none`, or null to omit `torch_dtype`. |

Other `model_args` keys are ignored by this backend.

### `generation`

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | no | `integer` | `64` | Passed to `model.generate(...)`. |
| `temperature` | no | `number` | `0.0` | If greater than `0`, enables sampling and is passed to `model.generate(...)`. |
| `top_p` | no | `number` | `1.0` | Used only when `temperature > 0`. |
| `chat_template_kwargs` | no | `mapping` | `{}` | Forwarded to `processor.apply_chat_template(...)`. |

## Random Backend

Use `backend: random` only for random baseline configs.

| Section | Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `model_args` | `seed` | no | `integer` / `null` | `generation.seed`, then unseeded RNG | Seed for deterministic random predictions. |
| `generation` | `seed` | no | `integer` / `null` | unseeded RNG | Fallback seed if `model_args.seed` is absent. |

## Preprocessing

Preprocessing is applied to the prompt before it is sent to the backend.

```yaml
preprocessing:
  steps:
    - name: prepend
      args:
        prefix: "<image>\n"
        if_missing: true
```

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `steps` | no | `list` | `[]` | Ordered list of preprocessing steps. |
| `steps[].name` | yes, per step | `string` | none | Preprocessor name. Currently only `prepend` is supported. |
| `steps[].args` | no | `mapping` | `{}` | Keyword arguments for the selected preprocessor. |

Supported preprocessors:

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Argument</th>
      <th>Required</th>
      <th>Type</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><code>prepend</code></td>
      <td><code>prefix</code></td>
      <td>yes</td>
      <td><code>string</code></td>
      <td>none</td>
      <td>Text inserted at the beginning of the prompt.</td>
    </tr>
    <tr>
      <td><code>if_missing</code></td>
      <td>no</td>
      <td><code>boolean</code></td>
      <td><code>true</code></td>
      <td>If <code>true</code>, do not prepend when the prompt already starts with <code>prefix</code>.</td>
    </tr>
  </tbody>
</table>

## Postprocessing

Postprocessing is applied to raw model output before task parsing.

```yaml
postprocessing:
  steps:
    - name: replace
      args:
        pattern: "^Answer:\\s*"
        replacement: ""
        flags: ["ignorecase"]
```

| Field | Required | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `steps` | no | `list` | `[]` | Ordered list of postprocessing steps. |
| `steps[].name` | yes, per step | `string` | none | Postprocessor name. Currently only `replace` is supported. |
| `steps[].args` | no | `mapping` | `{}` | Keyword arguments for the selected postprocessor. |

Supported postprocessors:

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Argument</th>
      <th>Required</th>
      <th>Type</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><code>replace</code></td>
      <td><code>pattern</code></td>
      <td>yes</td>
      <td><code>string</code></td>
      <td>none</td>
      <td>Regex pattern passed to <code>re.sub</code>.</td>
    </tr>
    <tr>
      <td><code>replacement</code></td>
      <td>no</td>
      <td><code>string</code></td>
      <td><code>""</code></td>
      <td>Replacement string passed to <code>re.sub</code>.</td>
    </tr>
    <tr>
      <td><code>flags</code></td>
      <td>no</td>
      <td><code>list</code> of <code>string</code></td>
      <td><code>[]</code></td>
      <td>Regex flags. Supported values are <code>ignorecase</code>, <code>multiline</code>, and <code>dotall</code>.</td>
    </tr>
  </tbody>
</table>
