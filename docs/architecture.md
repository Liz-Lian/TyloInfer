# TyloInfer Architecture

This document defines the target architecture and module responsibilities; it does not mean the source has finished
migrating. [The implementation plan](plan.md) covers current status, remaining tasks and acceptance, and
[the refactoring conventions](refactor.md) cover implementation rules; where architecture descriptions conflict, this
document wins.

The example code and call signatures below only illustrate responsibilities, dependencies and data flow; a verbatim
implementation is not required. Names, parameter forms and code style may be adjusted as needed; the module
responsibilities, ownership, numerical and lifecycle constraints stated in the text must still be met.

This phase first establishes the complete chain of public interfaces and the CUDA Engine on Gemma4, then reuses it on
Qwen35, each covering the capabilities it already supports.
Compiling, running and compatibility of Metal/HIP are not constraints in this phase.

## Overall Structure

`csrc/engine/` is the complete inference library. The Engine object holds an injected CausalModel and Executor
separately, owns an internal Scheduler, and coordinates requests, the generation flow and resource lifecycles.
Callers submit requests and consume output; the Scheduler advances generation inside the Engine.
The Scheduler's source is compiled into the Engine target, and the Engine propagates its public headers and compile
definitions.

```text
App / Transport → Engine: submit, cancel, read output, management operations
                      │
              Scheduler → generation
                            ├─ Executor: establish the execution scope, manage execution resources and completion
                            └─ CausalModel::forward(input, state)
                                 → Attention / MLP / Encoder.forward inside the model
                                 → public ops: numeric contract and device dispatch
                                 → backend ops: borrow execution resources from the current scope
                                 → kernels / native of the same backend
```

requests, scheduler, generation, state and weights are the five runtime modules.
generation owns generation advance; models compose compute units and call public ops, and neither receive nor store an
Executor.
`engine/ops.*` defines the public numeric interface and dispatches to backends; `engine/executor.h` defines the
execution scope and lifecycle contract, and device implementations live in `engine/backends/<device>/executor/`.
Executor does not forward model forward calls or relay individual op calls.

TyloInfer's own C++ namespace is unified as `tyloi`, and the public tensor `tyloi::Tensor` is an owning handle that is
copyable, movable and returnable by value.
Activations, weights and state payloads share this handle; Tensor internally holds a polymorphic TensorImpl,
DenseTensor and each encoded Tensor implement a concrete data representation, and device execution belongs to the
backend.

## Final Target Layout

```text
csrc/
├── engine/
│   ├── engine.*            # inference instance interface and lifecycle coordination
│   ├── load.*              # library-level load/reload assembly; calls the unified model factory and backend resource entry points
│   ├── executor.h          # execution scope, completion tracking and lifecycle contract
│   ├── tensor.*            # Tensor handle, TensorImpl representation interface and ownership contract
│   ├── dense_tensor.*      # dense Tensor and view implementation
│   ├── ops.*               # public numeric interface, parameters and backend dispatch
│   ├── config.h            # instance-level configuration
│   ├── budget.*            # cross-module resource budget coordination
│   ├── requests/           # request/result contract, input preparation, output interpretation
│   ├── scheduler/          # request lifecycle, scheduling, backpressure and batching
│   ├── generation/         # prefill/decode, sampling and MTP policy
│   ├── state/              # generic state payload, pages, snapshots and session
│   ├── weights/            # model file reading, weight preparation and residency
│   ├── models/             # model definitions, assembly, forward and dedicated state
│   └── backends/
│       ├── cuda/           # executor, ops, kernels, native
│       ├── metal/
│       ├── hip/
│       └── cpu/
├── components/             # generic exceptions, IO, JSON, tokenizer, media codecs
├── transport/              # external protocol parsing and result transport
├── apps/                   # main and process startup, no model dispatch or runtime implementation
├── commands/               # argument parsing and command behaviour, calling library-level load and lifecycle interfaces
├── eval/
├── diagnostics/            # native diagnostics, numeric case execution and result export
└── tests/                  # native unit tests and fixed numeric regression
```

In the text below, the runtime modules, models and backends paths are all relative to `csrc/engine/`.
source means `weights/source/`; public ops live in the root `ops.*`, and device ops, kernels and native live in the
corresponding backend.
All devices share the strategy and lifecycle of the runtime modules; device computation and primitives are centralised
in backends.
The repository root `tests/numerics/` carries the offline numeric verification module, including cases, reference
result generation and comparison; it is not part of the inference library, and its contract is in
[Numerical correctness verification](#numerical-correctness-verification).

## Module Internals and File Responsibilities

`.*` means `.h/.cpp/.cu` as actually needed. File division expresses responsibility; it does not require a new class
per file.
Shared implementation is justified by actual reuse, and other models and backends organise files by their own
capabilities.

### Private Implementation and Naming

File names express responsibility, and privacy is decided by the owning module and the allowed consumers. Own business
files do not use `internal`, `detail` or `impl` in place of a responsibility name, and do not turn a cross-module
dependency into a private one by changing a suffix.
Helper functions, types and non-template implementation used by a single source file go into that `.cpp`, in an
anonymous namespace or as private class members; only implementation genuinely shared by several source files keeps a
private header with a clear responsibility, for example `weights/cache_resources.h`.
A private header is for the implementation of its own module only; it is not include-propagated by public headers and
is not a call entry point for other modules. Behaviour genuinely needed across modules is provided through the owning
module's interface, and callers do not extract internal state and then perform its lifecycle operations themselves.
Namespaces are organised by responsibility and owning module, for example `tyloi::engine`, `tyloi::models`,
`tyloi::weights`, `tyloi::cuda`. No shared state entry point or miscellaneous utility collection such as
`weights::detail` / `details` is created; a namespace provides no C++ private access control. Local template helpers
genuinely needed by a header stay scoped to that specific responsibility, and helper namespaces are not used to expose
mutable internal state.
Types with an explicit representational meaning such as `TensorImpl`, and an in-class `Impl` used to hide data, may
stay; one extra forwarding layer, directory or class is not added just to remove a suffix.

Existing files converge to the responsibilities below. The table states target ownership, not completed migration:

| Current file | Target location and naming |
| --- | --- |
| `generation/cuda_impl.h`, `generation/mtp/cuda_impl.h` | Generation advance is merged into the respective `generation.cpp`; model computation and state semantics belong to the family, CUDA execution belongs to the backend; after the split the mixed-responsibility header is deleted, with no wholesale rename or move |
| `models/prepare_internal.h` | The `special_token` declaration joins the existing `models/prepare.h`, implementation stays in `prepare.cpp`; no extra header for a single declaration |
| `weights/cache_internal.h`, `weights::detail` | Split by cache resources, prefill workspace and KV borrow state, see the weights naming table; delete the whole-state access entry point `detail::cache`, with no wholesale rename into another helper namespace |
| `weights/moe/expert_cache_internal.h` | Cache class declarations and the necessary private data belong to `moe/expert_cache.h`, non-template implementation to `expert_cache.cpp`; this header is an implementation contract inside the weights module and is not included by backends |
| `weights/moe/cache_types_internal.h` | Expert pool fields and layout belong to the existing `pool.h`, preparation and transfer records to `prepare.*`, slots and swap transactions to the cache/residency implementation; types used by a single file stay in the `.cpp`, with no rebuilt `types` miscellaneous header |
| `weights/moe/cached_source_internal.h` | Source, preparation and lease belong to the private implementation of `moe/cached_source.*`; the actual projection and GLU computation move into the existing backend projection/ffn implementation; backends consume the data through the preparation contract and do not include this private header |
| `with_model` in `models/load.h` | Family dispatch belongs to `models/load.cpp`, and the configuration, weights and optional component construction of each branch belong to the corresponding `<family>/load.*`; remove the old templates that expose concrete types to callers |
| Mixed assembly in `engine/load.*` | Keep the generic load/reload coordination, CLI string parsing belongs to commands, device resource initialisation belongs to backend; model special-casing belongs to the family, without moving the whole file |
| `engine/instance_impl.h`, `cuda::engine_detail` | Runtime lifecycle belongs to Engine, generation advance to generation, concrete model behaviour to the family; delete the family-enumerated `State` variant and the `EngineInstance` templates, and do not create a `detail` container to take them over |

Template definitions in headers may stay where instantiation requires it; files are still named by behaviour. Whether
convergence is done is judged by actual include and call relationships, never only by a reduced count of `_internal`
files.

### Engine Root

| File | Responsibility |
| --- | --- |
| engine.* | Submit, cancel, output access, status query, session control, unload and dependency re-binding |
| load.* | Library-level load/reload entry point: consumes typed placement options, combines ModelSource, weight cache, model factory, backend executor and Engine, and handles failure wrap-up and ownership transfer; does not parse CLI strings and does not implement family dispatch or model math |
| executor.h | Execution scope, work completion and draining contract; contains no numeric interface such as project, norm or attention |
| tensor.*, dense_tensor.* | Tensor owning handle, TensorImpl representation interface and DenseTensor implementation; logical metadata, storage enumeration and handle semantics are in [Unified Tensor contract](#unified-tensor-contract) |
| ops.* | Public numeric contract, parameters and device dispatch; the interface exposes no device-private type, and the implementation is selected among the backends allowed by the build |
| config.* | Engine's Config and runtime option validation: instance, session, batch and Graph policy, plus the device, stop token and budget configuration provided by the assembly entry point; does not depend on the whole CUDA load options type |
| budget.* | The public `Budget` coordinates state, weights and workspace budgets and handles request, normal/batch prefill and snapshot space reservation; models provide byte requirements and planning receives the device explicitly; it manages budget attach and detach, and each module manages its own resources and phase policy |
| metrics.h, resource_metrics.h | Public counter formats and Engine resource aggregation; producers read batch, KV, offload and predictor counters, and aggregation consumes only counters and the public resource contract |
| CMakeLists.txt | Organises the compile targets of the runtime core, models and device support according to real dependencies |
| cuda.cmake | CUDA instance assembly and compile target composition |

### requests

| File | Responsibility |
| --- | --- |
| request.h, result.h | External request, incremental result, usage, terminal state and error contract |
| media.h | Data contract and ordering information for image, audio and video input |
| processor.* | Organisation of input validation, template application, tokenize, constraint preparation and model input preparation |
| tokenizer.* | Inference-side adapter for the generic tokenizer, shared by input and output |
| constraint.*, sampling.h | Grammar constraint construction and state, sampling parameter definition |
| output.* | Token to text, structured result parsing, final result |
| emitter.* | UTF-8 incremental concatenation, stop string and text fragment handling |

requests obtain input rules, tokenizer assets and protocol semantics through the model's public contract and produce
owned output results.
Request queueing, cancellation, output channels and terminal state publishing are managed by the Scheduler.

### scheduler

| File | Responsibility |
| --- | --- |
| scheduler.* | Request table, scheduling loop, admission, cancellation, deadlines, backpressure and single terminal state publishing |
| request.* | Active request record: preparation state, progress, model state holding, generation handle and output channel |
| batch.* | Request combination, slot allocation, batch membership and this round's work list |
| policy.* | Priority, prefill budget and work selection |

requests/request.h is the submission contract, and scheduler/request.* is the only runtime record.
The Scheduler decides which requests run and how much work they get; generation advances the computation, and models
define the batch state layout.
Batches reference the same active requests and do not decide request terminal state on their own.

### generation

| File | Responsibility |
| --- | --- |
| generation.h, generation.cpp | Shared media preparation, request validation and cleanup; organises normal, session and MTP generation through the public model, state and execution interfaces |
| decode.* | Fixed decode input, sampling buffers and sampling key; Graph execution uses the public Executor |
| sequence.*, step.* | Generation coroutines; normal and batch generation share prefill/decode advance, progress and cancellation interrupt points |
| prefill_activity.* | Prefill activity counters and recent-activity records |
| batch.* | Generic batch generation advance, calling model batch capability |
| prefill_budget.h | Pure prefill budget computation and batch selection helpers |
| sampling.* | Sampling policy, penalties and constraint advance, with device computation handed to the backend |
| mtp/sequence.* | Generic MTP advance, verification and acceptance flow, and it carries execution condition checks |
| mtp/generation.* | Organises draft, target verification, acceptance and rollback through the public model, Mtp and MtpState interfaces, without instantiating the generation flow per family |
| mtp/policy.*, mtp/depth_controller.* | Speculation policy and depth selection |
| mtp/metrics.* | Generic MTP metrics |
| mtp/state.h | Instance-level depth policy state and previous-round generation statistics, held by Engine |
| mtp/batch_policy.* | Batch speculation policy and depth selection |

The generation entry point and normal and MTP advance are centralised in `engine/generation`;
Engine keeps the resource owner and call entry point, models keep model-specific computation, state and policy, and no
second copy of a generic generation implementation or forwarding shell is kept.
The public Engine and batch share request validation and cleanup; media preparation and progress timing have exactly
one implementation. session and MTP transactions are connected according to existing capability, and no empty
capability is constructed for paths that are not supported.
generation collaborates through the public CausalModel, State, Mtp, MtpState, Executor and budget/cache contracts; it
contains no CUDA executor/native/kernel or concrete family implementation header, and does not pass CUDA Mode,
Collectives, profiler or stream.
Model differences are expressed through the existing public capability and state interfaces; generation does not select
between old and new call styles by field probing, ADL forward/logits or template branching, and no second generation
hook interface is added.
CUDA Graph, timing, transfer and scratch are used through the public Executor/ops contracts, with concrete execution
left in the backend; model-specific positions, predictor history and forward variants are implemented by the family.
Missing capability is added to the interface that actually owns that behaviour, and the whole old execution context or
generation object is not passed in.
`cuda.h`, the two `cuda_impl.h`, `generation/ops.h`, `mtp/ops.h`, `instance.h` and `engine/instance_impl.h` are
existing transitional assembly and adaptation, not the target structure. They are deleted once their live capability
has been moved: generic flow belongs to generation, model adaptation and model construction to the family, library-level
resource assembly to `engine/load.*`, and device execution to the backend. Moving them to `apps/instance.h` or
`apps/model.h` does not count as responsibility convergence.
Qwen35 and other transitional models reuse the same instance entry point and do not maintain their own Engine loading,
generation, reload and release implementations; the transitional instance shell and the resource holding in
engine/text.h converge into Engine's single lifecycle.
Normal generation and batch share the prefill/decode, pause and publish loop; Graph preparation and replay stay inside
single-step computation. A generation coroutine holds its own computation objects or borrows the instance explicitly;
child coroutine borrows end first, then request resources are cleaned up.
generation borrows the model and executor held by Engine, establishes the execution scope outside the advance or
coroutine resume,
and then completes sampling by calling model forward and public ops through Tensor handles. forward returns hidden and
other model results, which are sampled directly or after a model-side logits projection;
the concrete result semantics are defined by the model contract. Generation code holds no family-private layout and
passes no executor to the model.
Generic MTP advance, verification and acceptance logic belongs to `generation/mtp`; model-specific depth policy
belongs to `models/<family>`, where the predictor provides the depth policy and draft sampling parameters; the default
adaptive and batch policies still come from `generation/mtp`. Generic generation code does not depend on concrete
family types.
Depth policy state and generation statistics are held by Engine; the target request State holds a separate
models::MtpState storing predictor caches, positions and Graph state. models::Mtp keeps weights and computation, and
every state operation receives its owning MtpState explicitly; positions and Graph counters are read from state.
The predictor's family-specific projection, capacity validation and cursor advance are implemented by the corresponding
family; common does not organise predictor computation by field probing.
Convergence reuses the existing generation and numeric tests, covering normal/MTP, chunking, cancellation, session
restore and Graph replay; converging declarations and includes is not a substitute for verifying executed behaviour.

### state

| File | Responsibility |
| --- | --- |
| kv_view.*, recurrent.* | KvView holding a Tensor, shared by plain and paged KV, with quantization and geometry metadata in state; recurrent and conv state payloads and operations |
| pages.h, paged_kv.* | Page identity, references, allocation generation, paged management and KV binding |
| kv_memory.*, kv_budget.h | KV capacity, budget, allocation and reclamation; allocators and reservation reuse the hot cache, VMM and streaming staging layouts, reserving on the actual device before allocating |
| kv_offload.* | Device/RAM/SSD transfer; input borrows Tensors, read results keep their allocation alive but do not monopolise hot block slots, and fixed gather buffers are reused per consumer barrier; KV page mapping policy stays in state |
| qsa_blocks.*, disk_budget.h | RAM/SSD storage of opaque KV byte blocks and disk reservation coordination |
| session/sessions.h, session/cache.* | `Sessions` public control, accounting and budget operations; the shared cache implements matching, storage and eviction directly, and snapshot types stay in the concrete instance |
| session/prefix_cache.* | Prefix matching, sharing and reuse |
| session/prefix_blocks.*, session/paged.* | Prefix byte block storage and session ownership |
| session/snapshot.h | Matching, budget and eviction of generic session snapshots |

state manages generic storage mechanisms and the session/cache lifecycle; KV and media caches hold Tensor owners, and no
duplicate KVCache or CUDA view type is added. Model snapshot layout, encoding and validation, complete-boundary
decision, joint restore and position semantics are defined by the corresponding family. Caches consume only the shared
metadata and Tensor enumeration, and count by actual backing identity with de-duplication.

### weights

| File or directory | Responsibility |
| --- | --- |
| loader.* | Raw asset reading and concrete Tensor preparation; `Loader` is created by `Cache`, publishes the data contract, and keeps backend resource borrowing in the private implementation; weights return Tensor handles, grouped returns an ordered Tensor list, and published weights are read-only by convention |
| placement.h, group.* | Execution placement and its validation, sharding and ordered Tensor group preparation; concrete shard layouts stay inside the implementation, backend group construction belongs to projection, and public weight groups expose no CUDA execution type |
| cache.*, workspace.* | `Cache` manages residency, budget, accounting, Graph policy and cache storage preparation, plus the load/prefill/decode phase policy; `Routes` holds routing transactions, `Warmup` restores warmup policy, `Priority` restores layer priority at scope end, and `Loan` moves Tensor outputs out before storage is returned; concrete cache, load and transfer details stay in the private implementation |
| memory.* | `Memory` records weight residency and PLE statistics; `LoadScope` manages the load accounting scope, and backend recording facilities stay inside the implementation |
| source/source.h, source/read.* | The read-only Source contract and tensor/asset reading helpers, containing no model architecture, layout or graph |
| source/gguf.*, safetensors.*, hf.*, mfq.* | Containers, shard indexes and byte-range reading |
| encoding/ | TqTensor, NintTensor, NvqTensor, MxTensor and other encoding implementations with layout validation; reuses bit layout, scale/codebook, row/block decode and packing logic |
| mx.h, sq.h, nint.h, vq.h | Preparation and format helpers for the corresponding encoded Tensor; reuse the encoding implementation, numeric computation belongs to backend ops |
| nint_cache.h, nint_row_cache.h, nint_row_pipeline.* | Single-layer prefill expansion cache, raw encoded rows and borrow leases; read-only row sources and cache identity are shared, and async read and upload stage are held by the request |
| row_stage.h | Upload, decode and stable output borrow contract for prepared table rows; consumers keep the stage alive until async computation and graph replay end, and ordinary queries return a Tensor owner |
| bias.h | Projection weights/bias data without computation methods, and dense row slice preparation; reading and validation are provided by Loader |
| fields.h | Enumerates Tensor owners in weights, prepares transfers and counts bytes; backing is expanded only inside accounting, de-duplicated by actual storage identity |
| moe/, tq/ | Runtime representation, preparation, transfer and cache implementation for the corresponding weights |
| moe/experts.* | Queries expert cache and shard attributes through Tensor, and provides placement, format capability, prefetch and CPU staging contracts; it does not receive a whole-layer Ffn, and backends do not access cache-private objects to rebuild this behaviour |
| moe/pool.h, dispatch.h, prepare.* | Expert pool, activation geometry, mixed-encoding dispatch metadata and async upload backing lifecycle |
| moe/cpu_cost_model.h, cpu_calibration.h, dispatch_plan.h | Expert CPU cost, real hardware calibration, CPU/GPU work split and transfer quota |
| moe/cache_policy.h, ranker.h, routes.h, residency.* | Expert slots, replacement, routing history and residency policy |
| moe/profile.h | Expert profile data, field validation and explicit path/sidecar selection |

Weight preparation, residency, transfer and cache policy belong to the shared `engine/weights`; computation scratch and
device transfer primitives belong to the backend Executor/native. Backends do not create mirror directories per weight
or upper module.
CachePolicy defines the cache policy; the CUDA load configuration and the cache use the same type internally, while load
options and executor options are still provided by their own owners.
The load owner enters its Executor's scope, Cache takes the actual resources at the load-initialisation and
after-validation configuration points; Loader keeps the execution resources needed for lazy loading alive, and public
headers expose no concrete execution type.

Cache's interface is weight loading, cache preparation and lease operations; callers need not know cache fields,
initialisation order or budget fallback rules. `weights::detail` currently mixes resource ownership with storage
borrowing, and the target placement is in the table below. Names state the real responsibility, and private headers
plus C++ access control limit the consumers.

| Current name | Target name and location | Responsibility and consumers |
| --- | --- | --- |
| `detail::CacheData` | Private nested type `Cache::Resources`, defined in `weights/cache_resources.h` | Holds MoE/TQ caches, budget configuration, load execution resources and workspace; accessed only by Cache members and their transaction implementation |
| `detail::Workspace` | `PrefillWorkspace`, definition stays in `workspace.cpp` | Prefill backing, region records and reclamation accounting; owned exclusively by Cache, and the private resource header keeps only the needed forward declarations |
| `detail::KvOwner` | `KvLoanState`, definition stays in `workspace.cpp` | KV borrow registration and expert cache association; kept alive by Cache, `KvRegion`'s private implementation keeps a weak reference, and lease operations keep protecting and retiring the region |
| `detail::cache(Cache&)` | Deleted, no replacement whole-state accessor | Member implementation works on private resources directly, and collaborators such as Loader obtain the results they need through load/prepare behaviour |

`Cache::Resources` replaces the existing aggregate record; no new resource management class or per-field forwarding is
added. The public `cache.h` keeps only the forward declaration and owning pointer of that private nested type and does
not declare `detail::CacheData`; the whole object is not kept exposed through a `state()` / `impl()` accessor.
Single-file helper functions move into the anonymous namespace of their `.cpp`, and no `weights::details`,
`cache_detail` or similar miscellaneous collection is added.
The existing Loader constructor and MoE/TQ load paths are changed to obtain the resources they need through Cache's
assembly entry point. Lazy creation of the TQ cache, budget selection and resource registration are done by Cache, and
TQ loading handles table reading, shape validation and weight construction; load functions do not extract `moe`, `tq`
and budget fields and assemble the cache themselves. Collaborators receive only the dependencies they actually need,
following the existing preparation and loading contracts, with no per-field getter and no generic forwarding layer.
Dependency passing keeps the resource keep-alive needed by lazy loading, async readers and leases.

Read-only parameters and dense groups receive a Tensor owner; F32 conversion and concatenation products are kept alive
by a Tensor owner, and concrete representations are borrowed only inside preparation and numeric implementation.
The decode/prefill storage, leases and statistics of the expert borrow region are held by the weight cache.
Cache provides the transfer coordination method, the decode path and the prefill/routing decision separately; the model
keeps the top-k math and override values. Swap configuration belongs to Cache, Loan reports the actual transfer bytes,
and Budget coordinates space.
Swap completion, prefill storage retirement and predictor swap preparation are executed by Cache; swap completion is not
the same as draining every prefill copy or releasing the workspace.
KV and offload register real storage and protection boundaries through a KvRegion lease; retiring a region completes the
expert swap first, then waits for readers and uploads and revokes addresses.
Budget coordinates only binding and space requirements; the execution owner cleans up the borrow region after draining
consumers and before leaving the device scope.
source is an internal implementation of weights, and the model factory accesses data and prepares weights through a
single entry point. Final output parameters are loaded into each family's own members, and the interface carries no
empty placeholder resource.

Expert cache and numeric computation take the preparation result of `moe/prepare.*` as the seam, reusing the existing
pool, routing and lease representation, with no generic cache executor added. The input contains expert source
identity, selected routes and execution placement; the result provides the readable encoded fields/expert mapping for
this call, the work split and a keep-alive lease. The preparation result handed to the backend exposes no complete
cache class, slot ledger or mutable residency state.
weights decides residency, on-demand reading, the CPU/GPU work split and transfer preparation; the backend consumes the
preparation result and performs the math and reduction in the existing projection/ffn path. The computation in
`MoeCachedSource::forward*` moves into these backend implementations; the cache object does not accept activations and
computation scratch to return computed output, and does not call back into backend projection/ffn.
Preparation may use backend device copy and event primitives; the returned result carries ready dependencies, consumers
establish the execution order, and "preparation finished" must not be mistaken for all GPU work being finished. The
lease covers the last consumer and graph replay, and slot reuse, route commit, cancellation and swap retirement still go
through the owning cache contract.
Backends collaborate through a narrow prepare/prefetch interface and do not include `cached_source_internal.h` or the
complete private definition of `expert_cache`. The migration keeps heterogeneous expert execution, double buffering,
pre-quantized activation reuse, GLU fusion and cache hit paths; synchronous waiting for all uploads does not replace the
existing overlapped execution.
Lease retirement is verified through the preparation contract and the actual numeric consumers for cold/hot experts,
swap, cancellation and Graph readers, keeping the original numeric criteria and transfer overlap checks; directly
accessing cache-private members is not used as a cross-module test contract.

### models

```text
models/
├── CMakeLists.txt     # model implementation targets and source lists
├── source.*           # ModelSource's model-semantics contract and file format selection factory
├── source/            # format interpretation, canonical model view and their compile targets
│   ├── model_config.*, model_graph.* # configuration, structure and semantics description
│   ├── hf.*, hf_source.*, gguf.*, gguf_source.*, mfq.*, mfq_source.*
│   ├── legacy_graph.*, legacy_names.*, mlx_names.h # old names and layout interpretation
│   └── tq_fixture_source.* # model view for explicit synthetic input
├── causal_model.h     # model behaviour interface, capability and model state contract
├── metadata.h         # Info, public capability data and read-only model metadata
├── state.h            # model state interface plus shared position and speculation cursor
├── input.h            # Input, Prepared, ForwardOptions: forward input, preparation result and options
├── batch.h            # shared batch results; concrete batch factory declarations belong to each family
├── prepare.*          # shared preparation entry point, stop token parsing and family dispatch
├── load.*             # unified model factory and family dispatch, returning unique_ptr<CausalModel>
├── session/
│   ├── snapshot.h, state.* # model session composition, shared snapshot and Tensor enumeration
│   ├── codec.*, paged.* # codec selection, prefix identity and snapshot byte read/write
│   └── cache.*, size.h # model-side restore adaptation and snapshot capacity computation
├── common/
│   ├── capabilities.h # capability decisions for model implementations and optional components
│   ├── load.*         # shared weight assembly; the public header uses forward declarations to isolate device implementation
│   ├── components.h   # capability and accounting of assembled components, holding no cross-family resource
│   ├── prompt.h       # shared input preparation; row prefetch is executed by the model together with its State
│   ├── mlp.h          # model-held dense MLP weights and important-neuron branch; uses only the public Tensor
│   ├── ffn.*          # shared FFN composition of residual, normalization, MLP/MoE and trace; holds no execution resource
│   ├── mtp.h          # predictor behaviour contract; the generation entry point and advance implementation belong to generation
│   └── vision/
│       ├── grid_vision.* # grid configuration, geometry and position rules
│       ├── grid_forward.h # grid encoding, merge and input preparation templates
│       └── grid_encoder.* # shared grid encoder, private layer weights and forward
├── qwen35/
│   ├── config.*       # configuration interpretation and validation
│   ├── model.*        # CausalModel implementation, model definition and component holding
│   ├── weights.h      # model weight types organised by layer and role
│   ├── load.*         # weight name mapping, model assembly
│   ├── forward.*      # embedding, layer order, norm, hidden/logits computation
│   ├── attention.*    # Full/Linear Attention compute units and their math steps
│   ├── mlp.*          # MLP/MoE compute unit; whether to split files is decided by actual size
│   ├── state.*        # request/batch state, position semantics, Graph state access
│   ├── session.*      # model snapshot validation and joint restore
│   ├── batch.*        # model batch state and interface
│   ├── batch_ops.*    # batch state operations and model batch execution
│   ├── predictor.*    # MTP predictor definition and weights
│   ├── mtp.*          # prediction, commit, rollback adaptation
│   ├── mtp_state.*    # request-related mutable MTP state
│   ├── input.*        # special tokens, media positions and preprocessing rules
│   └── vision.*       # vision encoding, projection and text fusion
└── <family>/           # configuration, forward, state and adaptation of the other families
```

Models compose Layer, Attention, MLP and Encoder per family; compute units hold read-only weights and config and provide
forward. Gemma4's model, forward, loading and State use the same set of public contracts, and no old computation path
with an Executor parameter is kept; capability is declared by the family's actual implementation, and optional
component declarations in the graph are used for validation and accounting.
No unified base class, generic layer union type or one-file-per-unit rule is forced; only parts whose structure and
contract are genuinely identical belong to common.
Layer weight records do not carry request reset, speculation commit or Graph scheduling; forward and prefill math
compose concrete compute units directly, and layer cache reset, commit, rollback and warmup enumeration are provided by
LayerState. Normal requests, the predictor and batch reuse the same layer state definition, with no separate state
proxy holding compute unit pointers.
Optional components are assembled by the corresponding family; they are combined only when several real owners need
them, a single object returns its pointer directly, and no holding container is created when there is no object.
For example, a layer forward calls `attention.forward(x, state.attention, step)` and then `mlp.forward(x)`,
and the units call numeric interfaces such as `ops::project` internally; residual, normalization, gating and their order
are defined by the model.
Qwen35's linear recurrence and softmax attention use different numeric contracts and must not be forced into one
all-family attention by a flag.
KV, recurrent/conv, position and speculation state are held per request or batch; neither models nor compute units store
an Executor, live State or scratch. Layer forward and prefill receive the required input, position and commit
parameters directly, without a shared layer context or an unpacking forwarding entry point.
Graph model adaptation may stay in model.* or state.*; model semantics that need device adaptation still belong to the
corresponding family. Graph computation and checkpoints are organised by the graph driver, which coordinates batched FFN
chunking and deferred residual merge, while layer state keeps only its own layer cache. MoE routing and combination math
belong to the model and execute expert computation through public numeric ops; expert cache, residency and transfer
belong to weights and the backend. Neither the model nor the executor holds a whole-layer Pipeline, and no matching
registry or borrow handle is created. Qwen4's target Graph uses the existing TQ path and the predictor uses ordinary
forward; adjacent-layer routing prediction, diagnostics or graph budget entry points that depended on Pipeline are not
kept.
Models compute snapshot data volume and copy peak; the instance coordinates cache eviction and space reservation, and
generic generation code does not read family-private cache layouts.
The model decides the logical advance range of prefill; compute block size and space reservation are still bound by the
instance budget.
Cross-phase prefill exposes the actual encode and decode phases; the outer layer ends the weight cache policy at phase
boundaries, and the model is not given the cache manager through a callback.
The model estimates snapshot bytes and peak from the explicit request State; Cache releases prefill buffers before the
estimate, and Budget coordinates cache eviction and workspace reservation before the copy.
The shared scope, input/output and resource contract of the vision encoder is in
[Vision encoder design](#vision-encoder-design).

### backends

```text
backends/
├── dtype.h                # shared scalar type definitions, no device handles
├── cuda/
│   ├── executor/
│   │   ├── executor.*     # execution scope, resource reuse, completion tracking and draining
│   │   ├── config.*       # CUDA execution configuration, device availability and parallel placement initialisation
│   │   ├── options.*      # dynamic CUDA execution options and reading of that configuration
│   │   ├── batch_graph.*  # batch graph state, keys and invalidation
│   │   ├── decode_graph.* # capture, input transfer, buffers, reuse and invalidation
│   │   └── graph_registry.* # captured graph registration and execution lifecycle
│   ├── ops/               # attention, ffn, projection, norm, rope, vision, sampling numeric implementation
│   ├── kernels/           # .cu/.cuh kernels organised by op family
│   └── native/            # Tensor storage and device primitives, memory, stream/event, communication
├── metal/
├── hip/
└── cpu/
    ├── native/            # threads, memory copy, capability detection and timing
    ├── executor/          # CPU worker pool for expert row tasks
    └── ops/projection/    # host numeric computation for compressed weights
```

backend owns device execution and shares Engine, Scheduler and generation. Different devices are implemented according
to their actual capability; they are not required to have the same Graph mechanism or the same file list. The paths
below are relative to backends/cuda:

Each device's `backends/<device>/` carries only `executor`, `ops`, `kernels` and `native`; the root `backends/` provides
the shared `dtype.h` and the device directories. Device directories must not mirror the upper `engine` module
directories such as `requests`, `generation`, `scheduler`, `state`, `weights` or `models`.
Generic mechanisms belong to the same-named module in the outer `engine`, model knowledge belongs to `models`, and the
call site of a device API does not change the upper business ownership.

| File or directory | Responsibility |
| --- | --- |
| ops/attention/ | Numeric contracts and fused implementation of softmax, sparse attention over shared KV, recurrent update and similar; it does not organise a model's whole-layer Attention |
| ops/ffn/ | dense.* implements the MLP numeric path, routed.* implements route selection, expert scaling, reduction and routed GEGLU; the remaining activation, residual and workspace layout are organised by numeric responsibility, and storage reuse and keep-alive belong to the execution facility |
| ops/ffn/options.h | FFN numeric policy and per-execution policy override; not attached to whole-layer weights or a computation object |
| ops/projection/ | Projection of ordinary and id-selected weight banks, bias, embedding and encoding implementation; reading, slicing and weight preparation belong to weights |
| ops/norm/, ops/rope/ | Normalization, position table preparation, positional encoding and YaRN numeric implementation; the model holds the public Tensor table, position tables are prepared on an explicit device, and rotation position and input are co-located |
| ops/vision/attention.h | Numeric helpers for non-causal/segmented attention and axial RoPE; contains no complete encoder or Transformer layer definition |
| ops/sampling.* | Sampling, penalties, constraints and MTP probability computation; generation strategy belongs to generation |
| kernels/ | Kernels and prepack entry points; the required GGML host helper symbols are provided by native/ggml.cu |
| native/device.*, scope.h | Error checking, device/context selection and scope restoration; contains no model or instance management |
| native/tensor/, memory.*, allocation.h | Device memory, temporary kernel view, allocation and mapping primitives reused by each Tensor implementation; no device-specific copy of the Tensor inheritance tree and no residency budget handling |
| native/stream.*, graph.* | stream/event ownership and Graph capture, upload, replay and destruction primitives |
| native/copy_batch.h, dma_batch.h, vmm.* | DMA, virtual address and physical page mapping; weight cache, upper batch and KV mapping policy belong to their own modules |
| native/collectives.*, native/branches.*, native/profile.* | Peer access and communication, device execution branches and profiling |

backend ops own the algorithm, fusion, kernel selection and launch of already defined numeric operations, and use
workspaces and synchronisation primitives through the internal execution facility.
Projection grouping capability and merged prefixes are expressed by the public ops; the model receives no Projector or
profiler. Residual addition distinguishes clearly between keeping the input type and adopting the execution
configuration, and the serial branch policy captured by Graph is kept down to the internal nested numeric scope.
id-based projection keeps the route axis; the routing selection bias is passed in through the public topk, and the model
explicitly organises clamping, activation and reduction. Encoding format and parallel submission stay in the backend;
starting/stopping, reporting and exception recovery of temporary profiling are held by the backend scope; automatic
collection scope saves external state, reports on normal exit, and repeated compute segments restore the same collection
state.
Sparse attention metadata is owned by the backend and reused per device, stream and geometry; the Graph storage
enumeration covers keep-alive and identity checks for these buffers.
backend ops cannot receive a whole Model, branch by family, or call back into model forward; a prepared projection
object holds weights and derived data, and at execution time takes resources and options from the scope, capturing no
executor, profiler or policy reference. The Attention/MLP steps of the model stay in models.
kernels implement device computation and native provides device primitives; stream ownership and the current execution
choice are managed separately.

## Engine Instance Target Design

Engine receives CausalModel, Executor and instance options separately through its constructor; resource assembly is
described below.
The model factory returns `std::unique_ptr<CausalModel>` and the assembly entry point transfers ownership to Engine;
construction, rebind and Engine members use the same exclusive-ownership contract. generation and other synchronous
calls borrow a `CausalModel&` or `const CausalModel&`, do not copy model smart pointers, and add no model handle
forwarding layer.
Several requests sharing one model's computation and read-only weights do not create several model owners; borrows by
suspended coroutines and async consumers must end before the model is unloaded or replaced.

| Object | Ownership and behaviour |
| --- | --- |
| Engine | Exclusively holds the model and the primary Executor, holds the Scheduler and the resource owners, and coordinates instance lifecycle and budget |
| CausalModel | Composes compute units and holds read-only weights; provides capability, input preparation, State creation, forward/logits and batch/MTP/session behaviour |
| Executor | Manages execution scope, stream/event, workspaces, Graph execution resources and async completion; contains no model math and no numeric forwarding interface |
| Model State | Created by CausalModel and held by a request or batch; exclusively holds KV, recurrent/conv, rolling history, position and speculation transactions; layout, allocation requirements and restore behaviour are defined by the model |

A model forward looks for example like `model.forward(input, state)`, and compute units receive only the input, the
state they need and this step's math parameters.
The model reports actual growth and the graph capacity plan, and Engine/Budget coordinate reservation before execution;
the shared model does not store a budget callback of the request owner.
The generic `Input` provides no `attention_mask`; the concrete mask is handled inside compute units and ops.
No Executor is injected layer by layer in the constructor or in forward, and Projector, profiler, stream or scratch are
not extracted and passed to compute units.
A model is not bound to a particular Executor object; weights, state and ops must still satisfy device compatibility,
and switching backend does not mean the original device data can be reused directly.

| Engine public capability | Contract |
| --- | --- |
| submit | Receives an owned request and returns an output handle; a successful submit means the request is under management, not that it runs immediately |
| cancel | Marks and wakes cancellation handling; the internal loop completes cleanup and terminal state publishing |
| Request output | Bounded event channel with incremental results, usage and completion/cancellation/failure events |
| info/status, session/control | Capability and status queries, session and control operations; conflicts are coordinated by the internal loop |
| Unload, re-bind, shutdown | Quiesce requests, wait for device consumers, release or accept dependencies, and finish thread and resource wrap-up |

Engine advances execution itself. Admission and bounded steps are internal mechanisms, and the Scheduler calls
generation directly.
The device scope is established on the thread that actually executes; external threads submit requests and control
operations through the public interface.

### Model Assembly and Reload

The seam for model construction is in `models/load.h`, reusing `models::load` and returning
`std::unique_ptr<CausalModel>`. The caller provides ModelSource, resolved model options, the weight Loader and the
placement configuration; it provides no concrete family type, no model type callback and no Executor. `models/load.cpp`
selects the family factory from the model description using simple dispatch, with no new registration framework; the
family factory completes configuration interpretation, weight role mapping, model-specific option validation and
optional component construction.

The seam for library-level assembly is in `engine/load.h`, with load and reload implemented in `engine/load.cpp`, kept
separate from the runtime responsibilities of the `Engine` class. This module obtains the Source, prepares the weight
cache and execution environment, calls the unified model factory, configures request handling and public runtime
resources, and transfers ownership to Engine. Device resource creation and device limit validation are done by the
corresponding backend; model resource requirements are provided through the public model/state contracts. The model
factory creates no executor, and the assembly implementation reads no family-private layout.

The current `engine/load.h` publishes `internal::resolve_cuda_config`, `internal::setup_cuda_load`, `finalize_kv` and
`finish_load`. Runtime, evaluation and diagnostics call individual preparation steps from it, and the old model loading
templates also handle KV commit and cache wrap-up, so callers must understand the load order. The target load interface
encapsulates the complete flow, ownership transfer and failure cleanup; the responsibilities converge as in the table
below, and load phases are not kept public by renaming the namespace.

| Current behaviour | Target ownership and call contract |
| --- | --- |
| `resolve_cuda_config` | Runtime option validation and Config construction belong to the existing `engine/config.*` and receive the typed options they need; CUDA environment options are interpreted by the backend and then handed to assembly, so the generic configuration module does not depend on the whole `CudaEngineOptions` |
| `setup_cuda_load` | Draining and execution resource initialisation belong to the backend, weight cache initialisation belongs to Cache, and model-specific option validation belongs to the family; `engine/load.cpp` coordinates, and commands no longer clear executor fields or schedule these steps themselves |
| `finalize_kv` | KV space preparation and borrow registration follow the state, Budget and weights contracts, with the library-level assembly coordinating the commit; the family provides only the required state and resource requirements and does not finish global KV/expert budget wrap-up |
| `finish_load` | Cache finalize belongs to Cache, device work completion is guaranteed by the owning Executor contract, and load timing and reporting stay in the assembly implementation; keep the existing cache commit and wait semantics — the timing code alone is not moved out and the synchronisation is not deleted |

Complete load and reload publish to the caller only an instance for which the necessary preparation has finished; on
failure the assembly module performs the wrap-up, and the caller is not responsible for calling some finish function.
Helper implementation used only by assembly stays in `engine/load.cpp`. If evaluation and diagnostics only need to
validate configuration, weight preparation or a single model, they use the interface of the corresponding
responsibility directly; they neither depend on load-private headers to reuse one step nor are forced to create a full
Engine. After migration, shared declarations follow the actual consumers, reuse the existing config, Cache, Executor and
model factory, and add no generic load-phase framework.

`apps` only starts commands; `commands` parse CLI strings into typed options, call the library-level load entry point
and compose Transport. Neither contains a concrete family's model/load/forward/state header, and neither maintains a
backbone switch, a model `variant`, a `with_model` template or a Gemma4 special case. Generic availability checks for
batch, MTP, multimodal and session consume the public capability contract; concrete model limits are validated by the
family, and the application layer does not infer capability from model names.

Engine receives the model, Executor and public resource owners through construction or through rebind while quiescent,
and does not parse model paths or hold load callbacks. The code below only illustrates the assembly relationship inside
`engine/load.cpp`; concrete parameters reuse existing types and no generic dependency container is added:

```cpp
auto executor = make_executor(device_options, execution_resources);
auto source = open_model_source(model_path);
std::unique_ptr<models::CausalModel> model;
{
    auto scope = executor->enter(); // device work during loading also uses its owning execution environment
    auto weight_loader = weight_cache.loader(*source, weight_options, placement);
    model = models::load(source, model_options, weight_loader, placement);
}
auto engine = std::make_unique<Engine>(
    std::move(model), std::move(executor), engine_options);
```

The factory returns a CausalModel, having completed the assembly of configuration, weight roles and component
definitions; live State is created when a request or batch is established.
Concrete State, MtpState and batch objects are implemented by the family against the public construction contract, and
callers do not combine each family's `Model/State/Components` types. Device warmup, Graph creation and timing belong to
the backend's execution contract and are not special-cased by model type in an application template.
The pseudocode omits the explicit transfer of root resources; the holding relationships of weights, readers and device
resources must cover all borrowers.
Load failure cleans up objects that were not transferred and device work, and publishes no usable instance.

reload is coordinated by the same library-level assembly entry point: Engine stops accepting, cleans up requests and
waits for device consumers, and releases Graph, batch, session, model state and execution dependencies; the assembly
entry point then constructs new dependencies, and Engine validates, binds and resumes accepting.
By default old resources are released before loading, so two models are never resident at once; on failure the instance
stays unloaded and an error is returned.
Requests, state and Graph are never reused across dependency bindings, and the Scheduler is always managed by the same
Engine.

Changing the weights, forward, state or optional components of an existing family changes only that family and the
modules whose shared semantics genuinely change; adding a family adds its own implementation, its dispatch in
`models/load.cpp` and its models build list, and does not change apps, generic runtime commands or an Engine type list.
New product commands or new public model capabilities are changed according to their real responsibility, and no
universal option table is designed to hide necessary changes.
The current unified factory covers only Gemma4; other families are still loaded by the old templates in `models/load.h`.
The migration must keep those families' configuration, placement, warmup, MTP and multimodal capability, have them
implement the same public model contract, and only then delete the old template path. "Unified factory" must not be
implemented as a reduction of the support scope, and the choice between old and new paths does not move into apps.
Generic evaluation and whole-model diagnostics also use models through `models::load` and CausalModel, without depending
on the `with_model` template. Diagnostics that genuinely need concrete layers or private state live in the corresponding
diagnostics/test file and may use family-internal seams; commands only select the diagnostic task and do not carry its
model loading and computation details.
Loading, capability and state creation of each supported family are verified through the unified factory, keeping the
existing numeric cases; load failure cleanup, unload and cross-family reload are verified through the library-level
entry point. The model and load library should build without compiling apps and be callable from tests; these are
migration acceptance requirements, not a claim that the current source already meets them.
Load convergence must also verify that: the instance is published only after cache commit and the necessary device work
have completed; any failed preparation phase completes wrap-up of the resources already created; and configuration
diagnostics can run independently through the config interface. Keep the existing numeric and resource lifecycle
criteria, and do not use the call count of private phase functions as an acceptance criterion.

### Public Ops and Backend Dispatch

Public ops define the math semantics, shape/dtype, output and aliasing, explicit state modification and error
contract, and their parameters contain no Executor.
Models and ops read data through Tensor references, hold and return results by Tensor value, and callers never
dereference smart pointers; ordinary ops dispatch by input device, and weight ops combine layout, execution placement
and actual residency to check compatibility.
The backend selects an implementation by shape, logical dtype and storage encoding; the existing host computation,
staging, sharding and on-demand expert paths are kept, and the execution backend is not inferred from CPU backing.
Ordinary ops validate device compatibility of input, weights and state; cross-device operations such as copy state the
source and target explicitly, and creation ops with no input specify the device explicitly.
Simple dispatch comes first, with no generic registration system; each public call selects a backend once, and internal
kernels do not repeat cross-backend dispatch.
Unimplemented combinations are rejected explicitly, with no implicit data movement and no CPU fallback. A tensor's device
cannot decide the stream; the stream is chosen by the execution scope.

Model Attention organises steps such as QKV, QK norm, RoPE, gating and output projection; the public attention op
expresses only an explicit numeric contract.
A fused op may cover several steps, but it must state its compute range, keep residual, dtype conversion, accumulation
order and the original numeric criteria, and it does not select an algorithm by model name.
requests/request.h, models/input.h and op inputs express request, model forward and numeric operation respectively;
public interfaces expose no CUDA private type.
Concrete models organise forward and replay directly, and Executor does not interpret slot-arranged Task/Inputs; no
public Task IR, task list forwarding shell or second op entry point is kept.

### Executor and Execution Scope

Executor coordinates the execution resources below, reusing native, workspace and Graph implementation internally; it
does not receive an Attention object to organise a whole-layer FFN. Model layer order, request scheduling and weight/KV
management each stay with their original owner.
Executor queries byte snapshots of the device and allocator, and Engine and each resource owner aggregate weight, KV,
cache and budget statistics.

| Execution responsibility | Contract |
| --- | --- |
| Current execution environment | On entry it saves and activates the device, stream and execution resource bindings on the thread; on exit or exception it restores them, and backend ops borrow through internal entry points |
| Branching and dependencies | Ops decide which numeric steps may run in parallel, and the execution facility provides stream/event, fork and join; it does not choose which request runs |
| Compute workspace | Ops define layout and capacity requirements; the execution facility owns reuse, parallel isolation and reclamation after the last consumer ends, and does not take over weight or KV cache |
| Completion and draining | Tracks submission and completion, keeps Tensor backing and borrow leases alive; after cancellation it stops further submission and wraps up, and drains before unload/reload |
| Graph and helper resources | Manages capture/replay resources, fixed buffers, compute library handles and optional profiler; the model provides state conditions and generation decides when to use Graph |

scope is a temporary binding on the current thread, not a resource owner; it stores no "current model/request/State"
and sets no process-level current Engine.
Independent projection execution policy travels with the scope, not attached to a weight list; storage packing and
execution grouping options are interpreted by the load and execution facilities respectively.
Reuse the existing ContextGuard, StreamGuard and similar mechanisms, with resources kept alive by their owning executor
or lease, and the old CudaExecutionContext is not renamed and moved in wholesale.
A scope covers the host execution interval of one continuous block of submitted work: usually one decode round, one
batch or one prefill chunk, and it may contain several MTP forwards.
It is not fixed to one token, does not cover a whole request or wait period, and does not rebuild the stream on each
entry.

Suspending a coroutine does not destroy its local variables, so the scope must be established outside the coroutine
resume and must not be a local object inside the coroutine that survives a suspend point:

```cpp
{
    auto scope = executor.enter();
    sequence.resume(); // this run up to the next suspend or completion; forward receives no executor
}                      // restores the thread environment, implies no GPU sync
```

Re-enter the owning scope on every resume or when work moves to another thread; synchronous paths wrap the actual
submission interval by the same rule.
Leaving the scope does not mean the GPU is finished, and does not reclaim a workspace, output backing, state view or
weight lease that is still in use;
Tensor results may be returned before device completion, cross-stream consumption establishes a dependency, and host
reads and resource retirement wait on a completion event or on drain.

## Model Files, Weights and Model Semantics

| Interpreted or managed object | Sole owner |
| --- | --- |
| File containers, raw metadata/assets, tensor names and byte ranges | weights/source |
| Bit layout of numeric encodings, scale/codebook, block/row rules | weights/encoding |
| Weight preparation, placement, transfer, residency, on-demand expert cache and lease | weights |
| Model configuration, graph, weight roles, name mapping and model-specific layout | models |

The model factory reads configuration, assets and tensors through weights; Source provides raw metadata, byte ranges and
bounded readers, and ModelSource interprets model architecture, names and graph. Model interpretation belongs to
mfq-model-source, raw reading and preparation belong to mfq-weight-source.
source reads the shape/dtype/encoding identifier recorded by the container, encoding interprets the encoding, and
models decide weight roles and model layout.
Initial load and on-demand load share the reader, file and mapping holders, and a whole quantized tensor is not
expanded by default; configuration, tokenizer and template share the parsed result.
A pure encoding layout may be used by source and ops/kernels, without depending on model or cache management objects.

### Unified Tensor Contract

Following [PyTorch's separation of handle and implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorBase.h),
use two layers, Tensor and TensorImpl: Tensor is an instantiable owning handle that encapsulates implementation
ownership and common metadata queries, and is the tensor type used by models, state and public ops.
Tensor internally holds a shared TensorImpl, reusing `std::shared_ptr` first, with no custom reference counting for
this. Public data flow uses Tensor values and references; it does not expose `shared_ptr<Tensor>`,
`shared_ptr<TensorImpl>`, nor require callers to use `.get()`, dereference or `make_shared`.
TensorImpl takes over the responsibilities of the former abstract Tensor and is a representation interface with a
virtual destructor; it promises only the logical shape/dtype, layout identification, storage enumeration/access and
lifecycle contract common to all implementations. It does not take part in data passing on the model side, and the
creation and access of concrete representations stay inside loading, storage and op implementation.
DenseTensor represents dense data; TqTensor, NintTensor, NvqTensor, MxTensor and others inherit TensorImpl by data
representation, with format fields and layout validation in each implementation, reusing weights/encoding. Backend
internals may keep using concrete representations directly, and not every kernel call has to be wrapped in a public
handle.
Concrete implementations keep reusing TensorStorage's buffer ownership and backing identity; this handle design does
not change the shared storage mechanism. TensorImpl does not pile up format fields, and derived classes do not hold a
complete QuantLinear/Experts, router, loader or execution manager.
On-demand TQ experts use TqBank to store source identity, logical shape and shards; selected images are kept alive and
accounted by the weight cache and lease, and numeric computation belongs to CUDA ops.

Holding objects use Tensor values, reading input uses `const Tensor&`, in-place modification uses `Tensor&`, and
results are returned by value as Tensor; async consumers store a handle copy when they need keep-alive, and a borrowed
reference does not extend lifetime.
Copying a handle shares the implementation and backing and does not copy data; assignment replaces the object held by
the current handle; view uses a separate metadata object and shares backing; clone creates independent data, and no
copy-on-write is introduced.
A default-constructed Tensor is empty and `defined()` returns false; optional bias, position and similar use an empty
handle, and required inputs are validated by the op. A valid tensor with zero elements is not the same as an empty
handle.
Weights and other data use the same Tensor; no ConstTensor or read/write permission type is added. Published weights and
their aliases stay read-only by calling convention; `const Tensor&` does not mean the underlying data is globally
immutable and does not stop other handles from modifying shared data.

```cpp
Tensor hidden = ops::affine(input, weight, bias);
hidden = ops::gelu(hidden);
std::vector<Tensor> outputs = ops::project_group(hidden, weights);
```

| Data | Logical representation | How ops use it |
| --- | --- | --- |
| Activations, norm, bias, KV/recurrent payload | DenseTensor or a compatible encoded implementation | Handled by public ops, model State organises the state semantics |
| Ordinary/quantized weights | Logical `[output dim, input dim]` | project, embedding and similar read through Tensor references |
| Expert weights | Logical `[expert count, output dim, input dim]` | Expert ops receive routing explicitly; mixed encoding, sharding and on-demand reading are organised by the concrete representation |
| Grouped weights | Ordered Tensor handle list or named model fields, read-only by convention | project_group returns Tensor results in the same order; joint projection and physical packing are kept |

The shape/dtype of every implementation states logical numbers, and nbytes states the logical expanded byte count; real
storage accounting enumerates all numeric, scale/codebook and similar buffers and de-duplicates by backing identity.
stride, data_ptr and the directly accessible TensorView belong to the dense implementation or to a constrained view; the
public handle promises no naked-pointer access for arbitrary layouts, and models do not interpret encoded payloads.
Device copies keep the encoding and cover all auxiliary data; cast, reshape, slicing, indexing and grouping define their
layout, aliasing and materialisation rules item by item, and unsupported combinations are rejected before modification
or commit, with no implicit whole-model dequantization.
Execution placement and dynamic residency are expressed separately, and shard validation covers every partition;
TensorImpl derived classes are divided by data representation, and device differences are handled by the backend.

Numeric entry points are unified in public ops, which select an existing backend algorithm by device, logical dtype and
layout; Tensor gains no op method collection, TensorImpl carries no computation virtual functions, and backend calls
native/kernel directly, avoiding a call back into public ops.
Pure metadata/CPU operations need no CUDA scope; CUDA submission has its owning Executor scope established by the
assembly, load or scheduling entry point, and Tensor never creates an Executor or stream itself.

Loader creates concrete implementations and publishes them through Tensor handles, and the model provides names, roles,
group order and shape requirements; all aliases obey the weight read-only convention.
router, bias, shared gate and hash tables are published through Tensor handles as well; the same parameter is not stored
twice as a dense value and an encoded projection, and reading a concrete representation stays inside the op
implementation.
A Tensor handle keeps all of its backing alive through the implementation object; readers, mappings, residency caches,
prefetch and transfer leases belong to weights, computation scratch belongs to the execution facility, and on-demand
experts provide the backend a readable view for this call plus a lease.
Handle/implementation lifetime and device completion are managed separately: the lease covers the last consumer, output
is moved out before borrowed storage is returned, and load failure, cancellation, unload and reload follow the same
retirement rule.
Cache and Graph reuse according to actual backing, lease and storage identity, not by Tensor object address or equal
shape, and no global identity registry is added.

## Full Execution Chain for One Request

1. **Submit and queue.** Engine returns the request output handle; the Scheduler registers the request and admits it by
   priority, capacity and deadline.
2. **Prepare input.** requests completes validation, template, tokenize and constraint preparation, and calls model
   input handling.
   Tokens, media positions, embedding and context usage stay consistent, and expensive preparation may advance in
   segments and respond to cancellation.
3. **Prepare state.** The model creates request or batch State; after matching text and media identity it restores
   session/prefix, or initialises new state.
4. **prefill.** The Scheduler selects a chunk or batch; generation calls model forward inside the execution scope,
   consumes the non-reused prompt, updates State, and obtains sampling scores for the required positions through the
   model-side logits projection.
5. **Sampling and decode.** generation selects a token, then repeatedly feeds the generated token to the model and
   computes the next result;
   MTP and Graph are enabled by capability; each advance re-activates the execution environment, and model compute units
   complete the numeric work through public ops.
6. **Output.** requests interprets tokens, handles stop conditions and forms incremental results; the Scheduler
   publishes to a bounded channel and pauses further work under backpressure.
7. **Wrap-up.** Normal completion, cancellation and failure all finish the session save policy and device/request
   cleanup, then publish the single terminal state.

The first generated token comes from the logits at the last prefill position; already output tokens are normally
consumed in the next forward.
State positions are committed for the inputs that actually completed, and MTP accept, reject and replay follow the same
rule.
One internal advance may handle a media segment, a prefill chunk, several tokens or a wait for batch results; it is not
fixed to one output token.

## Text and Multimodal Input/Output

| Stage | Owner |
| --- | --- |
| External messages, media fields and order parsing | transport; requests/media.h keeps the input correspondence |
| JPEG/PNG, audio and video byte codecs | Generic tools in components |
| Template execution, tokenize, generic grammar constraints, incremental result parsing | requests, reusing components/tokenizer |
| Scaling, cropping, normalization, patch/grid, audio features and time/position rules | models/<family>; genuinely shared parts go to models/common |
| Media placeholders, special tokens, vision/audio encoding, projection and text fusion | models/<family>, computation completed through public ops |
| Model-specific output decoding and protocol semantics | provided by models; requests organises results, transport encodes and transmits |

Model input/output rules are provided through a public contract, and requests and the Scheduler maintain no per-family
dispatch table.
The above is the target division of responsibility; the support scope of input types and media capability is declared
by the model's public capability.

### Vision Encoder Design

The outward module is the concrete vision encoder, with an entry point such as `encode(input)`, passing state
explicitly when there is state; each encoder defines its own input and result contract.
Layer weights, attention/MLP composition and residual order are internal to the encoder; there is no generic
VisionBlock, vision::Layer, vision base class or callback assembly table. Engine organises input preparation through
CausalModel and does not select a vision encoder directly.

| Module | Ownership and shared scope |
| --- | --- |
| SigLIP encoder | `models/minicpmo45/vision.*`; keeps separate Q/K/V, GELU tanh, padding mask and this model's patch/position rules |
| RADIO encoder | `models/nemotron_h/radio_vision.*`; keeps the fused-QKV layout interpretation, GELU erf, CLS/register token and position interpolation rules |
| grid encoder | `models/common/vision/grid_encoder.*`; reused by the existing grid models, owning the composition of grid layer weights, positional encoding and segmented attention |
| axial patch encoder | `models/deepseek_v41/axial_patch_vision.*`, `axial_patch_encoder.*`; configuration, geometry and encoding implementation belong to the family actually in use, keeping RMSNorm, axial RoPE, gated MLP and the existing spatial alignment computation |
| Adapting vision output to text input | The corresponding `models/<family>/vision.*`; organises resampler/merger/projector/aligner, completing output selection, media placeholders and text fusion; encoders with the same numeric model belong to common by actual reuse |

Sharing is limited to the existing `pre_norm_layer`/`mlp` helpers, backend numeric ops, and encoders that actually use
the same structure and contract.
No common Block is wrapped again just because the pre-norm layer order is the same. CUDA model forward stays in models,
and `_cuda.*` may distinguish device implementation.

**Input and output.** Input is preprocessed pixels/patches plus the grid/mask that this encoder needs; shape, dtype and
valid region are explicit.
Cross-module input and output use public Tensor handles, segmented encoding publishes results by value and keeps
suspended input alive. Output states feature width, patch order and CLS/register retention rules; axial patch output
already includes spatial alignment and text-width projection, and callers do not project again. Family adaptation
produces the media embedding of `[N_media, D_text]` (N_media is the media token count), and fuses text after validating
the count, order and placeholder range of each segment. The encoder does not interpret prompt tokens or modify text KV.

**Execution and ownership.** The encoder holds read-only configuration and weight handles, computes through public ops,
and neither injects nor extracts Executor/Projector.
Expensive encoding has a scope established by the outer layer at each StepSequence resume; activation, progress and
result belong to the request or coroutine, and the media cache has its lifecycle managed by state with identity defined
by the model.
The encoder holds no primary executor and no active request, and does not operate MoeExpertCache directly. Weight
preparation and lease belong to weights, computation scratch belongs to executor;
when underlying storage is reused, the result must be moved out before return, the lease covers the last device
consumer, and yield/cancellation/failure follow the same lifecycle.

**Weights and ops.** The model defines QKV roles, layer weight enumeration and mask/position rules; weights provide
generic weight representation, field enumeration, reading, slice preparation and transfer, without depending on model
layers. Projections with bias belong to ops/projection, shared by vision, audio and TTS; norm, attention, RoPE and GELU
belong to their op families. Weight data and computation are separate, and `load/rows/forward` are not wrapped in a
vision class.
Projection keeps two computation contracts: finish dense+bias in the weight dtype and convert back to the input dtype,
and add bias in the projector result dtype.
GELU tanh/erf, softmax precision, attention output dtype, mask, contiguous behaviour and the dense slice limits of fused
QKV all stay unchanged.

## State, Weights and Lifecycle

Active requests and batches own model State, state manages the generic payload, weights manage files and weight
residency, and the backend Executor/native manage execution workspaces and device resources. Engine holds the actual
owners and coordinates dependency order.

- Request logical state is independent; when physical storage is shared, the identity of request, slot, page and
  allocation generation is kept.
- Graph, scratch, state view, transfer backing and weight leases live until the last device consumer ends.
- session/control, budget adjustment and dependency replacement are coordinated by the internal loop; operations that
  need quiescence wait for conflicting requests and device work to end.
  Cache manages MoE workspaces and borrow policy, Budget coordinates KV growth and the RAM reservation of KV offload;
  non-owning bindings are released before the consumer retires, and Budget and expert space stay alive until cleanup
  ends.
- shutdown stops accepting, cancels and cleans up requests, waits for device consumers and execution threads to exit,
  and then releases resources in dependency order.
- Every successfully submitted request has exactly one terminal state; output tail and cleanup results are handled
  first, and errors distinguish request failure from instance unavailability.

### Session, MTP and Graph

| Capability | Owner |
| --- | --- |
| Session/prefix storage, matching, budget and eviction | state/session |
| Shared KV snapshot, encoding and restore | models/session/state.* |
| Model snapshot byte read/write tools | models/session/paged.* |
| Current session composition and codec selection | models/session/snapshot.h, models/session/codec.* |
| Family snapshot layout, encoding and validation, complete-boundary decision, target/predictor joint restore, position semantics, and enumeration of held Tensors | models/<family> |
| MTP draft/verification/acceptance policy and generation advance | generation/mtp |
| CUDA Graph capture, cache, replay, invalidation and resource keep-alive | backends/cuda/executor/ |
| Model capture conditions, state access and restore | models/<family> |
| CUDA Graph low-level handle operations | backends/cuda/native/graph.* |

session restore prepares, validates the candidate and waits for device consumers first, then commits; on failure it
keeps consistent restore or reset semantics.
The KV, conv, recurrent, position and MTP hidden payload of a session snapshot are kept alive through Tensor handles and
are read-only by convention; restore results and the generation entry point keep using Tensor values, and capture keeps
clone isolation.
A mixed snapshot stores its layout marker separately and carries no backend runtime state object; the concrete codec and
restore implementation validate the representations they actually support.
A model block snapshot is responsible for its own layout and enumerates its held Tensors through `tensors(visitor)`.
Caches consume the traversal result, de-duplicate by actual backing identity, count the physical bytes of shared
storage, and do not add alias views twice.
The Tensors returned by complete and prefix restore are taken back by the owning State; compatible layouts reuse backing,
and growth returns a new owning handle.
MTP recurrent/conv state, position and the necessary hidden data follow the same accept and rollback rules. The predictor
creates its MtpState after assembly is complete and then finishes the KV budget; unload retires Graph and request State
before releasing predictor weights.
The predictor receives Tensor references through models::Mtp, handles optional position in the same step and returns
Tensor handles by value; chunked draft and advance-loop history and restore results use the same ownership contract, and
reading a concrete representation stays inside the implementation.
Graph takes the capture stream and its owning Context first, then writes its members; the budget check after upload uses
the held Context.
Graph borrows address-stable Tensor objects inside its owning State and holds the real backing; reuse depends on batch
layout, state/scratch addresses and storage identity, and warmup isolates production state.
slot/page reuse, batch changes and state restore trigger the corresponding synchronisation or invalidation. generation
coordinates when a graph is used and released.
A Graph result handle keeps the fixed output buffer alive but does not freeze its content; a result that must survive
later replay is cloned by the consumer that actually needs it.

## Numerical Correctness Verification

The inference library, native executables and their distribution packages do not depend on Torch/LibTorch. PyTorch is
only an optional dependency of the offline reference generator, running in a separate development environment; running
and comparing the fixed baseline needs no Torch installation.
No second Torch execution branch for producing reference answers is kept in models or ops. Numeric verification depends
on native public interfaces, and runtime modules do not depend back on the verification module.

### Modules and Interfaces

The offline verification module takes "cases plus fixed reference results, native execution entry point" as input and
returns a comparison report. Cases and result files form the cross-process seam; the reference framework and the native
program run the same case separately and never link to each other.
Tests run through the existing interface of the layer under test: ops use public ops, compute blocks use model compute
units, complete models use model/Engine entry points; dedicated kernel tests may test a kernel directly, but that does
not replace upper-level composition tests.
Case reading, result alignment, error judgement and reporting are centralised in the verification module, and callers do
not re-implement tolerance rules. The file contract does not require a generic executor interface, plugin registration
or a second model scheduling system.

```text
Fixed case (inputs, weight identity, compute config, observation points, error criteria)
    ├─ Standalone reference generator → fixed reference results
    └─ Native test/diagnostics → actual results
                          │
Fixed reference results ──┴─→ Comparison report
```

| Location | Responsibility |
| --- | --- |
| `tests/numerics/` | Case and result contract, standalone reference generation scripts, fixed samples, comparison and reporting; the reference framework is imported only in the corresponding generation script |
| `csrc/tests/`, `engine/backends/<device>/tests/` | Reuse the existing test entry points to run native op, state and backend cases; simple ops keep an independent CPU math formula as reference |
| `csrc/diagnostics/` | Reuse existing diagnostic capability to run larger compute blocks and model cases and export the agreed observation points; reuse library-level loading and the model factory, depend on no apps implementation, and add no test parameter to production requests |
| `tyloinfer/evaluation/`, `csrc/eval/` | Keep model quality evaluation and the existing reference data capability; file identity, token and export helpers are shared by actual reuse and are not called by the inference library |

The reference generator uses PyTorch ops, upstream model implementations or simple CPU math implementations; each case
states its reference source and support scope, and rewriting every model is not required. An independent numeric
comparison must not produce expected values by calling the native op, kernel or the same decode implementation under
test.
Comparison between native paths is implementation-consistency verification. The existing `reference_cuda.py` calls
TyloInfer native programs and can serve as a regression or quality evaluation baseline, but it is not an independent
numeric reference just because it is named reference.

### Case and Result Contracts

A case fixes the actual input data, not only a random seed; model cases fix token IDs, positions, mask semantics and
initial state, and multimodal computation fixes the preprocessed input. Different implementations use the same input
and do not re-sample or re-tokenize on their own.
A case records op/model configuration, input and weight content identity, dtype, layout and the necessary step order,
and defines the outputs and state observation points aligned by name. Complete weights may be referenced externally by
content-validated value, and small samples are stored with the test.
Reference results also record the generating script and reference model version, framework version, device and the
related compute settings, including accumulation precision, TF32 and approximate algorithm choices; native results
record build version and execution configuration. Both sides must satisfy the numeric semantics of the same case;
identical hardware or optimisation algorithms are not required.
Reuse the existing JSON metadata and contiguous binary Tensor export, stating format version, name, shape, logical
dtype, file encoding and byte count. A low-precision result may be losslessly expanded to F32 for export, but the
original compute dtype must be recorded; a high-precision reference keeps its precision.
Results are exported in the agreed logical layout, and cases involving stride, slicing and offsets still execute in the
original layout. Wait for the related device work to finish before reading host results; observation does not change
the actual compute dtype, state advance or Graph path.
The existing `_logit3_` is an evaluation format that compresses and truncates part of the log-probability and stays in
use for its intended purpose; strict numeric comparison of ops, hidden and logits uses results without additional
lossy encoding, and exact reference values cannot be recovered from that format.

### Criteria and Coverage Levels

Comparison first validates case and asset identity, observation point completeness, shape, dtype and layout convention,
and then compares numbers. Missing output, incomplete files or incompatible configuration are failures, not a numeric
pass.
Integers, indices and results explicitly required to be exact are compared element-wise for equality. Floating point
results are judged by `abs(actual - expected) <= atol + rtol * abs(expected)` with tolerances fixed in advance in the
case; tolerances are set per op, dtype and accumulation/approximation semantics, existing numeric criteria are kept, and
one loose tolerance does not cover all models.
NaN fails by default; infinite values that are mathematically allowed must be stated in the case and checked for
position and sign. The report gives the failing observation points, the count of out-of-tolerance elements, the maximum
absolute error with its position and the actual/reference values; relative error or other aggregate metrics are
supplementary only and do not hide element-wise failure.

| Level | Comparison and coverage |
| --- | --- |
| Op | Independent formula or framework reference; covers the dtype actually supported, non-contiguous input, chunk tails and numeric extremes, with the high-precision reference computed from the same already-rounded input |
| Compute block | Fixed weights and input, comparing the agreed observation points of attention, MLP, residual and state update, locating the first deviation |
| Model | A fixed token sequence fed step by step, comparing hidden, logits and the necessary state; text agreement under free sampling is not a primary numeric criterion |
| Execution consistency | Chunked vs whole-segment prefill, prefill/decode, Graph vs ordinary execution, snapshot restore and similar for the same native computation should be equivalent within the agreed tolerance; such tests supplement the independent reference, they do not replace it |

Quantization verifies two separate things: whether the computation is correct for the same packed weights decoded
independently by the quantization kernel, and the quality loss of the quantized model relative to the original weights.
The latter uses evaluation metrics such as perplexity and log-probability, and a quantization error budget is not used
directly as the implementation tolerance of a kernel.
Performance measurement and correctness reporting are separate; the time cost of an observation export path does not
represent production performance, and a performance gain does not offset a numeric failure.

### Baseline Maintenance and Build Constraints

Daily regression reads fixed reference results, and the implementation under test is never allowed to rewrite expected
values automatically. A reference update must come with its generation source, configuration and reason for the
difference; adding dtype, shape or model capability adds the corresponding sample and criteria.
Fixed samples cannot cover every input, so the standalone reference environment is also used for new cases and random
differential testing; a random failure saves the actual input for reproduction. The comparison module itself is verified
with equal, out-of-tolerance, missing-output and non-finite value samples.
The report distinguishes pass, fail and not executed; a missing device, model or optional reference framework cannot be
recorded as a pass, and comparison against fixed results does not depend on the reference framework. GPU cases run in a
verification environment that has the corresponding device.
Native build and packaging do not look for or link LibTorch; the reference framework and model dependencies belong only
to the separate verification environment. Backend source is selected by CMake targets, and `TYLOINFER_NATIVE_CUDA` is
no longer used to switch native/Torch math paths inside the same production implementation.
Before removing the old Torch reference targets and conditional branches, first keep the cases, numeric criteria and
result observation capability that are still valuable, and confirm that the independent reference can be regenerated and
the fixed regression can run; checking source strings alone, or keeping a same-source comparison, is not a substitute for
these tests.

## Product Assembly and Dependency Direction

apps start commands; commands parse product arguments, call `engine/load.*`, and then compose Transport.
Loading and reload reuse the library-level assembly entry point, which calls the unified model factory and the backend
resource entry point and then calls Engine's lifecycle interfaces.
Inference requests in Transport use only Engine's public interface.

- requests, scheduler and generation use public behaviour contracts and depend on no concrete model or device-private
  implementation header.
- `engine/load.*` depends on the public model factory, weight and state contracts and the backend resource creation
  entry points; only `models/load.cpp` includes the loading declarations of each family. Models, the runtime library,
  evaluation and diagnostics do not include `apps/*` in reverse, and commands do not take over model dispatch moved out
  of apps.
- models compose compute units and use public ops, Tensor, weight and state contracts; model semantics and device
  adaptation belong to the family, and Executor is not referenced.
- Public ops dispatch to device implementations; backend ops receive weights, input and explicit state views and borrow
  resources from the current execution environment, and kernels/native provide device capability.
- File source and encoding implementation depend on no concrete model, and device primitives depend on no request
  scheduling or Engine driver.

The runtime core, model implementations and device support are built according to real dependencies, the final inference
library combines `engine/load.*` with these implementations, and the application entry point only links and calls the
library;
public headers do not implicitly pull in a concrete family, and a compile target is not added mechanically per
subdirectory.
The model source list belongs to `models/CMakeLists.txt` and the backend source list to the corresponding backend, and
the final composition target does not compile the load library from `apps/load.cpp`. `apps/cuda.cmake` defines only the
entry point and link relationships, and adding a family does not change that file.
Source, CMake, JIT/include, tools and packaging must end up consistent; the current implementation order and deferred
scope are authoritative in the plan.
The three CUDA executable entry points are in `csrc/apps/`, and command, evaluation and diagnostics implementation are
in `csrc/commands/`, `csrc/eval/` and `csrc/diagnostics/`. main starts commands through `commands/run.h`, and argument
dispatch belongs to commands.
Device and model each have a CMakeLists.txt, and instance assembly is composed by engine/cuda.cmake; evaluation,
diagnostics and entry points are defined by eval/cuda.cmake, diagnostics/cuda.cmake and apps/cuda.cmake respectively.
The optional Torch reference implementation is composed by a library-level target, and apps only link that library.
The single source root of CUDA kernels is `csrc/engine/backends/cuda/kernels`, shared by native/JIT/wheel.
The Python package and CLI are `tyloinfer`; the service entry point is in [pyproject.toml](../pyproject.toml).
Architecture changes keep the model math, dtype and numeric validation requirements; offline quantization and
calibration product entry points are outside this architecture.
