# TyloInfer Refactoring Plan

The target architecture is authoritative in [architecture.md](architecture.md). This document lists only the
remaining work, deferred items and acceptance conditions; it keeps no completed tasks and no per-commit progress.
**Close the Gemma4 production call chain first — load, prefill, decode, cancel, reload and unload — then finish
Phase 1; after that remove execution dependencies and live State bindings from models, then extend Qwen35 and
verify it end to end.**
Production CMake wiring required by Gemma4 is kept in sync with the source. File consolidation for CUDA and the
necessary common code is handled item by item in section 4; Metal/HIP and its old-directory cleanup, tool wiring,
test migration, and build and runtime acceptance remain deferred.
Nothing is compiled, run, or changed in tests right now; only static cross-checking is done. Phase 1 source is not
complete; interface inheritance and constructor injection do not mean callers have finished migrating. Source,
build, numerical and full-chain acceptance are judged separately.

## 1. Converge the public Tensor/weight interfaces and the Engine execution contract

Gemma4 keeps its current support scope: no new media, MTP, session or batch capability. Migrating other families
is not a precondition for converging their public interfaces.

1. **Connect the public data consumers.** Converge the remaining cross-module interfaces that still pass media and
   state payloads in and out as DenseTensor values. Local activation computation inside models and backends does not
   become a public contract, and Tensor gains no computational responsibilities. Weight and position-table placement,
   expert projection and mixed representations still require runtime acceptance.
2. **Complete the Engine execution contract.** Connect the generation, MTP and batch consumers that still depend on
   family types or private execution interfaces to the public model and execution contracts. CUDA resource
   acquisition and operations belong to the backend implementation. Real session restore for the old families is kept
   together with the consumer migration in item 1. Model computation stays in models; no resource bundle, callback
   shell or per-item forwarding interface is added.
3. **Make ownership and failure paths consistent end to end.** Keep view, encoding helper buffers, shard and
   on-demand expert keep-alive in sync; keep backing de-duplication accounting, device placement validation, failure
   cleanup and cancel/reload/unload draining. Unsupported view/cast combinations are rejected before modification or
   commit, with no implicit whole-model dequantization. KV allocation reservation, expert borrow-region return, Graph
   warmup restoration and last-consumer keep-alive still require runtime acceptance.

**Source completion criteria:** Gemma4 and its public production call chain use data through Tensor references and
owning handles; the public weight interface has no old wrappers or CUDA private types; the Engine organises assembly,
generation and release through the public model and execution contracts, with no downcast to a concrete executor;
models do not depend on Executor, and generic runtime code does not access family-private layouts. Build, numerical
and lifecycle runtime acceptance must still be completed once the deferral is lifted.

## 2. Remove model execution dependencies and live State bindings

Qwen4Exp and DeepSeek V4.1 keep their request caches in their own State. Qwen4Exp's layer weights, forward math,
LayerState and Graph scheduling are separated by responsibility, and MoE executes through public numeric ops; the
remaining work is weight/KV coordination in the graph driver, the rest of the profiling wiring and execution resource
assembly, then removing the execution context from the top level and from the predictor.

- Remove the remaining dependencies on CudaExecutionContext and Executor from models and compute units; concrete
  models organise Attention/MLP and other compute units and call them through public ops, with execution resources
  obtained by the backend inside the scope. Keep the math, dtype, CPU/layer placement, projection order and error
  propagation.
- Move the remaining execution scratch in request State and MtpState into the backend, and keep resource accounting
  and Graph storage dependencies in sync; keep the separation of predictor weights from request state, and the
  existing lifecycle of the Engine's adaptive policy. A standalone MtpState is not a condition for declaring
  execution-dependency cleanup complete.
- As execution dependencies are removed, synchronise the allocation planning, predictor, prefill, batch, streamed KV,
  session and Graph consumers; capture and restore use the owning State, keeping Qwen35's joint-restore
  stage/drain/publish transaction, storage identity and last-consumer keep-alive, without switching the current
  request through a shared model.
- Adopt the Phase 1 public Engine/ops contracts and delete the remaining family assembly-selection adapters,
  execution borrow shells and forwarding entry points according to their real responsibilities, not just renaming
  them; the backend Ffn weights and MoE routing composition in the remaining models converge with the model weight
  contract; converge the remaining Transform/compute-graph callbacks, with concrete models interpreting layer order.
  Internal per-layer records inside a model may stay; no general Block inheritance tree or unified layer union type is
  reintroduced.

**Source completion criteria:** the models and compute units above do not store an executor, an execution context or
live State; shared models do not carry request state through mutable aliases, and forward and its related consumers
use the owning State explicitly; the CUDA production path of the models above has no old execution adapter
dependencies. Deferred tools, tests and Metal/HIP consumers are not required to keep the old implementation.

## 3. Extend Qwen35 and verify the full chain

After the first two phases are complete, connect Qwen35's embedding, forward, logits, replay and existing
predictor/batch/session/Graph adapters to the public model, state and ops contracts, and remove the per-layer
Executor/Collectives parameters.
Once build wiring and test work resume, verify the real path from load and input preparation to output and release;
reference, eval, diagnostics, service and the affected distribution consumers are accepted together.

| Check area | Behaviour that must be covered |
| --- | --- |
| Load and weights | Default TQ, explicit MFQ compatibility, source/config/YaRN/placement validation, RoPE/weight order, CPU layer count and the conditions actually supported or rejected; logical/physical bytes, layout, aliasing and grouped output of encoded Tensors |
| Forward and generation | Full/Linear mixed layers, normal/chunked prefill, decode, MRoPE, internal mask, explicit positions, output norm, lm_head; compare logits, valid state and committed positions |
| Batch and isolation | Request/slot two-way ownership, page generation, batch changes, independent scratch, cancellation and storage reuse; model batch state is interpreted only on the model side |
| MTP | Accept/reject, partial replay, position commit, target/predictor and recursive/conv state consistency; unsupported suffixes stay explicitly rejected |
| Session/prefix | Text and media identity, candidate prepare/stage/drain/publish, hit accounting, failure fallback/reset, cache budget and isolation of unrelated requests |
| Graph | Capture matches direct execution, warmup isolation, state/scratch address and storage identity matching, invalidation synchronisation, last-consumer keep-alive |
| Execution and lifecycle | Backend dispatch and device validation, nested scope and exception recovery, coroutine suspend/resume, parallel workspace isolation, async keep-alive, load failure, cancellation, backpressure, draining, reload, unload, a single terminal state, multi-Engine isolation and budget return |
| Input and output | The supported text and multimodal paths, template/tokenizer, stop conditions, incremental text and structured results |
| Tools and distribution | reference per-layer low-memory loading, standalone State/Executor and execution scope; real eval/diagnostics entry points; loading after JIT/wheel installation |

Keep fixed inputs, dtype, sampling settings and the original numerical error criteria; do not change error
propagation, recovery or release order.
Cover communication on/off, default TQ and explicit MFQ for the affected targets; TP/EP and CPU/layer placement use
the existing support conditions, with no widened capability claims.

## Qwen35 phase completion criteria

- The CUDA production consumers have completed the public contract migration; Qwen35's math, state and
  batch/MTP/session/Graph and media semantics all belong to models, and generic runtime modules have no
  family-private dependencies.
- The deferred build, test, tool and distribution adapters are restored and completed; production, batch, reference,
  eval, diagnostics and service paths all have real verification.
- The numerical, state, lifecycle and failure paths in the table above pass acceptance. Static checks, skips,
  returning 77 or model_branch=skipped do not count as a runtime pass; while required verification is unfinished the
  phase stays incomplete.

## 4. Consolidate csrc files and module ownership

The files below still need consolidation for CUDA and the necessary common code. They are listed item by item and
interleaved with the earlier phases according to dependencies, without changing the Gemma4-first order.
When implementing each item, revise the corresponding target responsibilities and file ownership in architecture.md
at the same time; no second complete directory table is created.
Generic generation source is centralised in engine/generation; the public Engine and the transitional CUDA entry
share media preparation, request validation and cleanup, and normal generation and batch share the advance loop. The
focus stays on the public execution contract; its test and distribution references are still restored as deferred
items.
Whether files are merged is decided by responsibility and actual consumers; line count is not a reason for deletion,
and no family is required to have the same file list.

Capability decisions, the model factory and the Text instance responsibilities depend on the model/Engine contract
migration described above; moving files alone is not completion.

1. **Consolidate capability decisions and component descriptions.** Scope: common/capabilities.h, components.h and
   each family/components.*.
   Public capability data is provided by models/metadata.h; implementation selection belongs to the model factory, and
   the concrete vision, MTP and duplex capabilities and their accounting are provided by the actual family. Remove the
   branches in common helper code that read family-private members.
   Completion criteria: the public model contract provides capabilities and accounting; common no longer assembles the
   same description by family type special-casing or member probing; no registry or callback table is added.

2. **Unify the model factory and instance assembly entry points.** Scope: engine/load.*, models/load.h and the
   with_model selection and callback assembly inside them.
   Model selection and model assembly are centralised in models/load.* and return CausalModel; library-level load and
   reload coordination belongs to engine/load.*, CLI parsing belongs to commands, device resource initialisation
   belongs to backend, reusing the Engine lifecycle and draining mechanism; apps keeps only main.
   Completion criteria: Engine does not parse model paths or include every family; the model factory does not maintain
   a second instance lifecycle; the old callback-style assembly entry is deleted once its consumers are connected.

3. **Converge the transitional instance responsibilities in engine/text.h.** Scope: execution context, weight and KV
   holding, budget and model adapters.
   Resource holding and coordination belong to Engine, generation advance reuses engine/generation, and model
   computation, state and optional components belong to the corresponding family; the transitional Text assembly shell
   is removed as the model factory converges.
   Completion criteria: the transitional Text assembly shell is deleted, Engine holds resources through the public
   model and execution contracts, and does not duplicate the generation flow that has already converged.

Each source consolidation synchronises the CUDA production callers, includes, the required CMake source lists and
target dependencies; no copy of the old path, forwarding header or duplicate definition is kept.
Affected test and fixture references, reference/eval/diagnostics, JIT, wheel, source archives and Metal/HIP
references are registered item by item as deferred items; they are synchronised and accepted once restored, and
deferred consumers are not a reason to keep the old implementation.
Delete an entry once its source consolidation is complete; build, numerical and lifecycle verification are still kept
according to their own status — putting directories in place is not runtime acceptance.

## Deferred items

| Item | What to do when restored |
| --- | --- |
| CMake and old build entry points | Converge the remaining include roots, target dependencies and native/Torch wiring, and handle the build entry points of old tools and deferred backends; verify the combination of apps entry points, the eval/diagnostics libraries and the optional Torch reference library, keeping numeric options and output paths |
| Metal/HIP and its old-directory cleanup | Adapt to the tyloi namespace and the Tensor abstraction interface, and to pointer/reference contracts; synchronise the Gemma4 MLP and shared Attention loading interfaces, the grid encoder embedding adaptation and the old path references involved in section 4; keep no compatibility aliases and do not block the CUDA mainline |
| Tool wiring | Synchronise the model/State, request output, loading and execution resource interfaces and the Torch edge conversions in reference, eval, KL, diagnostics and the minicpmo45 command, including reference's old KVCache consumers, Gemma4 public layer loading and forward, public Loader creation and the header that owns the MFE helper declarations; include them in full-chain acceptance |
| Test migration, changes and runs | Synchronise the public Tensor, unified KvView and quantization contracts, media cache and field traversal, Loader/Loan, model State/MtpState and ops interfaces, model-exclusive ownership, typed placement options and placement.h validation, and the generation, models/source/session, weights/moe paths; migrate Gemma4's single forward and state interfaces, Layer/LayerState/LayerSnapshot, the old Task, Qwen35 and explicit layer forward/prefill parameters, expert projection and biased routing, sparse attention, native FFN/residual, the FfnMode/FfnOptions policies and routed ops, DSpark and staged prefill and their other consumers; delete the old tests and targets for the removed Pipeline and restore acceptance for the plain MoE and TQ target Graph interfaces that are actually kept; keep the numerical criteria, and restore the Graph, budget, profiling-range restoration, lifecycle, shared_execution_test, kernel package and source archive checks |
| JIT, packaging and build/runtime acceptance | Synchronise source, resource declarations and post-install loading, and verify the real production path |

Deferred items are not recorded as complete, and are not bypassed with old interfaces, forwarding headers or source
copies; independent features and full-chain acceptance for other CUDA families come later.
Namespace migration covers TyloInfer's own code and the necessary consumers; MFQ file format/encoding identifiers and
third-party namespaces keep their own semantics.

## Plan maintenance

- Completed source tasks are deleted from the plan; unfinished build and runtime acceptance is kept. Stable
  responsibilities live in the architecture document, and Git records the process.
- Restore the necessary verification according to the actual changes, without lowering the original criteria;
  unverified items must not be recorded as a runtime pass.
- The machine has 32 GB of memory; avoid keeping several models or several heavy verification runs resident at once,
  and do not pin -j.
