# TyloInfer CUDA Refactoring Conventions

This document maintains the stable target responsibilities, scope and invariants. The current structure is in
[architecture.md](architecture.md); remaining tasks and completion criteria are maintained only in
[plan.md](plan.md). Per-commit progress is not recorded here.

## Goals and scope

Starting from TyloInfer's current implementation, separate model math, live request state, weight residency and
execution resources, and complete the connection of real production consumers. The design reference is MFQ
refactor/cuda at a93e73a27a30e566a440eb5dc146873d75064e94; the reference is used only for responsibilities,
interfaces and numerical conventions. No intermediate implementation is copied, and its build or test results are not
cited as evidence of completion for TyloInfer.

Execution order, the current model mainline and the deferred scope are authoritative only in [plan.md](plan.md);
interface changes synchronise the required consumers, and a shared interface is not used to extend other models'
capabilities. Load, generation, batch, MTP, session, Graph, cancel and unload are accepted within the existing
support scope.

**This phase refactors CUDA and its necessary common code only.** Compilability, runnability and compatibility of
Metal/HIP are neither constraints nor acceptance conditions; the shared interfaces and directories needed by CUDA may
be adjusted directly, without waiting for other backends to adapt and without keeping old interfaces, compatibility
layers, forwarding headers or source copies for them. Metal/HIP adaptation is left for later. Python ops and
execution logic are out of current scope; CMake, JIT/packaging and test synchronisation are handled when the plan
restores them.

During the split, if the current code directly exposes a wrong responsibility, dependency or object ownership, it may
be fixed in the same batch; interfaces and assembly advance as planned. Throughout, keep the math, dtype, validation,
error propagation and lifecycle constraints.

The default TQ path follows the same ownership and execution rules; the MFQ compatibility format stays explicitly
opt-in. The removed offline quantization and calibration product entry points are not restored.

## Target responsibilities

Target directories and responsibilities are authoritative only in [architecture.md](architecture.md); this document
does not copy another directory table. In the final structure, models live in `csrc/engine/models` and own model
config, weight roles, forward, model-specific State/session, batch, MTP, Graph adaptation and assembly; the generic
`engine/generation` owns generation advance, sampling and the MTP flow; `engine/scheduler` owns request lifecycle,
admission, batching, backpressure and terminal state; `engine/requests` owns the request/result contract and
input/output interpretation; `engine/state` owns the generic state payload, pages, snapshots and session;
`engine/weights` owns reading, preparation, residency, transfer and weight statistics. Device execution belongs
entirely to the executor, ops, kernels and native of `csrc/engine/backends/<device>`; apps only start commands;
commands parse arguments and call library-level load/reload; transport only handles external protocols. The file
responsibilities, dependency direction and final structure of each directory are in
[Final target layout](architecture.md#final-target-layout) and
[Module internals and file responsibilities](architecture.md#module-internals-and-file-responsibilities).
Each device's `backends/<device>` carries only executor, ops, kernels and native; the root `backends` provides the
shared dtype.h and the device directories, and device directories do not mirror the upper `engine` modules. Generic
mechanisms stay in the corresponding `engine` module, model knowledge stays in `models`, and the place where a device
API is called does not change its business ownership.

TyloInfer's own C++ namespace is unified as tyloi, the public abstraction interface is tyloi::Tensor, and DenseTensor
and each encoding implementation inherit from it; the mfq and mfq_tensor_backend namespaces and their compatibility
aliases are deleted.
Torch conversion stays at the adaptation edge; MFQ file format/encoding identifiers and third-party namespaces keep
their original semantics.

File ownership is decided by model knowledge, not by class names such as State, Run, Batch or MTP.
Model holds read-only shared weights, and each request's State is separate; the types and behaviour of model State
belong to models, generic payloads and transaction mechanisms belong to engine/state. Stateful implementations are
allowed inside the models directory.

Models compose Attention, MLP, Encoder and other compute units, hold read-only weights and config, and call public
ops through forward.
No common Block/Attention base class is forced; model differences are expressed as model-side math steps, and fused
ops define a clear compute range. Public ops own the numeric contract and device dispatch, backend ops implement the
algorithm; they do not receive a whole Model or a model-forward callback, and no Plan/Bindings, common Task IR or
second generation loop is added. Model assembly and family instantiation belong to models.

Engine's production implementation and public headers do not depend on concrete family types, names or private
layouts.
Model compute units use public ops, Tensor, weight and state contracts; they do not receive or store an Executor, and
constructor injection is not a way around this rule.
The public model interface does not expose backend-private Task/Tensor or managers; model semantics inside device
adaptation still belong to the family, numeric implementation belongs to backend ops.
Using requires, aliases or forwarding headers does not change where the real dependency lives.

Model implementations are carried by separate build targets; the generic Engine does not compile or transitively link
concrete model implementations; the final inference library or entry point combines the two. Device dependencies are
isolated by CMake targets, and dependency problems are not hidden by new include roots or source copies.

## Engine instance responsibilities

The final Engine takes an injected `CausalModel` and `Executor` through its constructor and has a built-in Scheduler;
the Scheduler advances generation directly inside the Engine. Engine coordinates requests, budget, failure cleanup,
unload and lifecycle, but does not parse model paths, does not hold load callbacks, and does not pass a full context
to models or ops. The target structure is in
[Architecture design](architecture.md#engine-instance-target-design).

The library-level load entry calls the model factory and the backend resource entry according to model and device
configuration, opens weights and assembles the Engine, and commands coordinate reload through that entry. The model
factory returns a `CausalModel` bound to a single model; reload does not maintain a variant of every family. During
migration, the existing generation/MTP, batch, Graph, session and Budget logic is reused; no request loop or
algorithm is duplicated.

Text, EngineInstance and Context/CudaExecutionContext are transitional implementations, consolidated into the target
responsibilities.
Executor owns execution scope, stream/event, workspaces, Graph execution resources and completion/draining; it does
not provide duplicated numeric entry points such as project/norm/attention, and it does not take over model loading or
request policy.

## Ownership and interface constraints

- CudaExecutionContext is dismantled according to its real owners; it cannot simply be renamed Executor or execution
  scope. Execution resources belong to the backend execution facility, and the state and weight managers stay in their
  own modules; models do not extract Projector, profiler, stream or scratch and pass them item by item, and projection
  records do not capture these resources in callbacks.
- No model-specific Runtime forwarding shell is introduced or restored; model assembly reuses existing load results
  and the shared execution interface.
- Reuse the existing Loader, caches, execution resources, Batch, Graph and budget mechanisms; compute units organise
  the actual math steps, weights are assembled by the concrete model's loading code, and layer order plus this step's
  position and confirmed prefix are interpreted by the concrete model and passed in directly; no extra Model/Block
  forwarding layer, shared layer context or generic inheritance framework, and no registry, callback table, proxy
  Model, old Run bridge or second execution loop.
  New interfaces may break compatibility; CUDA and its necessary common consumers must be synchronised; Metal/HIP
  adaptation does not block this phase.
- Requests/batches hold the mutable State of the target and of the predictor; read-only weights may be shared;
  computation uses the owning State explicitly; models and compute units do not store live State or execution
  workspaces.
  Ops define the workspace layout, the backend guarantees isolation and reuse per device and stream, and Graph
  includes execution scratch in storage keep-alive and identity checks; Engine coordinates budget and lifecycle as
  real growth happens, and does not store budget callbacks in the shared model.
  KV is first reclaimed according to the candidate's actual layout, then selected and reserved in the final layout
  according to available capacity; dynamic KV grows by actually mapped pages, without reserving full capacity that has
  not been allocated.
- Activations, weights and state payloads share the Tensor owning handle, borrowed by reference, held by value and
  returned by value; the representation interface promises only common data and lifecycle contracts, and the field and
  layout validation of each representation belongs to the concrete implementation, following the
  [Unified Tensor contract](architecture.md#unified-tensor-contract).
  Format implementations reuse encoding, numeric computation belongs to ops; no second public weight system is built
  and no complete old weight/execution manager is moved into a subclass. Object sharing, view aliasing and data clone
  semantics are explicit, and weights and their aliases are read-only.
  Unify logical shape/dtype and expanded-byte semantics; real accounting enumerates all backing and de-duplicates;
  layout-restricted operations are validated before modification or commit; a reference is not a substitute for
  keep-alive, and the lease covers the last consumer.
- Public ops dispatch by device and validate data compatibility; backend ops borrow resources from the current
  execution scope, do not call model forward or request scheduling, and do not depend on source, loaders or
  state/weights managers; native provides only device primitives and a generic allocation-record scope, and does not
  handle models, requests, weight types or residency budget policy.

## Invariants

- Keep the math formulas, dtype, accumulation order, compiled numeric options, original tolerances and all error
  checks.
- The scope is established by the resource owner outside synchronous advance, warmup or coroutine resume; on exit,
  exception or suspension the thread environment is restored; re-enter when resuming again or switching threads.
  Pure metadata/CPU operations need no CUDA scope; CUDA submission entry points such as loading establish the owning
  scope, and Tensor never implicitly creates an Executor or stream.
  The scope is not kept in a coroutine-local variable that survives a suspend point; it is not bound to the current
  model, request or State, and there is no process-level current Engine.
  The scope does not rebuild the stream per token, and exiting it does not imply a device sync; async completion,
  dependencies and keep-alive rules are in [Execution scope](architecture.md#executor-and-execution-scope).
- forward uses the request State passed in explicitly; state is not put back into a shared Model through
  bind/swap/unbind or a scope proxy, and one State is never passed while another is actually accessed. Model
  reset/checkpoint/restore composition belongs to models; the low-level transaction reuses Engine.
- Load, execution and storage resources have an explicit owner; failed loads, reload, cancellation and destruction
  wait for real consumers to finish before retiring state, weights and execution resources. Keep the original failure
  cleanup and exception propagation.
- Handles must reject stale access across owners or after reuse; failing to prepare another request must not pollute
  unrelated requests. Model cross-layer restore keeps its transaction boundary; existing candidate restore, cache
  failure policy and position validation must not be lost.
- Positions are committed for the inputs that actually completed forward. MTP target/predictor, recursive/conv and
  sampling state all follow the same accept, reject and partial-replay rules. Predictor weights are separate from the
  request MtpState, and the target State holds the latter; operations receive state explicitly, and the Engine's
  adaptive policy stays independent.
- Graph captures the actual state view and keeps storage, scratch and transfers alive until the last consumer
  finishes; warmup isolates production state. Request/slot/page identity and Graph constraints are in
  [architecture.md: session, MTP and Graph](architecture.md#session-mtp-and-graph); check invalidation,
  synchronisation and reuse first, and do not add a unified generation mechanism or claim an ABA defect without
  evidence.
- The target Scheduler belongs inside Engine and advances generation directly; keep the existing semantics of request
  submission, scheduling, cancellation and backpressure in the transitional logic. In the transitional stage step
  still returns owned typed events before publishing, each accepted request has exactly one terminal state, and media
  and Duplex use consistent cancellation and backpressure rules.
  The generic generation entry, normal and MTP advance and production instantiation are centralised in
  engine/generation, with no implementation copy left in the engine root or models/common.
  The public Engine and the transitional CUDA entry share media preparation, request validation and cleanup, and
  normal generation and batch reuse the same prefill/decode advance loop; session/MTP transactions are connected
  according to existing capability, and request cleanup waits for child coroutines to end their borrows.
- There is only one copy of the source. Synchronise this phase's CUDA callers and includes; deferred build, test and
  JIT/packaging references are registered for restoration and are not counted as migration complete. Shared file
  migration is conditioned on the CUDA production consumers being connected, without waiting for Metal/HIP adaptation
  and without keeping their old paths by copying shared implementations.
- CUDA kernels have exactly one source root, csrc/engine/backends/cuda/kernels, shared by native/JIT/wheel.
  When removing third_party or changing distribution paths, keep the required licences and NOTICE.

## Verification and documentation maintenance

The current constraint is: no compiling, no running, no changing tests — only static cross-checking of source,
references and documentation.
Build, numerical and lifecycle runtime acceptance stay open until the user lifts that constraint.
Source connection, static cross-checking, build passing and runtime passing must be stated separately;
while the required verification is unfinished, the Qwen35 phase in the plan stays incomplete.

Later, build only the current targets actually affected by the verification, then run the corresponding tests,
avoiding stale artefacts.
Qwen35's production, batch, reference, eval, diagnostics and service consumers are all in acceptance scope;
standalone op tests or source searches are not a substitute for full-chain connection. The concrete completion
criteria are in the plan.

Cover host communication on/off, CUDA default TQ and explicit MFQ compatibility configurations according to the impact
scope; standalone compute/loading support having communication off, and the communication requirements of the full
service and the Torch reference stay explicit.
Where source/JIT/distribution paths are involved, verify the actual resource loading after installation.

CPU policy, formats and readers are verified by direct behaviour and real files; device liveness, Graph, async
reclamation and numerics use the real CUDA path. Missing hardware or models, a return of 77, skip or
model_branch=skipped all count as unverified. Keep fixed inputs, original tensors, numeric settings and reproducible
commands; distinguish differences that already existed in the old implementation from problems introduced by the
migration, and do not loosen tolerances, change dtype or skip failing inputs.

The machine has about 32 GB of memory; avoid running several heavy tasks at the same time; do not pin -j or MAX_JOBS.
Verification scope matches the change; for a pure documentation change only check the source descriptions, paths,
links and consistency between documents.

The division of labour between documents stays fixed: architecture writes the target architecture and
responsibilities, refactor writes the stable constraints, plan maintains current status, remaining work and completion
criteria in one place. Completed entries are removed from the plan, and commit history is traced through Git;
do not append per-commit summaries, static audit logs or repeated "current progress".
