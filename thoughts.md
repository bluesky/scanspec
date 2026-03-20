# Scanspec 2.0 design requirements

> **Release scope**: scanspec 2.0 is a major breaking release — aligned in spirit with 1.x but with no backwards compatibility for JSON specs or Python APIs. ophyd-async will be updated to work with the new API; no back-compat is required there either.

## Currently
Scanspec as it is now supports:
- **Specs**: Composable, serializable scan description (Linspace, Product, Snake, Ellipse, etc.) serialized as tagged-union JSON (`"type": "Product"`, …). Pydantic field descriptions drive UI form generation; a FastAPI service lets a UI POST JSON and get back validation and scan previews.
- **Memory efficiency**: `Product` stores a *stack* of compact per-axis `Dimension` arrays (e.g. 100+2000+2000 floats = ~32 kB) rather than expanding the full Cartesian product (3.2 GB). Expansion and snaking are lazy, computed via index arithmetic on demand.
- **Flyscan consumption**: `Path(stack).consume(N)` materialises N-point chunks of `lower`/`midpoint`/`upper`/`gap`/`duration` arrays, feeding motion controllers and detectors batch-by-batch. `Fly(spec)` enables bounds calculation so the motor moves continuously through each frame; `gap` flags turnarounds.

It is currently used in ophyd-async to supply:
- The number of frames for a detector (all the same): `int(np.prod([len(dim) for dim in stack]))`
- The batched points to pass to a motion controller for PVT interpolation: `slice = path.consume(N)` passed to https://raw.githubusercontent.com/bluesky/ophyd-async/refs/heads/main/src/ophyd_async/epics/pmac/_pmac_trajectory_generation.py
- The triggers inserted at each gap in a PandA sequence table: https://github.com/bluesky/ophyd-async/blob/b36695e9146a94a73b3f9689b1d35fd9e4697e6c/src/ophyd_async/fastcs/panda/_fly_logic.py#L102-L152

## Required user stories

There are some user stories we must support with 2.0. All are in scope for the initial 2.0 release; they may be delivered in stages to produce MVPs along the way.

### Servo cycle rate motor positions

At the moment kinematics are in the motion controller, so PVT interpolation will produce compound motor positions every servo cycle which are translated into raw motor positions by the controller. We need to lift these kinematics up into ophyd so they can be used in analysis and so we can do path planning up front for collision avoidance. This means we need a position at each motion servo loop cycle rather than at nominal detector position. This would require us to slice the path in time rather than index by detector frame, saying "I want motor positions every 0.2ms" that require no PVT interpolation

> **Clarification**: scanspec does not need kinematic calculations or motion smoothing. The `Spec` provides a continuous function for axis motion within each contiguous section; discontinuities are exposed as `gap` flags (as today) and ophyd-async handles the trapezoidal bridging. `Spec.calculate()` returns a compact **intermediate structure** (a stack of `Dimension`-like objects) where each dimension holds a reference to a numpy-enabled function; actual position arrays are generated on demand by calling those functions with a time or index array. This keeps deserialization memory cost O(spec complexity) not O(scan size), while allowing arbitrarily dense arrays to be produced in batches at consumption time. A separate `Spec.midpoints()` iterator path must also be maintained for software step-scanning.

### Specified livetime and deadtime for detectors

We also need to be able to specify the livetime (and optionally the deadtime) for detectors rather than the duration. This should be expanded when sent to ophyd-async so that the motion will then get the duration. We still need the ability to just specify duration and ophyd will fill in livetime and deadtime from it.

> **Clarification**: `duration = livetime + deadtime`. Livetime/deadtime is commonly a per-spec scalar pair but must also support per-point arrays (for ptychography variable-gap patterns). The API must expose which mode is in use, as only some detectors support per-point timing. When `deadtime` is omitted, ophyd-async fills it in; scanspec does not mandate a value.

### Supporting continuously monitor detectors

We need the ability to continuously monitor a detector for the entire duration. E.g. capture timestamped temperatures from a PV while doing a grid scan or acquire camera at 10Hz while stretching a sample

> **Clarification**: Continuously monitored detectors are a **separate top-level concept** (not a node in the `Spec` tree). Their output is associated with the scan by timestamp correlation only; no frame-index coupling.

### Multiple detectors at multiple rates

At the moment scanspec has the concept of "scan index" and "duration" and all detectors are expected to be triggered at that rate. We would like to support different detectors at different rates, but still specify the relationship between them (e.g. this one goes 10 times faster than the other, but this one is free-running)

A maximal example to demonstrate would be:
- At each DCM energy in a given range
  - In parallel
    - Do an X,Y flyscan scan over a given grid
      - With saxs and waxs pilatus detectors taking 1x frame with 3ms livetime 1ms deadtime
      - With timestamp, X and Y encoders from the PandA taking 10x frames with 0.3ms livetime and 8e-9 deadtime
    - Do N time triggered detector acquisitions
      - With front and side optical cameras taking 1x frame with 48ms livetime 1ms deadtime
      - With timestamp from PandA

> **Clarification**: Detectors in the **same stream** must trigger at integer ratios of each other (validated at spec creation). Detectors in **different streams** have no phase lock — only timestamps tie them together. Data model is leaning towards `Spec[AxisT, DetectorT]` embedding detector descriptions in the spec tree, but a separate `DetectorSpec` parallel tree is also a candidate; prototype examples needed to decide.

### Gaps in detector trigger pattern

For ptychography we may want to leave variable gaps between detector exposures, so the trigger pattern would look something like "0.1s livetime, 0.01s deadtime, 0.1s livetime, 0.3s deadtime" repeated for each line of a gridded motion pattern. Care needed to make sure the motion trajectory is an integer number of these repetitions.

## Optional user stories

These are user stories that we could support, we should consider them in the design, but not if it makes the implementation too messy

### Fast shutter

Most beamlines will be happy with "open the fast shutter at the beginning of the scan and close it at the end", which doesn't need to be expressed in scanspec. However for some beamlines they would like to open and close their shutters during the turnaround gaps. It is acceptable to then specify the motion for the turnaround in scanspec, with shutters being just another "axis" (although one that can only be 0 or 1).

### Waiting for a sample environment

For a purely time-based scan, we would like to express "take 10 frames with 1s livetime, then trigger a sample environment, then wait for an input to say it's done, then take 100 frames with 0.1s livetime".

If we cannot do this we can do it as a custom plan.

### Allowing positions to be specified as relative

Allow positions for axes to be specified as relative instead of absolute. This allows fully relative scans, or repetition of relative axes in inner dimensions to an absolute outer axis.

> **Clarification**: Relative positions must survive JSON round-trip serialization (e.g. `{"type": "RelativeLine", ...}`). "Relative to what" is resolved at execution time in ophyd-async, not at spec-creation time.

### Ending a segment early

Allowing ending one segment of the scan early. This would be useful for something like:

- At each DCM energy in a given range
  - For each X position of sample, until value from detector is 
    - Acquire 1x frame from detector with 0.1s livetime

The complications are that this makes the specified scan shape only a *maximum* shape. It also means we need some kind of "completion condition" mini language. This is probably a requirement to discard, but it would be good to work out where it would fit if we put this in the future.

> **Decision**: Treat as a **future extension point**. The 2.0 design must not preclude it, but it will not be implemented in 2.0. When added, the scan shape becomes a *maximum* shape and a "completion condition" mini-language would be required.

## Non-functional requirements

From developing the original scanspec, here are some requirements from the technical side

### Keeping the memory efficiency

At the moment deserialization does not take any more memory for large scans than small scans. We should keep it that way. We also need to know the size of each dimension taking memory for the whole scan. When we calculate a dimension, instead of creating upper/lower/midpoints/duration, then we should keep a reference to a numpy enabled function that can calculate these on demand from a time array.

### Ensuring we can use these easily for software step scans

Although the driving requirements for this rewrite is flyscans, we also need to be able to use this in step scans. So the `for midpoint in spec.midpoints()` functionality should be maintained, and thought given to how the detectors to be triggered at each point could fit into this API.

### Pydantic BaseModels

Pydantic dataclasses are not well supported, we only use them so we get positional args. If we are doing a rewrite we should switch to BaseModels. We should be able to support positional args by Spec have a custom `__init__(self, *args, **kwargs)` method that binds args to kwargs then calls the superclass. For typing we should be able to use `dataclass_transform` to support this statically. This needs prototyping.

### Bluesky streams

Downstream we will be making a stream for each set of detectors that are triggering at multiples of each other. We should include that stream name in the Spec, defaulting to `primary`.

> **Clarification**: A stream name identifies a group of detectors related by integer trigger ratios. The stream name lives on the detector grouping node, not on individual axes.
