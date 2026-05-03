# PRD Implementation Status

**PRD version:** 1.4.0 Final
**Last updated:** 2026-05-03

---

## Legend

| Status | Meaning |
|---|---|
| ⬜ Not started | No code written yet |
| 🔵 In progress | Actively being implemented |
| 🟡 Done, needs review | Code written, awaiting review/testing |
| ✅ Complete | Reviewed and merged |
| ⏳ Blocked | Waiting on a dependency or decision |
| ❌ Deferred | Moved to a later milestone |

---

## Open questions

| # | Question | Status | Blocks |
|---|---|---|---|
| OQ-1 | Is `sharp` a hard or optional dependency? What is the degradation behavior when absent? | ✅ Resolved | Used `imagescript` + `imghash` instead — zero native deps, pure JS/WASM |

---

## Feature 1 — `analyze_image` tool

| FR | Description | Status | Notes |
|---|---|---|---|
| FR-1.1 | Register `analyze_image` tool when proxy enabled | 🟡 Done | Registered conditionally on `tool=on && mode!=off` |
| FR-1.2 | Tool schema (images, question, model, crop, reason) | 🟡 Done | TypeBox schema with all fields |
| FR-1.2.1 | Crop semantics (region, normalized, pixels) | ✅ Complete | All three forms resolve to pixels; cropping applied via ImageScript |
| FR-1.2.2 | `reason` field (analytics logging) | 🟡 Done | Logged in telemetry entry |
| FR-1.2.3 | Tool description text for agent | 🟡 Done | Full description per PRD spec |
| FR-1.3 | Path resolution and security | 🟡 Done | Delegates to existing readImageFileWithReason + path allowlist |
| FR-1.4 | Model override parameter | 🟡 Done | Agent-initiated override honoured |
| FR-1.5 | Per-provider consent for tool calls | 🟡 Done | Checks existing consent |
| FR-1.6 | Result LRU cache | 🟡 Done | LRUCache with configurable size, keyed by (hashes, crop sig, question hash, model) |
| FR-1.7 | Configuration (tool, max-images-per-call, cache-size) | 🟡 Done | Slash commands + env vars |
| FR-1.8 | Result fencing with `<vision_proxy_analysis>` tag | 🟡 Done | buildAnalysisFence with all attributes |
| FR-1.8 | `crop_origin` attribute on cropped results | 🟡 Done | Included in fence when crop applied |
| FR-1.8 | `image` attribute suffixed with `#crop:x,y,w,h` | 🟡 Done | Included in fence when crop applied |
| FR-1.9 | Telemetry (`vision_proxy.tool_call` entries) | 🟡 Done | Appended via pi.appendEntry |
| FR-1.10 | Precedence rule (tool > generic, no correction entry) | 🟡 Done | Tool result returned directly; no correction entry |

## Feature 1 — Infrastructure

| Item | Description | Status | Notes |
|---|---|---|---|
| INFRA-1 | `image-size` integration for dimension extraction | 🟡 Done | extractDimensions() + storeImageMeta() |
| INFRA-2 | `imagescript` integration for cropping | ✅ Complete | cropImage() + piAiImageToBuffer/bufferToPiAiImage helpers |
| INFRA-3 | `imghash` integration for pHash | ✅ Complete | computePHash() + hammingDistance() helpers; lazy-loaded |
| INFRA-4 | In-memory `_imageMeta` map (hash → width/height/filename) | 🟡 Done | Map exported, populated on ingestion |
| INFRA-5 | Update `readImageFileWithReason` to return basename | 🟡 Done | `filename` field added to ReadImageResult |
| INFRA-6 | Update `fenceUntrusted` to handle all three fence tags | 🟡 Done | Regex covers description, analysis, joint_description |
| INFRA-7 | Region → pixel rectangle resolution | 🟡 Done | resolveRegion() + normalizedToPixels() |
| INFRA-8 | Normalized → pixel rectangle resolution | 🟡 Done | normalizedToPixels() with clamping |
| INFRA-9 | Pixel clamp + zero-area validation | 🟡 Done | clampPixels() with null return on zero area |

## Feature 2 — Multi-image batched comparison

| FR | Description | Status | Notes |
|---|---|---|---|
| FR-2.1 | Auto-proxy: joint description for N ≥ 2 images | ✅ Complete | Joint call after per-image analysis, cached in session | |
| FR-2.2 | `analyze_image` with ≥2 images: batched vision call | 🟡 Done | Sends all images; uses adaptive prompt when available |
| FR-2.3 | Joint calls obey consent + injection fence + include_context | ✅ Complete | Reuses same consent/fence flow | |
| FR-2.4 | `maxBatch` config | 🟡 Done | Config field + slash command + env var |
| FR-2.5 | Adaptive joint-call system prompt | ✅ Complete | buildAdaptiveJointPrompt with comparison structure | |
| FR-2.5.1 | Filename hint patterns (Appendix D) | ✅ Complete | generateFilenameHints + extractVersion; before/after, old/new, versioned, numbered, date-ordered | |
| FR-2.5.2 | pHash similarity hint | 🟡 Done | computePHash() + hammingDistance() available via imghash |
| FR-2.5.3 | Hints are advisory | ✅ Complete | Hints only appended to prompt, not required | |
| FR-2.5.4 | Hints suppressed for tool path | ✅ Complete | Hints not included in analyze_image tool calls | |
| FR-2.6 | Joint description fencing with `<vision_proxy_joint_description>` | ✅ Complete | buildJointDescriptionFence with dimensions JSON | |
| FR-2.7 | Joint description cost telemetry | ✅ Complete | CUSTOM_TYPE_JOINT entry with images and description | |

## Feature 3 — `/vision-proxy describe` slash command

| FR | Description | Status | Notes |
|---|---|---|---|
| FR-3.1 | `describe` and `redescribe` subcommands with extended crop syntax | ✅ Complete | parseDescribeArgs + full slash handler | |
| FR-3.2 | `describe` semantics (resolve, joint, --save) | ✅ Complete | Resolves images, supports --question/--crop/--model/--save | |
| FR-3.3 | `redescribe` sugar for `describe --save` | ✅ Complete | parseDescribeArgs with isRedescribe flag | |
| FR-3.4 | Inline consent prompts | ✅ Complete | Checks hasConsent before calling vision model | |
| FR-3.5 | TUI output with `[Vision Proxy]` prefix | ✅ Complete | All output uses [Vision Proxy] prefix | |
| FR-3.6 | `vision_proxy.command` logging | ✅ Complete | CUSTOM_TYPE_COMMAND entry with command, images, question, model, latency | |

## Feature 4 — Optional grounded-coordinate output

| FR | Description | Status | Notes |
|---|---|---|---|
| FR-4.1 | `supportsGrounding` per-model flag | 🟡 Done | Via groundingModels config map |
| FR-4.1.1 | Curated Tier 1 default list | 🟡 Done | Shipped in DEFAULT_CONFIG |
| FR-4.1.2 | Grounding-models slash commands (add/remove/list/reset) | ✅ Complete | With excluded-model confirmation prompt, --format flag, default qwen_pixels | |
| FR-4.2 | Native-format grounding system prompt injection | 🟡 Done | buildGroundingInstruction() for all formats |
| FR-4.3 | No grounding instruction when `supportsGrounding: false` | 🟡 Done | Returns empty string |
| FR-4.4 | Proxy does not parse/rewrite returned coordinates | 🟡 Done | Pass-through |
| FR-4.5 | Additive — model ignoring instruction is harmless | 🟡 Done | Instruction appended, not required |
| FR-4.6 | Crop + grounding: coordinates relative to crop, `crop_origin` for mapping | 🟡 Done | crop_origin in fence attributes |
| FR-4.7 | Per-model `grounding_format` registry + default mappings | 🟡 Done | groundingModels map + getGroundingFormat() |

## Cross-cutting

| Item | Description | Status | Notes |
|---|---|---|---|
| CC-1 | Update auto-proxy fence to `<vision_proxy_description>` with width/height/filename attributes | 🟡 Done | buildDescriptionFence() used in before_agent_start |
| CC-2 | Update `context` handler to use new fence format | 🟡 Done | Uses buildDescriptionFence() in context handler |
| CC-3 | Configuration: new fields in `vision-proxy.json` | 🟡 Done | All new fields with defaults |
| CC-4 | Configuration: new slash subcommands (tool, max-images-per-call, max-batch, cache-size) | 🟡 Done | All four subcommands + interactive menu |
| CC-5 | Configuration: new env var overrides | 🟡 Done | All env vars parsed in readEnvOverrides() |
| CC-6 | Telemetry: new session entry types | 🟡 Done | CUSTOM_TYPE_TOOL_CALL constant |
| CC-7 | Security: closing-tag neutralisation for all three fence tags | 🟡 Done | Updated fenceUntrusted regex |
| CC-8 | Backwards compat: 1.3.0 config files load unchanged | 🟡 Done | sanitize() fills new fields with defaults |
| CC-9 | Update README.md | ✅ Complete | Updated for 1.4.0 GA with all features documented | |

## Tests

| Area | Status | Notes |
|---|---|---|
| Unit: crop resolution (region → pixels) | ✅ Complete | resolveRegion, resolveCropEntry |
| Unit: crop resolution (normalized → pixels) | ✅ Complete | normalizedToPixels with clamping |
| Unit: pixel clamp + zero-area validation | ✅ Complete | clampPixels |
| Unit: region name validation | ✅ Complete | isValidNamedRegion |
| Unit: `CropEntry` union discrimination | ✅ Complete | resolveCropEntry |
| Unit: filename hint pattern matching | ✅ Complete | generateFilenameHints + extractVersion |
| Unit: version extraction (non-contiguous, decimal) | ✅ Complete | extractVersion |
| Unit: `fenceUntrusted` with all three tags | ✅ Complete | |
| Unit: `_imageMeta` map population | ✅ Complete | storeImageMeta + extractDimensions |
| Unit: LRU cache key stability across crop forms | ✅ Complete | cropSignature + buildToolCacheKey |
| Unit: grounding format registry lookups | ✅ Complete | getGroundingFormat |
| Unit: config backwards compatibility | ✅ Complete | sanitize defaults for new fields |
| Unit: slash command argument parsing (--crop forms) | ✅ Complete | parseDescribeArgs: region, normalized, pixels, --question, --save, --model, redescribe restrictions | |
| Unit: readEnvOverrides for 1.4.0 fields | ✅ Complete | |
| Unit: buildDescriptionFence / buildAnalysisFence | ✅ Complete | |
| Unit: cropImage (ImageScript cropping) | ✅ Complete | 10×10 PNG → 5×5 crop, JPEG encoding, OOB |
| Unit: piAiImageToBuffer / bufferToPiAiImage | ✅ Complete | Round-trip base64, default MIME |
| Unit: computePHash | ✅ Complete | Valid image → hex hash |
| Unit: hammingDistance | ✅ Complete | Identical, differing, null, unequal length |
| Integration: `analyze_image` tool end-to-end | ⬜ Not started | Requires runtime with model registry |
| Integration: auto-proxy + tool in same turn | ⬜ Not started | |
| Integration: joint description for N ≥ 2 auto-proxy | ⬜ Not started | |
| Integration: `/vision-proxy describe` slash command | ⬜ Not started | |

---

## Milestones

| Milestone | Target | Status |
|---|---|---|
| 1.4.0-beta.1 | Feature 1 (`tool=off` default), crop forms, dimensions in fence | 🟡 Done, needs review |
| 1.4.0-beta.2 | Feature 3 (slash commands with crop syntax) | ✅ Complete |
| 1.4.0-beta.3 | Feature 2 (`maxBatch=1` default), adaptive prompt, hints | ✅ Complete |
| 1.4.0-beta.4 | Feature 4 (grounding registry, Tier 1 list) | ✅ Complete |
| 1.4.0 | Flip `tool=on`, `maxBatch=4` | ✅ Complete |

---

## Changelog

| Date | Change |
|---|---|
| 2026-05-03 | **1.4.0 GA**: Flipped defaults (tool=on, maxBatch=4). Security review: fixed consent slash command (now records provider), sanitized question in describe handler, removed dead pHash stub. Updated README.md. 180 tests passing. |
| 2026-05-03 | Milestone beta.4: Feature 4 grounding-models slash commands (add/remove/list/reset), excluded-model warnings, parseGroundingFormat validator. 180 tests passing. |
| 2026-05-03 | Milestone beta.3: Feature 2 (multi-image batched comparison). Auto-proxy joint descriptions for N≥2 images, adaptive joint prompt with comparison structure, filename hint patterns (Appendix D), pHash infrastructure, buildJointDescriptionFence. 172 tests passing. |
| 2026-05-03 | Milestone beta.2: Feature 3 (`/vision-proxy describe` + `redescribe`) fully implemented. parseDescribeArgs with all three crop forms, --question/--crop/--model/--save flags, consent checks, [Vision Proxy] TUI prefix, vision_proxy.command telemetry. 152 tests passing. |
| 2026-05-03 | OQ-1 resolved: chose `imagescript` + `imghash` over `sharp`. INFRA-2 (cropping) and INFRA-3 (pHash) implemented. Crop now applied to image bytes in `analyze_image` tool. 134 tests passing. |
| 2026-05-03 | Milestone beta.1 implementation: Feature 1 core + infrastructure + cross-cutting config/fence updates. 112 tests passing. |
| 2026-05-03 | Initial status document created |
