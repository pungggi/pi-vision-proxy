# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.7.0] - 2026-06-20

### Added

- **Session image recall** — the agent can now re-query an image it saw earlier in the session without a re-attachment or file path. Every `<vision_proxy_description>`, `<vision_proxy_analysis>`, and `<vision_proxy_joint_description>` block already carries an `image="..."` id; passing that id back to `analyze_image` (or `/multimodal-proxy describe`) recalls the original image bytes and re-runs the targeted question or crop against it (e.g. *"zoom into that screenshot from before"*).
- Image bytes are retained **in process memory only** — never written to the session log or disk — in a byte-bounded LRU store (`PI_VISION_PROXY_IMAGE_RECALL_BYTES`, default 64 MB, oldest-first eviction).
- **New env var**: `PI_VISION_PROXY_IMAGE_RECALL_BYTES`.
- Unit tests for `parseRecallRef` (bare hash, `sha256:` prefix, `#crop` suffix, case normalization, path rejection) and the recall store (round-trip, dedup, byte-budget eviction, oversized-single-image retention, recency bumping).

### Changed

- `analyze_image` now accepts a recall handle (the fence `image="..."` id) as an image reference in addition to a file path; the previous hard rejection of `sha256:` references is removed. The tool schema, tool description, and the injected Vision Proxy system-prompt section document the recall handle.

## [1.4.0-beta.1] - 2026-05-03

### Added

- **`analyze_image` tool** — agent-facing tool for targeted re-querying of images with multi-form crop support (FR-1.x). Disabled by default during beta; enable with `/vision-proxy tool on`.
- **Three crop forms**: `region` (named areas like `top-right`, `center`), `normalized` (0.0–1.0 fractional coordinates), and `pixels` (absolute pixel coordinates). All resolve to pixel rectangles with clamping and zero-area validation.
- **Image dimension extraction** via `image-size` package. Dimensions and filenames stored in an in-memory `_imageMeta` map populated on first image ingestion.
- **Enhanced fence tags**: `<vision_proxy_description>` now carries `image`, `width`, `height`, `filename`, and `crop_origin` attributes. New `<vision_proxy_analysis>` fence for tool results with optional `grounding_format` attribute.
- **LRU result cache** for `analyze_image` calls, keyed by (image hashes, crop signature, question hash, model).
- **Grounding format registry** with curated Tier 1 defaults (Qwen, Molmo, DeepSeek, InternVL, Gemini). Grounding instructions appended to system prompt per model's native format.
- **New configuration**: `/vision-proxy tool on|off`, `max-images-per-call <n>`, `max-batch <n>`, `cache-size <n>`.
- **New env vars**: `PI_VISION_PROXY_TOOL`, `PI_VISION_PROXY_MAX_IMAGES_PER_CALL`, `PI_VISION_PROXY_MAX_BATCH`, `PI_VISION_PROXY_CACHE_SIZE`, `PI_VISION_PROXY_PHASH_THRESHOLD`.
- **Security**: `fenceUntrusted` now neutralizes all three fence tag types (`description`, `analysis`, `joint_description`).
- **`readImageFileWithReason`** now returns the file's basename in the `filename` field.
- **Telemetry**: `vision_proxy.tool_call` session entries with crop form, latency, cache hit status.
- 112 unit tests covering crop resolution, LRU cache, dimension extraction, fence building, grounding lookups, config backwards compatibility, and all new env var parsing.

### Changed

- `VisionConfig` extended with `tool`, `maxImagesPerCall`, `maxBatch`, `cacheSize`, `pHashSimilarityThreshold`, `groundingModels` fields. Backwards compatible — 1.3.0 config files load unchanged with sensible defaults.
- Version bumped to `1.4.0-beta.1`.
- Added `image-size` as a runtime dependency.

## [1.3.0] - 2026-05-01

### Added

- Two-step model picker (`/vision-proxy pick`): provider first, then model. Replaces the single flat list of 400+ models.
- Current provider is shown first with a ★ marker and pre-selected — picker opens directly on the model list, no need to re-select the same provider every time.
- `← Change provider` option inside the model list to switch providers without restarting the picker.
- `🔍 Type to filter models…` option for providers with more than 8 models. Uses fuzzy character-order matching (e.g. `cs4` matches `Claude Sonnet 4.5`). Single matches are auto-selected.
- `fuzzyMatches()` helper exported from `internal.ts` with full test coverage.

### Changed

- Duplicated picker code between `/vision-proxy pick` and the interactive `Model:` row consolidated into a single `pickVisionModel()` function.

## [1.2.0] - 2026-05-01

### Added

- `/vision-proxy pick` sub-command. Lists vision-capable models from the registry with friendly names and provider tags via `ctx.ui.select`. Avoids typing canonical ids like `accounts/fireworks/models/kimi-k2p6`.
- Interactive `Model:` row in `/vision-proxy` config now opens the same vision-only picker (was raw text input).
- `friendlyModelLabel(config, registry)` helper. Status line and notifies now display `Kimi K2.6 [fireworks]` instead of `fireworks/accounts/fireworks/models/kimi-k2p6` when the registry knows the model.

### Changed

- "Model not found" error now points to `/vision-proxy pick` instead of `/vision-proxy model`.

## [1.1.0] - 2026-05-01

### Changed

- Settings (mode, model, context) now persist across sessions to `~/.pi/agent/vision-proxy.json`. Previously settings were stored only in session entries and lost when starting a new session. Config precedence (highest → lowest): environment variables → session entries → persistent file → defaults.

### Added

- `readPersistentFile()` / `writePersistentFile()` helpers for file-based config storage.
- `fileConfig` parameter on `resolveConfig()` to layer persisted file config between defaults and session entries.
- Tests for persistent file round-trip and layered config resolution.
