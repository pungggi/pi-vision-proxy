/**
 * Unit tests for vision-proxy pure helpers.
 *
 * Run:
 *   node --experimental-strip-types --test extensions/__tests__/internal.test.ts
 *
 * Requires Node 22+ for native TypeScript stripping. No build / no deps.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
	buildConversationContext,
	CUSTOM_TYPE_CONFIG,
	CUSTOM_TYPE_CONSENT,
	CUSTOM_TYPE_DESCRIPTION,
	DEFAULT_CONFIG,
	envFlags,
	fenceUntrusted,
	findDescriptions,
	hasConsent,
	hashImageData,
	parseModelString,
	pluralImages,
	readEnvOverrides,
	resolveConfig,
	sanitize,
	shouldStripImages,
	splitSubcommand,
	toPiAiImage,
	type VisionConfig,
} from "../internal.ts";

// SessionEntry minimal shape — typed loose because peer dep types are not loaded in test
type Entry = any;

const customEntry = (customType: string, data: unknown): Entry => ({
	type: "custom",
	customType,
	data,
});

const messageEntry = (role: "user" | "assistant", text: string): Entry => ({
	type: "message",
	message: { role, content: [{ type: "text", text }] },
});

describe("parseModelString", () => {
	it("accepts valid provider/model pairs", () => {
		assert.deepEqual(parseModelString("anthropic/claude-sonnet-4-5"), {
			provider: "anthropic",
			modelId: "claude-sonnet-4-5",
		});
		assert.deepEqual(parseModelString("openai/gpt-4o"), { provider: "openai", modelId: "gpt-4o" });
		assert.deepEqual(parseModelString("provider/path/with/slashes"), {
			provider: "provider",
			modelId: "path/with/slashes",
		});
	});

	it("rejects malformed strings", () => {
		assert.equal(parseModelString(""), null);
		assert.equal(parseModelString("/foo"), null);
		assert.equal(parseModelString("foo/"), null);
		assert.equal(parseModelString("noslash"), null);
		assert.equal(parseModelString("provider with space/m"), null);
		assert.equal(parseModelString("provider/has space"), null);
	});
});

describe("sanitize", () => {
	it("clobbers garbage to defaults", () => {
		const out = sanitize({
			mode: "weird" as any,
			provider: "bad provider",
			modelId: "bad model id",
			systemPrompt: "",
			includeContext: "yes" as any,
		});
		assert.equal(out.mode, DEFAULT_CONFIG.mode);
		assert.equal(out.provider, DEFAULT_CONFIG.provider);
		assert.equal(out.modelId, DEFAULT_CONFIG.modelId);
		assert.equal(out.systemPrompt, DEFAULT_CONFIG.systemPrompt);
		assert.equal(out.includeContext, DEFAULT_CONFIG.includeContext);
	});

	it("preserves valid values", () => {
		const cfg: VisionConfig = {
			mode: "always",
			provider: "openai",
			modelId: "gpt-4o",
			systemPrompt: "custom prompt",
			includeContext: false,
		};
		assert.deepEqual(sanitize(cfg), cfg);
	});
});

describe("readEnvOverrides", () => {
	it("returns empty when env unset", () => {
		assert.deepEqual(readEnvOverrides({}), {});
	});

	it("reads valid mode", () => {
		assert.deepEqual(readEnvOverrides({ PI_VISION_PROXY_MODE: "always" }), { mode: "always" });
		assert.deepEqual(readEnvOverrides({ PI_VISION_PROXY_MODE: "off" }), { mode: "off" });
	});

	it("ignores invalid mode", () => {
		assert.deepEqual(readEnvOverrides({ PI_VISION_PROXY_MODE: "bogus" }), {});
	});

	it("reads model string", () => {
		const out = readEnvOverrides({ PI_VISION_PROXY_MODEL: "openai/gpt-4o" });
		assert.equal(out.provider, "openai");
		assert.equal(out.modelId, "gpt-4o");
	});

	it("ignores malformed model string", () => {
		assert.deepEqual(readEnvOverrides({ PI_VISION_PROXY_MODEL: "noslash" }), {});
	});

	it("parses includeContext truthy/falsy values", () => {
		for (const v of ["1", "true", "yes", "on", "TRUE", "On"]) {
			assert.equal(readEnvOverrides({ PI_VISION_PROXY_INCLUDE_CONTEXT: v }).includeContext, true, `truthy ${v}`);
		}
		for (const v of ["0", "false", "no", "off", "FALSE"]) {
			assert.equal(readEnvOverrides({ PI_VISION_PROXY_INCLUDE_CONTEXT: v }).includeContext, false, `falsy ${v}`);
		}
		assert.equal(readEnvOverrides({ PI_VISION_PROXY_INCLUDE_CONTEXT: "garbage" }).includeContext, undefined);
	});
});

describe("envFlags", () => {
	it("reports presence per variable", () => {
		assert.deepEqual(envFlags({}), { mode: false, model: false, context: false });
		assert.deepEqual(
			envFlags({
				PI_VISION_PROXY_MODE: "x",
				PI_VISION_PROXY_MODEL: "y",
				PI_VISION_PROXY_INCLUDE_CONTEXT: "",
			}),
			{ mode: true, model: true, context: true },
		);
	});
});

describe("resolveConfig", () => {
	it("returns defaults with no entries and empty env", () => {
		const cfg = resolveConfig([], {});
		assert.deepEqual(cfg, DEFAULT_CONFIG);
	});

	it("env wins over persisted", () => {
		const entries: Entry[] = [customEntry(CUSTOM_TYPE_CONFIG, { mode: "off" })];
		const cfg = resolveConfig(entries, { PI_VISION_PROXY_MODE: "always" });
		assert.equal(cfg.mode, "always");
	});

	it("uses last persisted entry", () => {
		const entries: Entry[] = [
			customEntry(CUSTOM_TYPE_CONFIG, { mode: "off" }),
			customEntry(CUSTOM_TYPE_CONFIG, { mode: "always" }),
		];
		assert.equal(resolveConfig(entries, {}).mode, "always");
	});
});

describe("fenceUntrusted", () => {
	it("neutralizes opening tag", () => {
		const out = fenceUntrusted("<vision_proxy_description>");
		assert.notEqual(out, "<vision_proxy_description>");
		assert.ok(out.includes("​"), "ZWSP injected");
	});

	it("neutralizes closing tag, case-insensitive", () => {
		const out = fenceUntrusted("</VISION_PROXY_DESCRIPTION>");
		assert.notEqual(out, "</VISION_PROXY_DESCRIPTION>");
	});

	it("leaves unrelated text intact", () => {
		assert.equal(fenceUntrusted("plain text <other>"), "plain text <other>");
	});
});

describe("hashImageData", () => {
	it("is deterministic and 32 chars", () => {
		const a = hashImageData("hello");
		const b = hashImageData("hello");
		assert.equal(a, b);
		assert.equal(a.length, 32);
	});

	it("differs for different inputs", () => {
		assert.notEqual(hashImageData("a"), hashImageData("b"));
	});
});

describe("pluralImages", () => {
	it("singular vs plural", () => {
		assert.equal(pluralImages(1), "1 image");
		assert.equal(pluralImages(0), "0 images");
		assert.equal(pluralImages(5), "5 images");
	});
});

describe("splitSubcommand", () => {
	it("splits sub and value with arbitrary whitespace", () => {
		assert.deepEqual(splitSubcommand("model anthropic/claude"), { sub: "model", value: "anthropic/claude" });
		assert.deepEqual(splitSubcommand("model    anthropic/claude  "), {
			sub: "model",
			value: "anthropic/claude",
		});
		assert.deepEqual(splitSubcommand("CONSENT YES"), { sub: "consent", value: "YES" });
	});

	it("handles bare sub with no value", () => {
		assert.deepEqual(splitSubcommand("consent"), { sub: "consent", value: "" });
	});

	it("handles empty input", () => {
		assert.deepEqual(splitSubcommand(""), { sub: "", value: "" });
	});
});

describe("buildConversationContext", () => {
	it("returns empty for no message entries", () => {
		assert.equal(buildConversationContext([]), "");
	});

	it("concatenates user and assistant text in order", () => {
		const entries: Entry[] = [
			messageEntry("user", "first"),
			messageEntry("assistant", "reply"),
			customEntry("other", {}),
		];
		const out = buildConversationContext(entries);
		assert.equal(out, "User: first\nAssistant: reply");
	});

	it("keeps only the last 8 message entries", () => {
		const entries: Entry[] = [];
		for (let i = 0; i < 12; i++) entries.push(messageEntry("user", `m${i}`));
		const out = buildConversationContext(entries);
		const lines = out.split("\n");
		assert.equal(lines.length, 8);
		assert.equal(lines[0], "User: m4");
		assert.equal(lines[7], "User: m11");
	});

	it("truncates assistant content to 500 chars", () => {
		const long = "x".repeat(800);
		const out = buildConversationContext([messageEntry("assistant", long)]);
		assert.ok(out.startsWith("Assistant: "));
		assert.equal(out.length, "Assistant: ".length + 500);
	});

	it("truncates total to last 3000 chars with ellipsis", () => {
		const entries: Entry[] = [];
		for (let i = 0; i < 8; i++) entries.push(messageEntry("user", "y".repeat(490)));
		const out = buildConversationContext(entries);
		assert.ok(out.length <= 3001);
		assert.ok(out.startsWith("…"));
	});
});

describe("findDescriptions", () => {
	it("collects hash → description from custom entries", () => {
		const entries: Entry[] = [
			customEntry(CUSTOM_TYPE_DESCRIPTION, { hash: "abc", description: "desc-a" }),
			customEntry(CUSTOM_TYPE_DESCRIPTION, { hash: "def", description: "desc-b" }),
			customEntry("other", {}),
			customEntry(CUSTOM_TYPE_DESCRIPTION, { hash: "", description: "skip" }),
		];
		const map = findDescriptions(entries);
		assert.equal(map.size, 2);
		assert.equal(map.get("abc"), "desc-a");
		assert.equal(map.get("def"), "desc-b");
	});
});

describe("hasConsent", () => {
	it("returns false with no entries", () => {
		assert.equal(hasConsent([]), false);
	});

	it("uses the most recent consent entry", () => {
		const entries: Entry[] = [
			customEntry(CUSTOM_TYPE_CONSENT, { granted: true }),
			customEntry(CUSTOM_TYPE_CONSENT, { granted: false }),
		];
		assert.equal(hasConsent(entries), false);

		const granted: Entry[] = [
			customEntry(CUSTOM_TYPE_CONSENT, { granted: false }),
			customEntry(CUSTOM_TYPE_CONSENT, { granted: true }),
		];
		assert.equal(hasConsent(granted), true);
	});
});

describe("toPiAiImage", () => {
	it("passes through new shape", () => {
		const img = { type: "image", data: "AAAA", mimeType: "image/png" } as any;
		assert.deepEqual(toPiAiImage(img), { type: "image", data: "AAAA", mimeType: "image/png" });
	});

	it("converts legacy { source: { data, mediaType } } shape", () => {
		const legacy = { source: { data: "BBBB", mediaType: "image/jpeg" } };
		assert.deepEqual(toPiAiImage(legacy), { type: "image", data: "BBBB", mimeType: "image/jpeg" });
	});

	it("throws on unsupported shape", () => {
		assert.throws(() => toPiAiImage({} as any), /Unsupported image content shape/);
	});
});

describe("shouldStripImages", () => {
	const cfg = (mode: VisionConfig["mode"]): VisionConfig => ({ ...DEFAULT_CONFIG, mode });

	it("off → never strip", () => {
		assert.equal(shouldStripImages(cfg("off"), undefined), false);
		assert.equal(shouldStripImages(cfg("off"), ["image", "text"]), false);
	});

	it("always → always strip", () => {
		assert.equal(shouldStripImages(cfg("always"), undefined), true);
		assert.equal(shouldStripImages(cfg("always"), ["image"]), true);
	});

	it("fallback → strip only when model lacks image input", () => {
		assert.equal(shouldStripImages(cfg("fallback"), ["text"]), true);
		assert.equal(shouldStripImages(cfg("fallback"), undefined), true);
		assert.equal(shouldStripImages(cfg("fallback"), ["text", "image"]), false);
	});
});
