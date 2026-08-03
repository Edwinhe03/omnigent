import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveServerInfo, sandboxOptionLabel, sandboxProviderOptions } from "./capabilities";
import type { ServerInfo } from "./capabilities";

/** A ServerInfo with only the sandbox fields a test cares about set. */
function info(overrides: Partial<ServerInfo>): ServerInfo {
  return {
    accounts_enabled: false,
    single_user: true,
    login_url: null,
    needs_setup: false,
    databricks_features: false,
    managed_sandboxes_enabled: true,
    sandbox_provider: null,
    sandbox_providers: [],
    sharing_mode: "on",
    public_sharing_enabled: true,
    server_version: null,
    smart_routing_enabled: false,
    harness_install_enabled: false,
    installable_harnesses: [],
    dictation_available: false,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("sandboxProviderOptions", () => {
  it("yields one entry per configured provider, in order", () => {
    // The order the operator configured is the order the user sees.
    expect(
      sandboxProviderOptions(
        info({ sandbox_provider: "modal", sandbox_providers: ["modal", "e2b", "daytona"] }),
      ),
    ).toEqual(["modal", "e2b", "daytona"]);
  });

  it("falls back to the single provider when the list is empty", () => {
    // An older server reports only the scalar, and must still offer a row.
    expect(
      sandboxProviderOptions(info({ sandbox_provider: "modal", sandbox_providers: [] })),
    ).toEqual(["modal"]);
  });

  it("yields one unnamed row when the server names no provider", () => {
    // An embedding deployment may enable sandboxes without naming one.
    const options = sandboxProviderOptions(info({}));
    expect(options).toEqual([null]);
    expect(sandboxOptionLabel(options[0])).toBe("New Sandbox");
  });
});

describe("resolveServerInfo", () => {
  it("keeps the provider list from the probe", async () => {
    // Regression: the probe rebuilds ServerInfo field by field, so a
    // forgotten field is dropped before any component sees it.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          managed_sandboxes_enabled: true,
          sandbox_provider: "modal",
          sandbox_providers: ["modal", "e2b"],
        }),
      }),
    );
    const { resolveServerInfo: resolve } = await import("./capabilities");
    const resolved = await resolve();
    expect(resolved.sandbox_providers).toEqual(["modal", "e2b"]);
  });

  it("defaults the provider list to empty when the server omits it", async () => {
    // Must land as [] so sandboxProviderOptions can read .length.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ managed_sandboxes_enabled: true, sandbox_provider: "modal" }),
      }),
    );
    const { resolveServerInfo: resolve } = await import("./capabilities");
    const resolved = await resolve();
    expect(resolved.sandbox_providers).toEqual([]);
    expect(sandboxProviderOptions(resolved)).toEqual(["modal"]);
  });

  it("drops non-string entries from the provider list", async () => {
    // /v1/info is untrusted input, as with installable_harnesses.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          managed_sandboxes_enabled: true,
          sandbox_providers: ["modal", 7, null, "e2b"],
        }),
      }),
    );
    const { resolveServerInfo: resolve } = await import("./capabilities");
    const resolved = await resolve();
    expect(resolved.sandbox_providers).toEqual(["modal", "e2b"]);
  });
});

// Each probe test re-imports the module for a fresh cache, so the
// top-level import needs a reference.
void resolveServerInfo;
