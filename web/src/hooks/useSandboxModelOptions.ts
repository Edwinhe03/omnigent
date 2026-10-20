import { useQueries, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { authenticatedFetch } from "@/lib/identity";
import type { NativeModelOption } from "@/lib/types";

export interface SandboxModelOptions {
  models: NativeModelOption[];
  configured: boolean;
  configuration_revision: string | null;
  provider_label: string | null;
  default_model: string | null;
  status: "ready" | "empty" | "unavailable" | "unconfigured";
  error?: string;
}

export function sandboxModelOptionsKey(
  provider: string | null,
  harness: string | null,
  agentId: string | null,
  user: string | null,
) {
  return ["sandbox-model-options", user, provider, harness, agentId] as const;
}

async function fetchSandboxModelOptions(
  provider: string,
  harness: string,
  agentId: string | null,
): Promise<SandboxModelOptions> {
  const params = new URLSearchParams();
  if (agentId !== null) params.set("agent_id", agentId);
  const response = await authenticatedFetch(
    `/v1/sandbox-providers/${encodeURIComponent(provider)}/harnesses/${encodeURIComponent(harness)}/model-options?${params}`,
  );
  if (!response.ok) {
    let message = `Could not load sandbox models (HTTP ${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Keep the status message when the response is not JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as SandboxModelOptions;
}

/** Preview the future host's configuration without provisioning a sandbox. */
export function useSandboxModelOptions(
  provider: string | null,
  harness: string | null,
  agentId: string | null,
  user: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: sandboxModelOptionsKey(provider, harness, agentId, user),
    queryFn: () => fetchSandboxModelOptions(provider!, harness!, agentId),
    enabled: enabled && provider !== null && harness !== null,
    staleTime: 15_000,
    refetchInterval: enabled && provider !== null && harness !== null ? 15_000 : false,
    retry: false,
  });
}

/**
 * Harnesses whose sandbox model preview reports no usable model.
 *
 * Only a definitive "empty" counts; a failed or "unavailable" preview keeps the
 * harness selectable, so a catalog blip never blocks a working option.
 */
export function useSandboxHarnessesWithoutModels(
  provider: string | null,
  harnesses: ReadonlySet<string>,
  user: string | null,
  enabled: boolean,
): ReadonlySet<string> {
  const ordered = useMemo(() => [...harnesses], [harnesses]);
  const results = useQueries({
    queries: ordered.map((harness) => ({
      queryKey: sandboxModelOptionsKey(provider, harness, null, user),
      queryFn: () => fetchSandboxModelOptions(provider!, harness, null),
      enabled: enabled && provider !== null,
      staleTime: 15_000,
      retry: false,
    })),
  });
  const emptyKey = ordered
    .filter((_, index) => results[index]?.data?.status === "empty")
    .join("\n");
  return useMemo(() => new Set(emptyKey === "" ? [] : emptyKey.split("\n")), [emptyKey]);
}
