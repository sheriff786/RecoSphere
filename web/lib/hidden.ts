import "server-only";
import { cache } from "react";
import { api } from "@/lib/api";

/** The user's "Not interested" games (app_api item_feedback), once per request. Every recommendation list sends
 *  them to the engine as `exclude` (P1 phase G). If app_api fails, the page still renders, unfiltered. */
export const hiddenIds = cache(async (sid: string): Promise<string[]> => {
  try {
    return await api.hidden(sid);
  } catch (e) {
    console.error("hidden ids failed", e);
    return [];
  }
});
