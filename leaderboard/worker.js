// Cloudflare Worker: serves the leaderboard page plus a tiny KV-backed JSON API.
//   GET  /api/leaderboard  -> latest published payload from KV (no-store)
//   POST /api/publish      -> bearer-authed write of the payload into KV
//   *                      -> static assets (public/index.html), via the ASSETS binding

const KV_KEY = "leaderboard:latest";

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

function json(body, status = 200) {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: JSON_HEADERS,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/leaderboard") {
      if (request.method !== "GET") return json({ error: "method not allowed" }, 405);
      const body = await env.LEADERBOARD.get(KV_KEY);
      return json(body === null ? { status: "waiting" } : body);
    }

    if (url.pathname === "/api/publish") {
      if (request.method !== "POST") return json({ error: "method not allowed" }, 405);
      const expected = env.PUBLISH_TOKEN ? `Bearer ${env.PUBLISH_TOKEN}` : null;
      if (!expected || (request.headers.get("authorization") || "") !== expected) {
        return json({ error: "unauthorized" }, 401);
      }
      const text = await request.text();
      try {
        JSON.parse(text);
      } catch {
        return json({ error: "invalid json" }, 400);
      }
      await env.LEADERBOARD.put(KV_KEY, text);
      return json({ ok: true });
    }

    // Everything else is the static page.
    return env.ASSETS.fetch(request);
  },
};
