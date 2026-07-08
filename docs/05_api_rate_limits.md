# WeSee — API Rate Limits

The Publish API and Generation API are rate limited per workspace, by plan:

- **Growth**: 60 requests/minute, 5,000 generations/month.
- **Scale**: 240 requests/minute, 30,000 generations/month.

Starter plans do not include API access. Requests over the per-minute limit
receive HTTP 429 with a `Retry-After` header indicating seconds to wait. Bulk
generation endpoints accept up to 50 items per request. API keys are scoped to a
single workspace and can be rotated from Settings → API Keys. A rotated key stops
working immediately.
