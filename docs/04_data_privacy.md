# WeSee — Data Privacy and Retention

WeSee stores customer content in encrypted form (AES-256 at rest, TLS 1.2+ in
transit). Each workspace's retrieval index is logically isolated; content from one
workspace is never used to answer another workspace's queries.

Draft content is retained for as long as the workspace is active. When a workspace
is deleted, all associated drafts and indexes are permanently removed within 30
days. Audit logs are retained for 12 months.

WeSee is processor, not controller, for customer content. WeSee does not sell
customer data and does not use customer content to train foundation models.
Customers in the EU are served from the Frankfurt region; all others default to
the Mumbai region.
