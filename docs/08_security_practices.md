# WeSee — Security Practices

WeSee follows a least-privilege access model. Production access requires SSO plus
hardware-key MFA and is limited to on-call engineers. All access is logged.

Data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher.
Secrets are stored in a managed secrets vault, never in source code. WeSee runs
automated dependency scanning and quarterly third-party penetration tests.

Customers on the Scale plan can request SSO (SAML) and enforce workspace-wide MFA.
WeSee targets a 99.9% monthly uptime for the Publish API. Security issues can be
reported to security@weseegpt.com.
