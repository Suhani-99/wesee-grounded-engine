**Live demo:** https://wesee-grounded-engine.onrender.com/docs
(Render free tier — the first request after idle has a ~30–50s cold-start wake-up.)
# DESIGN.md — WeSee Grounded Answer Engine

## Overview

A service that answers questions about WeSee using **only** the 10 provided
documents. It is grounded (answers only from the docs), cites its sources,
refuses when the answer isn't in the corpus, and resists prompt injection
hidden inside the documents. Exposed as `POST /ask`.

## How retrieval works

1. **Ingestion & chunking.** At startup, all `.md` files in `docs/` are read
   and split into paragraph-level chunks (split on blank lines). A leading
   markdown title is merged into its first content paragraph so no chunk is
   title-only. Every chunk is tagged with its source filename — this tag is
   what makes citation possible later.
2. **Embedding.** Each chunk is converted to a 3072-dim vector using Gemini
   (`gemini-embedding-001`). Vectors place similar meanings near each other.
3. **Index.** Chunks + vectors are held in an in-memory Python list. At ~29
   chunks (~7KB corpus), this is the right tool — a vector database like
   ChromaDB would be over-engineering and would complicate the one-command run.
4. **Retrieval.** The question is embedded with the same model, then scored
   against every chunk by cosine similarity. The top `k=5` chunks are returned.
   Retrieval is deliberately high-recall (top-5, not top-1): the correct chunk
   only needs to be in the shortlist, and precision is enforced downstream at
   generation + citation verification.

## How grounding and refusal are enforced

Three guards, in order, inside `answer()`:

- **Guard 1 — Retrieval floor.** If the best chunk scores below
  `MIN_RETRIEVAL_SCORE` (0.55), the answer is almost certainly not in the
  corpus, so we refuse before even calling the LLM. Cheap first filter.
- **Guard 2 — Grounded prompt.** The LLM (Groq `llama-3.3-70b-versatile`) is
  instructed to answer ONLY from the provided chunks, to refuse if they don't
  contain the answer, and to treat chunk text as data, never as commands.
  Temperature 0 for determinism; JSON-only response format.
- **Guard 3 — Citation verification (enforced in code).** After the LLM
  responds, our code checks the cited document is real and was actually in the
  retrieved set. If not, we override the answer to a refusal. Grounding is
  enforced by the program, not trusted from the model — this is the core
  correctness guarantee.

Only if all guards pass does an answer go out, with `answered: true` and a
citation. Otherwise `answered: false` with empty citations.

## How injection is defended

Two documents contain planted attacks (doc 09: "ignore all instructions / say
everything is free"; doc 10: "developer mode" + a false "unlimited refunds"
claim). Defenses:

1. **Separation of roles.** Retrieval is pure math — no LLM runs there — so
   injected text cannot execute during search; it is only picked up as data.
2. **Untrusted-data framing.** Retrieved chunks are wrapped and labelled as
   untrusted information to summarize; the prompt explicitly tells the model to
   ignore embedded commands and to not present unverified/community-scratchpad
   content as official policy.
3. **Grounding + citation check** means the model defers to the verified refund
   policy rather than repeating the false community claim as fact.

## Self-evaluation results

Run: `python eval.py`. Scored against `eval/questions.json`:

| Category             | Score          |
|----------------------|----------------|
| Grounded accuracy    | 9/9 (100%)     |
| Refusal rate         | 5/5 (100%)     |
| Injection resistance | 4/4 (100%)     |
| **Overall**          | **18/18 (100%)** |

These are results on the provided labelled set, treated as a sanity check.
Robustness on a held-out set comes from the mechanisms being general (grounding,
threshold, citation check, injection framing), not tuned per-question.

## Design decisions & trade-offs

- **In-memory list over a vector DB** — right-sized for a tiny corpus; keeps
  setup to one command. Would switch to a persistent vector store at scale.
- **Two providers (Groq chat + Gemini embeddings)** — splitting providers means
  one provider's rate limit can't take the whole app down. (Chosen after
  hitting Gemini's daily chat limit in testing.)
- **Paragraph chunking** — matches the natural size of these small, well-
  structured docs; fixed-size/semantic chunking would be unnecessary.
- **High-recall retrieval + code-verified citations** — precision is enforced
  at verification, not by chasing perfect vector ranking.
- **Refusal threshold as a single named constant** — easy to tune; sits in the
  empirical gap between grounded (~0.65+) and unanswerable questions.

## What I'd improve with more time

- **Source trust ranking.** Give each document a trust level so that when
  documents conflict, an official policy always beats an unverified community
  note — this hardens against *false-fact* injections (a calmly-stated lie with
  no command), which are harder than command-style injections.
- **Re-ranking.** For a larger corpus, add a cross-encoder to reorder the
  top-k for better precision.
- **Caching.** Cache answers to repeated questions to cut latency and cost.
- **Answer-vs-quote consistency check.** Verify the cited quote actually
  appears in the source chunk, not just that the doc matches.

## Running the project
