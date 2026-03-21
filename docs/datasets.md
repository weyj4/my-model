# Pretraining Datasets: Common Crawl, Derivatives, and FineWeb

## Common Crawl

Common Crawl is a nonprofit that has been crawling the public web since 2008 and releasing the raw results as open data. Each monthly crawl captures somewhere between 2–4 billion web pages. The cumulative archive is in the hundreds of terabytes and is hosted on AWS S3, freely accessible.

### What it actually contains

Each crawl release ships in three file formats:

- **WARC** (Web ARChive): The raw format. Contains full HTTP request/response pairs — headers, HTML, everything. One WARC file can be hundreds of GB. This is the primary artifact.
- **WAT**: Metadata extracted from WARC — HTTP headers, links, detected language, etc. — without the full page content. Useful for filtering without reading the raw HTML.
- **WET**: Plain text extracted from the HTML in the WARC files, one document per record. The simplest entry point if you just want text.

The WET files are what most NLP pipelines start from — they've already stripped HTML tags and extracted visible text. However the extraction is naive (no boilerplate removal, no quality filtering), so you get a lot of navigation menus, cookie banners, SEO spam, and garbled text.

### Scale

A single monthly crawl is roughly 70–80TB of WARC. The WET subset (text only) is around 20TB per crawl. Most dataset pipelines work from a subset of crawls rather than the full history.

---

## Derivative Datasets

### C4 (Colossal Clean Crawled Corpus)

Created by Google for training T5 (2019). Starts from WET files, applies relatively simple heuristics:

- Keep only lines ending in a terminal punctuation mark
- Remove pages with bad words (a blocklist)
- Deduplicate at the three-sentence span level
- Keep only English pages (via langdetect)

Result: ~750GB of text, ~156B tokens. The filtering is quite crude by modern standards — it was designed to be fast and reproducible, not optimal. Still widely used as a baseline because it's well-characterized.

### The Pile

Created by EleutherAI (2021). Importantly, it is **not just Common Crawl** — it's a weighted mixture of 22 sources:

| Source | Approximate weight |
|---|---|
| Pile-CC (filtered Common Crawl) | ~18% |
| Books3 (books, now legally contested) | ~12% |
| OpenWebText2 | ~10% |
| GitHub code | ~8% |
| Wikipedia (en) | ~5% |
| PubMed, ArXiv, StackExchange, etc. | remainder |

The diversity was intentional — the hypothesis was that mixing high-quality domain-specific data (code, papers, books) improves reasoning and knowledge even at low mixing weights. This influenced all subsequent pretraining data design. GPT-NeoX, GPT-J, and many other open models were trained on The Pile.

The Books3 component is now legally contested (copyright claims), and some derivative models have had to retrain without it.

### RedPajama

Created by Together AI (2023) as an open reproduction of LLaMA's training data mixture. LLaMA's data mix was described in the paper but not released; RedPajama reverse-engineered it:

- Common Crawl (filtered)
- C4
- GitHub
- Wikipedia
- Books (Project Gutenberg, legal)
- ArXiv
- StackExchange

RedPajama-v2 (late 2023) went further — 30 trillion tokens from 84 Common Crawl snapshots, with quality signals attached (but filtering left to the user). It's more of a "bring your own filter" dataset than a ready-to-use corpus.

### DCLM (DataComp for Language Models)

Created by a multi-institution collaboration (2024). The key insight was treating dataset curation as an empirical benchmark problem: try many different filtering pipelines, train small models on each, measure downstream task performance, keep what works.

Their best pipeline uses a **fastText classifier trained on high-quality reference data** (primarily OpenHermes, a curated instruction dataset) to score and filter Common Crawl pages. The intuition is: pages that look like high-quality human-written text score high; spam, boilerplate, SEO content score low.

DCLM-Baseline is roughly 3.8 trillion tokens and trains models that score competitively with Llama 3 and Mistral on standard benchmarks at equivalent token counts. It's the most serious empirical challenge to FineWeb as a default choice.

---

## FineWeb

Created by HuggingFace (2024), released with a detailed technical report. The goal was to produce the best-quality openly documented pretraining dataset, with ablations showing which filtering decisions actually matter.

### Source

84 Common Crawl snapshots from 2013 to 2024, starting from the WET text extracts.

### Pipeline: step by step

**1. URL filtering**

Block lists of known low-quality domains (adult content, spam, malware). Applied before any content is read — cheap and fast.

**2. trafilatura text extraction**

Rather than using Common Crawl's built-in WET text extraction, HuggingFace re-extracts text from the raw WARC HTML using **trafilatura**, a Python library specifically designed for article extraction. It uses a combination of heuristics and an XPath-based content scoring algorithm to identify the "main content" of a page and discard navigation, ads, footers, and boilerplate. This is a meaningful quality improvement over the naive WET extraction.

**3. Language identification**

**fastText's language identification model** (lid.176) assigns a language probability to each document. Only documents classified as English with high confidence are kept. FastText is used here because it's extremely fast — you can classify billions of documents in hours on a single machine.

**4. Quality filtering heuristics**

A battery of hand-crafted heuristics inspired by C4 and prior work, applied at both the document and line level:

- Minimum and maximum document length
- Ratio of lines ending in punctuation
- Ratio of duplicate lines and duplicate paragraphs
- Symbol-to-word ratio (flags documents heavy on special characters)
- Ratio of words containing alphabetic characters
- "Stop word" presence (function words like "the", "and" — their absence suggests non-natural-language text)

Each filter has a tuned threshold. HuggingFace published ablations showing which filters help and which are neutral.

**5. MinHash deduplication**

This is the most computationally expensive step. **MinHash with Locality Sensitive Hashing (LSH)** identifies near-duplicate documents across the entire corpus — not exact duplicates, but documents that are very similar (e.g., the same news article syndicated across hundreds of sites).

MinHash works by converting each document into a set of n-gram shingles (overlapping character sequences), then computing a compact "sketch" (the MinHash signature) that preserves the Jaccard similarity between documents. LSH then groups documents with similar signatures into candidate buckets for comparison. Documents above a similarity threshold are considered duplicates; only one is kept.

This step removes roughly 30–40% of documents that pass the quality filters, suggesting that web crawl data has enormous redundancy.

**6. "Educational quality" classifier (FineWeb-Edu variant)**

The `FineWeb-Edu` variant adds one more step: a **linear classifier trained on Llama-3-70B annotations**. HuggingFace prompted Llama-3-70B to score ~450k documents on "educational value" (0–5 scale), then trained a lightweight classifier on those annotations. Documents scoring ≥ 3 are kept.

This produces a much smaller but higher-quality subset (~1.3 trillion tokens vs ~15 trillion for the base FineWeb). Models trained on FineWeb-Edu outperform those trained on FineWeb on knowledge-intensive benchmarks (MMLU, ARC) despite seeing fewer tokens.

### Packaged subsets

| Name | Approximate tokens | Approximate size |
|---|---|---|
| `sample-10BT` | 10 billion | ~40GB |
| `sample-100BT` | 100 billion | ~400GB |
| `sample-350BT` | 350 billion | ~1.4TB |
| `default` (full) | ~15 trillion | ~44TB |

---

## Practical notes for training runs

For a learning project targeting GPT-2 scale (~100M parameters), `sample-10BT` gives you far more data than you'll use in a single run. The standard recipe for 124M parameters is roughly 10 billion tokens for a Chinchilla-optimal run, but typical learning-rate / architecture experiments use 1–2B tokens and stop early.

`sample-100BT` is more appropriate if you want to run serious ablations across architectural variants without data being a bottleneck.

FineWeb-Edu is worth knowing about — if you care about benchmark performance rather than just loss curves, training on Edu rather than base FineWeb gives meaningful gains for knowledge tasks.
