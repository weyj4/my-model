# Dataset Creation Algorithms: A Mini Curriculum

A self-directed curriculum covering the data structures and algorithms used to
build pretraining datasets for language models. Motivated by reading the FineWeb
technical report and understanding what's happening under the hood of pipelines
like Common Crawl → FineWeb → training corpus.

The goal is twofold: build intuition for how production datasets are created, and
practice algorithmically interesting problems that are relevant to ML infrastructure
interviews.

---

## Background Reading

Before starting, read these:

- **FineWeb blog post** — `huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1`
  The pipeline you're implementing in miniature. Read the deduplication and quality
  filtering sections carefully.

- **"Deduplicating Training Data Makes Language Models Better"** — Lee et al. 2022
  The suffix array approach to exact substring deduplication. Shows empirically that
  dedup improves model quality and reduces memorization.

- **"CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data"**
  The KenLM perplexity filtering approach. Foundation for many subsequent pipelines.

- **Mining of Massive Datasets, Chapter 3** — Leskovec, Rajaraman, Ullman (free online)
  The canonical reference for MinHash and LSH. Read this before implementing Week 1.

---

## Week 1 — MinHash and Fuzzy Deduplication

### The problem

The web has enormous redundancy — news articles syndicated across hundreds of sites,
boilerplate templates, mirrors. Training on duplicated data wastes compute and
encourages memorization over generalization. But with billions of documents you
can't compare every pair: that's O(n²) comparisons, completely infeasible.

MinHash with Locality Sensitive Hashing (LSH) solves this in roughly O(n).

### Concepts to understand

**Jaccard similarity** — the standard measure of set overlap:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```
For documents, A and B are sets of n-grams (overlapping character sequences).
Two documents that are 80% similar share 80% of their 5-grams.

**N-grams** — overlapping subsequences of length n. For character 5-grams,
"hello world" → `{"hello", "ello ", "llo w", "lo wo", "o wor", " worl", "world"}`.
These are the "shingles" that represent a document as a set.

**MinHash** — the key mathematical trick. If you apply a random hash function to
all elements of a set and take the minimum, the probability that two sets have the
same minimum hash equals their Jaccard similarity:
```
P(min_h(A) == min_h(B)) = J(A, B)
```
By computing many independent MinHash values (112 in FineWeb's case), you get a
compact signature that approximates Jaccard similarity between any two documents.

**LSH banding** — to find candidate duplicate pairs efficiently, split the hash
functions into bands (FineWeb: 14 bands × 8 hashes each). Two documents are
candidate duplicates if all 8 hashes match in any one band. This concentrates
detection probability around a similarity threshold — at 75% similarity, detection
probability is approximately 77%.

### Implementation exercises

1. **Jaccard similarity from scratch**
   - Implement `jaccard(doc_a: str, doc_b: str, n: int) -> float`
   - Test on pairs of similar/dissimilar sentences
   - Verify: two identical documents → 1.0, two completely different → close to 0.0

2. **MinHash signatures**
   - Implement `minhash_signature(doc: str, n_hashes: int, n: int) -> list[int]`
   - Use multiple hash functions (simplest: `hash(str(seed) + ngram)` for seed in range(n_hashes))
   - Verify: estimated Jaccard from signatures ≈ true Jaccard for large n_hashes

3. **LSH candidate finding**
   - Implement `find_candidates(docs: list[str], n_hashes: int, n_bands: int) -> set[tuple]`
   - Two documents are candidates if any band matches exactly
   - Should run much faster than all-pairs comparison

4. **End-to-end deduplication**
   - Download 1,000 Wikipedia article abstracts (or use any text corpus)
   - Run your MinHash deduplication pipeline
   - Manually inspect a few detected duplicates — are they real near-duplicates?
   - Try different n values (3-grams vs 5-grams vs 10-grams) and compare results

### What FineWeb actually did

- 5-grams on character level
- 112 hash functions, 14 bands of 8 hashes each
- Targets 75% similarity threshold
- Removed ~30-40% of documents that passed quality filters
- Run across 84 Common Crawl snapshots — required distributed CPU compute

---

## Week 2 — Suffix Arrays and Exact Substring Deduplication

### The problem

MinHash finds near-duplicate *documents*. But what about repeated *paragraphs*
or *sentences* that appear across otherwise different documents? A news story
might be unique as a whole document but contain a boilerplate legal disclaimer
copied verbatim across thousands of sites.

Suffix arrays solve a different problem: find all repeated substrings of length ≥ k
across a corpus, and remove them. This is the approach from Lee et al. 2022.

### Concepts to understand

**Suffix** — for string "banana", the suffixes are:
```
banana  (position 0)
anana   (position 1)
nana    (position 2)
ana     (position 3)
na      (position 4)
a       (position 5)
```

**Suffix array** — the array of starting positions of all suffixes, sorted
lexicographically. For "banana": `[5, 3, 1, 0, 4, 2]` (positions of "a", "ana",
"anana", "banana", "na", "nana").

**Why this enables substring search** — once suffixes are sorted, identical
prefixes cluster together. Repeated substrings appear as consecutive runs in the
suffix array. You can scan for runs where consecutive suffixes share a common
prefix of length ≥ k in O(n) time using the LCP (Longest Common Prefix) array.

**The scale trick** — concatenate your entire corpus into one string with special
separator tokens between documents. Build one suffix array over the whole thing.
Repeated substrings that cross document boundaries are cross-document duplicates.

### Implementation exercises

1. **Naive suffix array** (O(n log n))
   - Generate all suffixes as (position, suffix) pairs
   - Sort them — Python's sort on strings is O(n log n) comparisons × O(n) per comparison = O(n² log n) worst case, but acceptable for small inputs
   - Return just the positions

2. **LCP array**
   - Given a sorted suffix array, compute the LCP array: `lcp[i]` = length of longest common prefix between suffix at position i and suffix at position i-1
   - Kasai's algorithm does this in O(n) — implement it

3. **Find repeated substrings**
   - Given suffix array + LCP array, find all positions where `lcp[i] >= k`
   - These correspond to repeated substrings of length k
   - Return the positions in the original string

4. **Cross-document deduplication**
   - Take 50 documents, concatenate with separator character (e.g., `\x00`)
   - Build suffix array
   - Find repeated substrings of length ≥ 50 characters
   - Identify which document pairs share the repeated substring
   - Compare results with MinHash — what does each catch?

### Going deeper

The production implementation (used for LLM training corpora) uses SA-IS, a
linear-time suffix array construction algorithm. Understanding it conceptually
is worthwhile — look up the "induced sorting" intuition. You don't need to
implement it from scratch, but understanding why O(n) is achievable for suffix
arrays is a good systems thinking exercise.

The `datasketch` Python library has a production MinHash implementation.
The `pydivsufsort` library wraps the fast C implementation of suffix array
construction if you want to run on larger inputs.

---

## Week 3 — Bloom Filters

### The problem

A web crawler visits billions of URLs. Before fetching each URL it needs to
check: "have I seen this before?" A hash set would work but requires storing
every URL (hundreds of GB for a large crawl). A Bloom filter answers the same
question in O(1) using a fraction of the memory, with a tunable false positive rate.

### Concepts to understand

**Bloom filter** — a bit array of size m, with k hash functions. To insert an
element: hash it k times, set those k bits to 1. To query: hash it k times,
check if all k bits are 1.

Properties:
- **No false negatives** — if something was inserted, all its bits are set
- **Tunable false positives** — probability controlled by m (array size) and k (hash count)
- **No deletion** (standard Bloom filter) — bits can't be unset

False positive probability:
```
p ≈ (1 - e^(-kn/m))^k

where n = number of inserted elements
      m = bit array size  
      k = number of hash functions
```

Optimal k for given m and n: `k = (m/n) × ln(2)`

### Implementation exercises

1. **Implement a Bloom filter from scratch**
   - Bit array using Python's `bytearray` or `bitarray`
   - k hash functions using `hashlib` with different seeds
   - `insert(item)` and `query(item) -> bool` methods

2. **Measure empirical false positive rate**
   - Insert 100,000 URLs
   - Query 100,000 URLs you *didn't* insert
   - Compare empirical FP rate to theoretical formula
   - Sweep over m values, plot FP rate vs memory usage

3. **URL deduplication for a crawl simulation**
   - Generate 1M synthetic URLs (variations on a few domains)
   - Use your Bloom filter to simulate a crawler that skips already-seen URLs
   - Measure memory usage vs a plain Python set

### Extension: Count-Min Sketch

A related probabilistic data structure for frequency estimation — "how many times
have I seen this URL/ngram?" Used in dataset pipelines to find very common phrases
that might be boilerplate. Implement it once you have Bloom filters solid.

---

## Week 4 — Build a Mini Pipeline

### The goal

Build a complete data pipeline from raw web text to a trained model. Compare your
pipeline's output (empirically, via loss curves) against FineWeb. Does your
filtering actually help?

### Data source

Common Crawl WET files — free, ~500MB per file, pre-extracted plain text.
Download 1-2 files from `commoncrawl.org/get-started`:
```bash
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/[...]/wet/[...].warc.wet.gz
```
Or use the `datasets` library to stream a small slice:
```python
from datasets import load_dataset
raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

### Pipeline steps to implement

**Step 1: Basic text extraction and cleaning**
- Remove documents under 100 words or over 100,000 words
- Remove documents where > 30% of characters are non-alphabetic
- Remove documents without terminal punctuation on most lines
- Remove documents containing strings on a blocklist (adult content indicators, etc.)

**Step 2: Language identification**
- Install fastText's language identification model: `lid.176.bin`
- Filter to English documents with confidence ≥ 0.65
- How much of your raw corpus is non-English?

**Step 3: Quality heuristics**
Choose a subset of FineWeb's filters to implement:
- Fraction of lines ending in punctuation (FineWeb threshold: ≥ 0.12)
- Ratio of duplicate lines within a document
- Symbol-to-word ratio (flag documents heavy on `|`, `#`, `*`, etc.)
- Stop word presence (documents lacking "the", "and", "of" are suspicious)

**Step 4: MinHash deduplication**
- Use your Week 1 implementation or `datasketch`
- Run across your filtered corpus
- Log what percentage of documents are removed

**Step 5: Tokenize and save**
- Use your existing `pretokenize.py` as a template
- Save to `.npy` format

**Step 6: Train and compare**
- Train your GPT-2 model on your pipeline's output for 50M tokens
- Train on an equivalent 50M token slice of FineWeb `sample-10BT`
- Compare loss curves in W&B
- Does your filtering produce better or worse training signal?

### What to measure and log

For each pipeline step, log:
- Number of documents before and after
- Percentage removed
- Processing time

This gives you a "pipeline audit" that shows the contribution of each filter —
exactly what FineWeb's ablation tables show, just at much smaller scale.

---

## Interview Relevance

These algorithms come up in ML infrastructure interviews in a few ways:

**Systems design** — "Design a data pipeline to deduplicate a 100TB web crawl"
requires knowing that MinHash/LSH exists, that suffix arrays enable exact substring
matching, and understanding the compute/memory tradeoffs of each approach.

**Algorithm depth** — suffix array construction (SA-IS), Bloom filter math,
and LSH banding parameters show comfort with non-standard data structures beyond
Leetcode defaults.

**ML context** — being able to connect "why do we deduplicate" → "reduces
memorization, improves generalization, equivalent to seeing more diverse data per
compute dollar" shows you understand why these systems exist, not just how they work.

**Empirical validation** — the Week 4 project produces actual experimental
results showing whether your filtering decisions improved training. That's the
kind of artifact that demonstrates both engineering and ML understanding.

---

## Resources

- `datasketch` — Python MinHash and LSH library
- `pydivsufsort` — fast suffix array construction (wraps C library)
- `fasttext` — language identification (`lid.176.bin` model, 900KB, classifies 176 languages)
- `bitarray` — efficient bit arrays for Bloom filter implementation
- Mining of Massive Datasets (free PDF) — chapters 3 (LSH) and 4 (data streams)
- Common Crawl WET files — `commoncrawl.org/get-started`
- `lighteval` — HuggingFace's evaluation library used for FineWeb ablations
