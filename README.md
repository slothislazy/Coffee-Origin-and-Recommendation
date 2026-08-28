# Coffee Origin Classification and Content-Based Recommendation

A Natural Language Processing framework for analyzing expert coffee reviews. The system performs two tasks from the same review corpus:

1. **Origin classification** — predicting a coffee's geographic region from its sensory description.
2. **Content-based recommendation** — retrieving coffees whose tasting notes match a free-form flavor query.

This repository accompanies the MSc thesis *"Leveraging Natural Language Processing Techniques for Coffee Origin Classification and Content-Based Recommendation from Coffee Reviews."*

---

## Headline results

| Task | Best configuration | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Country-level classification (14 classes) | RoBERTa-base × 3 (mean) | 0.7466 | 0.6429 |
| Regional 4-way classification | RoBERTa-base × 3 soft-vote ensemble | **0.8497** | **0.8344** |
| Regional 5-way classification (US split out) | RoBERTa-base × 3 (mean) | 0.8143 | 0.7999 |

| Recommendation encoder | Precision@10 | nDCG@10 | MAP@10 |
|---|---|---|---|
| TF-IDF (baseline) | 0.1687 | 0.1760 | 0.0910 |
| SBERT MiniLM-L6 | 0.1652 | 0.1705 | 0.0886 |
| BGE-base-en-v1.5 | 0.1794 | 0.1856 | **0.1018** |
| E5-base-v2 | **0.1804** | **0.1866** | 0.1011 |

---

## Methodological highlights

- **Four-tier data-leakage scrubbing pipeline** masks 403 origin-revealing terms (country names and adjectivals, regional aliases, cultivar names, producer-context terms) before training. Without it, classifiers reach 91% accuracy by lexical memorization rather than learning from flavor evidence.
- **Multi-section input** concatenates the four primary review fields (*Blind Assessment*, *Notes*, *Who Should Drink It*, *Bottom Line*) so flavor information distributed across the whole review is captured, not just the tasting note.
- **Two-phase study design:** Phase 1 benchmarks five text-representation paradigms (TF-IDF, static embeddings, frozen contextual embeddings, fine-tuned transformers, LLM zero-shot). Phase 2 refines the winning paradigm across six design axes (time window, label granularity, pipeline architecture, input modality, training strategy, backbone capacity).

---

## Repository structure

```
.
├── 00_Coffee_Review_Full_Scrape_and_Parse.ipynb     Data scraping
├── 01_Preprocessing_Scraped_Data.ipynb              Cleaning + scrubbing pipeline
│
├── 02_TFIDF.ipynb                                   Phase 1: TF-IDF baseline
├── 03_Word2Vec.ipynb                                Phase 1: static embeddings
├── 04_Origin_Classification_Transformer_Embedding.ipynb   Phase 1: frozen BERT/SBERT
├── 04.1_RoBERTa.ipynb                               Phase 1: RoBERTa fine-tuning
├── 04.2_Origin_Classification_ModernBERT.ipynb      Phase 1: ModernBERT fine-tuning
├── 04.3-04.5_*.ipynb                                Scrubbing ablation (unscrubbed → scrubbed-plus)
├── 06_LLM_ZeroShot.ipynb                            Phase 1: Qwen2.5-7B zero-shot
├── 07_SeedHarness.ipynb                             Multi-seed evaluation harness
│
├── 08_Origin_Recent10yr_MinSamples60.ipynb          Phase 2: time-window variants
├── 09_Origin_Recent5yr_MinSamples40.ipynb
├── 10_Origin_Regional_4Way.ipynb                    Phase 2: regional granularity
├── 11_Origin_Regional_5Way.ipynb
├── 12_Origin_Hierarchical_Region_to_Country.ipynb   Phase 2: hierarchical pipeline
├── 13_Origin_Regional_4Way_TextPlusScores.ipynb     Phase 2: multi-modal fusion
├── 14_Origin_Regional_4Way_Ensemble.ipynb           Phase 2: soft-vote ensemble (best)
├── 15_Origin_Regional_4Way_RoBERTaLarge.ipynb       Phase 2: larger backbone
│
├── 05_Aspect_Guided_Retrieval.ipynb                 Recommendation: aspect extraction
├── 05.1_Personalized_Recommendation_System_BGE.ipynb  Recommendation: BGE/E5/SBERT/TF-IDF
│
├── 16_Results_Summary.ipynb                         Aggregates all artifacts into final tables
│
├── Data/                  Parsed CSVs (9,009 reviews × 16 columns)
├── artifacts/             Per-experiment outputs (results.json, metrics, predictions)
└── coffee_reviews_text/   Raw scraped review pages (one .txt per URL)
```

---

## Dataset

- **Source:** scraped from [coffeereview.com](https://www.coffeereview.com/) (Feb 1997 – Mar 2026).
- **Size:** 9,009 reviews × 16 columns.
- **Coverage:** 46 distinct countries of origin.
- **Fields:** identifiers (URL, roaster, coffee name, review date), categorical (country, roast level), numerical scores (overall rating + Aroma / Acidity / Body / Flavor / Aftertaste), and four unstructured text fields (*Blind Assessment*, *Notes*, *Who Should Drink It*, *Bottom Line*).

The scraping pipeline is maintained as a separate repository: [Coffee-Review-Data-Scraping](https://github.com/slothislazy/Coffee-Review-Data-Scraping).

---

## Reproducing results

### Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
.venv\Scripts\activate              # Windows PowerShell

pip install torch transformers sentence-transformers scikit-learn pandas numpy gensim beautifulsoup4 requests tqdm matplotlib seaborn jupyter
```

GPU recommended for fine-tuning (notebooks 04.x, 08–15). CPU is sufficient for TF-IDF, static embeddings, and recommendation experiments.

### Notebook execution order

1. **Data:** Run `00` to scrape, then `01` to preprocess. *(Or skip and use the included `Data/coffee_reviews_parsed.csv`.)*
2. **Phase 1 baselines:** Run notebooks `02`–`07` to reproduce Table II of the paper.
3. **Scrubbing ablation:** Run `04.3`, `04.4`, `04.5` to reproduce Table III.
4. **Phase 2 refinements:** Run `08`–`15` to reproduce Table IV. The 3-seed ensemble (notebook `14`) is the overall best configuration.
5. **Recommendation:** Run `05` then `05.1` to reproduce Table V.
6. **Summary:** Run `16` to regenerate aggregated results tables.

Each experiment writes its outputs to `artifacts/<experiment_name>/`, including `results.json`, per-seed metrics, and trained model checkpoints where applicable.

---

## Models used

| Component | Model | Source |
|---|---|---|
| Classification (primary) | RoBERTa-base | `roberta-base` |
| Classification (capacity ablation) | RoBERTa-large | `roberta-large` |
| Classification (comparison) | ModernBERT-base | `answerdotai/ModernBERT-base` |
| Frozen embeddings | BERT-base, all-MiniLM-L6-v2 | `bert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2` |
| LLM zero-shot baseline | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Retrieval (primary) | BGE-base-en-v1.5, E5-base-v2 | `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2` |


---

## Acknowledgments

Thesis supervised by Samuel Philip, with contributions from Hansel Aditia Hartono. Coffee Review content remains the intellectual property of Coffee Review and its contributors; the scraped data is shared for research purposes only.
