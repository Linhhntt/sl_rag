# Evidential Processor Integration

This file explains the parts (protocols, algorithms, fomulation) that we will develop more from fundamental RAG systems.

## 1. Source Trustworthiness Calculation

## 2. Retrieval-to-opinion
- [Best Match 25](https://www.luigisbox.it/glossario-ricerca/bm25/) (BM25): a ranking algorithm that identifies the relevance of a document to a given query. (term or lexical matching) [[1]](https://link.springer.com/article/10.1007/s12626-022-00103-1)
- [NLI consistency](https://aclanthology.org/2025.emnlp-main.1152/) determine logical relationships between pairs of sentences. (semantic: entailment vs. contradiction or neutral)
    - Based on axioms (Positively & Negatively Consistent) in [UncertaintyRAG](https://aclanthology.org/2025.findings-acl.852.pdf), when the context *supports*, uncertainty should *decrease*. In contrast, if context *contradicts*, uncertainty should *increase*.
    - Example: 
    ```
        P: "The dog is sleeping on the couch."
        H: "The dog is resting." → Entailment
        H: "The dog is running." → Contradiction  
        H: "The dog likes bones." → Neutral
    ```
- Example
```
- Query: "COVID vaccine efficacy"
- BM25 chunk: "Vaccines don't work at all" ✓ high lexical score
- NLI: "Vaccines don't work" contradicts "efficacy" → c=-0.8 → flags bad chunk
```

- Algorithm
```
r_norm = BM25(q, chunk) / max_BM25    # [0,1] lexical
c = NLI_entail(q + chunk) - NLI_contradict(q + chunk) # [-1,1] semantic

b = max(0, (r_norm + max(0, c)) / 2)  # Evidence FOR
d = max(0, (1-r_norm + max(0, -c)) / 2) # Evidence AGAINST  
u = 1 - (b + d)                       # Residual ignorance
```

# REFERENCES
1. [Legal Information Retrieval and Entailment Based on BM25, Transformer and Semantic Thesaurus Methods](https://link.springer.com/article/10.1007/s12626-022-00103-1)



