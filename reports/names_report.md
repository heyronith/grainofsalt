# Names report (Amendment A2)

## Model training data (OLMo-2-0425-1B)

- **Pretraining:** OLMo-mix-1124 (~3.9T tokens / 4T stage-1).
- **Mid-training:** Dolmino-mix-1124 (50B tokens; ingredient 3; no model merging).
- Source: Hugging Face model card `allenai/OLMo-2-0425-1B` (queried 2026-11-18).

## Search index

- **Index name:** `v4_olmo-2-1124-13b-instruct_llama`
- **API:** `https://api.infini-gram.io/`
- **Coverage:** OLMo-mix-1124 + Dolmino mid-training + post-training (OLMo-2-1124-13B-Instruct corpus)
- **Query date:** 2026-11-18
- **Tokenizer note:** Llama-2 tokenizer (infini-gram); our name token counts use OLMo-2 tokenizer

## Pipeline stats

- **n_familiarity_scored:** 1110
- **n_after_kl_screen:** 787
- **n_final:** 660
- **token_distribution:** {4: 346, 5: 199, 3: 115}
- **max_kl_selected_max:** 0.5549103749554197
- **max_kl_selected_mean:** 0.21870403038069036
- **amendment:** A2

## Pool p_ref (base-model reference)

- **birthplace:** max p_ref = 0.1756; target ≤0.15; values = ['Lisbon', 'Oslo', 'Bern', 'Naples', 'Valencia', 'Porto', 'Sofia', 'Bergen', 'Lyon', 'Bordeaux', 'Geneva', 'Zurich', 'Basel', 'Graz', 'Florence', 'Venice', 'Split', 'Cork', 'Swansea', 'Hull']
  - distribution: Lisbon=0.0271, Oslo=0.0214, Bern=0.0357, Naples=0.0450, Valencia=0.0440, Porto=0.0044, Sofia=0.0073, Bergen=0.0287, Lyon=0.0218, Bordeaux=0.0091, Geneva=0.1532, Zurich=0.0173, Basel=0.0062, Graz=0.0053, Florence=0.1756, Venice=0.0503, Split=0.0093, Cork=0.1100, Swansea=0.1347, Hull=0.0935
- **hobby:** max p_ref = 0.1112; target ≤0.15; values = ['fencing', 'pottery', 'chess', 'sailing', 'weaving', 'carving', 'brewing', 'cycling', 'swimming', 'hiking', 'camping', 'climbing', 'skiing', 'skating', 'dancing', 'singing', 'embroidery', 'crochet', 'blogging', 'magic']
  - distribution: fencing=0.0210, pottery=0.0400, chess=0.0502, sailing=0.1112, weaving=0.0554, carving=0.0398, brewing=0.0287, cycling=0.0479, swimming=0.0945, hiking=0.0639, camping=0.0311, climbing=0.0691, skiing=0.0460, skating=0.0189, dancing=0.0559, singing=0.0851, embroidery=0.0286, crochet=0.0309, blogging=0.0273, magic=0.0545
- **employer:** max p_ref = 0.3869; target ≤0.15; values = ['Zenthall Materials', 'Pelgrave Metrology', 'Rindorf Components', 'Eskvale Optica', 'Quenwick Technologies', 'Perndale Analytics', 'Caldrun Logistics', 'Kelvorn Processes', 'Torvale Metrology', 'Suthmere Solutions', 'Fenridge Metrology', 'Nalwick Bureau', 'Vorland Consortia', 'Gorswick Consortia', 'Bramsol Metrology', 'Jadvern Assemblies', 'Hespvale Materials', 'Velloran Engineering', 'Walperton Materials', 'Lobwick Metrology']
  - distribution: Zenthall Materials=0.0002, Pelgrave Metrology=0.0263, Rindorf Components=0.0317, Eskvale Optica=0.0208, Quenwick Technologies=0.3869, Perndale Analytics=0.0599, Caldrun Logistics=0.0722, Kelvorn Processes=0.0051, Torvale Metrology=0.0116, Suthmere Solutions=0.1030, Fenridge Metrology=0.0175, Nalwick Bureau=0.0442, Vorland Consortia=0.0199, Gorswick Consortia=0.0013, Bramsol Metrology=0.0000, Jadvern Assemblies=0.0000, Hespvale Materials=0.0000, Velloran Engineering=0.1784, Walperton Materials=0.0208, Lobwick Metrology=0.0001
- **university:** max p_ref = 0.2123; target ≤0.15; values = ['Mossford Institute', 'Fenridge Institute', 'Vorland Institute', 'Ashmere Institute', 'Gorswick Institute', 'Eskvale College', 'Bramsol College', 'Redmere Institute', 'Dunmere Institute', 'Belcrow College', 'Ellswick College', 'Brindle Institute', 'Arkwell Institute', 'Lobwick College', 'Elmworth Institute', 'Pelgrave College', 'Torvale College', 'Jadvern Institute', 'Keymere College', 'Bellwick College']
  - distribution: Mossford Institute=0.0046, Fenridge Institute=0.0073, Vorland Institute=0.0065, Ashmere Institute=0.1864, Gorswick Institute=0.0096, Eskvale College=0.1276, Bramsol College=0.0000, Redmere Institute=0.2123, Dunmere Institute=0.0153, Belcrow College=0.0466, Ellswick College=0.0045, Brindle Institute=0.1763, Arkwell Institute=0.0020, Lobwick College=0.0002, Elmworth Institute=0.0205, Pelgrave College=0.0477, Torvale College=0.0489, Jadvern Institute=0.0000, Keymere College=0.0100, Bellwick College=0.0737

## KL histograms (approx, 10 bins)

### birthplace
- [0.0124, 0.1612): 660
- [0.1612, 0.3099): 321
- [0.3099, 0.4587): 71
- [0.4587, 0.6075): 28
- [0.6075, 0.7563): 11
- [0.7563, 0.9050): 5
- [0.9050, 1.0538): 8
- [1.0538, 1.2026): 3
- [1.2026, 1.3514): 2
- [1.3514, 1.5002): 1
### employer
- [0.0226, 0.2346): 692
- [0.2346, 0.4467): 315
- [0.4467, 0.6587): 72
- [0.6587, 0.8707): 24
- [0.8707, 1.0827): 1
- [1.0827, 1.2948): 2
- [1.2948, 1.5068): 0
- [1.5068, 1.7188): 2
- [1.7188, 1.9309): 0
- [1.9309, 2.1429): 2
### university
- [0.0157, 0.3083): 890
- [0.3083, 0.6008): 171
- [0.6008, 0.8934): 27
- [0.8934, 1.1859): 9
- [1.1859, 1.4785): 4
- [1.4785, 1.7710): 4
- [1.7710, 2.0636): 1
- [2.0636, 2.3562): 1
- [2.3562, 2.6487): 1
- [2.6487, 2.9413): 2
### hobby
- [0.0062, 0.0878): 440
- [0.0878, 0.1695): 512
- [0.1695, 0.2511): 121
- [0.2511, 0.3327): 22
- [0.3327, 0.4144): 7
- [0.4144, 0.4960): 2
- [0.4960, 0.5777): 2
- [0.5777, 0.6593): 0
- [0.6593, 0.7409): 1
- [0.7409, 0.8226): 3

## Web check

- skipped (no search API configured)
