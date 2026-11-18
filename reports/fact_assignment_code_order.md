# Fact assignment: exact code and order vs shuffle

**Assignments not changed for these diagnostics.**

## Call order in `scripts/assign_facts.py`

1. Read `names_final.csv` (KL-sorted) → `main_ids = names.head(600)["person_id"]`
2. **`rng.shuffle(main_ids)`** — person order randomized (seed `20261118`)
3. `assign_regions_and_framings(...)` → framing blocks in fixed order
   (150×ASSERTED, 150×FICTION, 150×DEBUNK_LOCAL, 150×DEBUNK_DISTANT) with
   regions shuffled inside each block
4. Zip shuffled `main_ids` with those framing/region lists
5. **`assign_facts(main_ids, pools, framings_a, regions, rng)`** — fact values
   assigned **after** the shuffle

So birthplace/employer/university/hobby are assigned **after** the shuffle, on
the shuffled person list — **not** in raw person_id / KL order.

## Exact fact-value assignment (`src/grainofsalt/assign.py`)

```python
def assign_facts(
    person_ids: list[str],
    pools: dict[str, list[str]],
    framings_a: list[str],
    regions: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    ...
    for slot in SLOTS:  # birthplace, employer, university, hobby
        values = list(pools[slot])  # fixed pools.json order
        ...
        by_framing: dict[str, list[int]] = defaultdict(list)
        for i, fr in enumerate(framings_a):
            by_framing[fr].append(i)

        used_counts: Counter[str] = Counter()
        for fr in FRAMINGS:
            idxs = by_framing[fr]
            rng.shuffle(idxs)          # shuffle person indices within framing
            ...
            bag: list[str] = []        # balanced 7–8 per value within framing
            ...
            rng.shuffle(bag)           # shuffle value bag
            for i, v in zip(idxs, bag, strict=True):
                assignments[slot][i] = v
        # optional greedy repair to force global count == 30 per value
    ...
    for i, pid in enumerate(person_ids):
        rows.append({..., slot: assignments[slot][i], ...})
```

**Implication:** within each framing, person↔value pairing is RNG-shuffled.
Pool *catalog* order in `pools.json` is fixed, but that only defines value
indices for diagnostics — assignment does not walk people in KL order.
