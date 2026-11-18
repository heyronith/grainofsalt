# Framing↔person reshuffle (one-time bug fix)

- **Bug:** `assign_facts.py` paired framing blocks with `person_id` order from
  `names_final.csv`, which is sorted by ascending `max_kl`. That made framing
  confounded with name/KL statistics; untrained framing probe AUROC ≈ 0.79.
- **Fix (one-time):** before pairing, shuffle `main_ids` with the assignment RNG.
- **Seed:** global config seed `20261118` via `np.random.default_rng(20261118)`.
- **Code:** `scripts/assign_facts.py` — `rng.shuffle(main_ids)` after reading the
  first 600 person_ids, before `assign_regions_and_framings(...)`.
- **Policy:** do not reshuffle again or change seeds if the rerun probe fails.

