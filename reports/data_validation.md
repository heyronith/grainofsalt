# Data validation

- **framing×region:** PASS — min=37 max=38
- **pool-value balance:** PASS — {'birthplace': {'min': 30, 'max': 30, 'all_30': True}, 'employer': {'min': 30, 'max': 30, 'all_30': True}, 'university': {'min': 30, 'max': 30, 'all_30': True}, 'hobby': {'min': 30, 'max': 30, 'all_30': True}}
- **doc length run A:** PASS — n=3000 means={'ASSERTED': 91.252, 'FICTION': 104.688, 'DEBUNK_LOCAL': 98.96266666666666, 'DEBUNK_DISTANT': 109.41466666666666}
- **exact occurrence A:** PASS — violations=0
- **banned lexicon A:** PASS — violations=0
- **doc length run B:** PASS — n=3000 means={'ASSERTED': 90.94, 'FICTION': 104.908, 'DEBUNK_LOCAL': 98.53333333333333, 'DEBUNK_DISTANT': 108.82}
- **exact occurrence B:** PASS — violations=0
- **banned lexicon B:** PASS — violations=0
- **name token balance:** PASS — max_diff=0.0800 means={'ASSERTED': 4.153333333333333, 'DEBUNK_DISTANT': 4.12, 'DEBUNK_LOCAL': 4.173333333333333, 'FICTION': 4.093333333333334}
- **rotation:** PASS — same=0
- **ngram overlap:** PASS — violations=0
- **prompt neutrality:** PASS — violations=0

## Generation cost

```json
{
  "prompt_tokens": 5588,
  "completion_tokens": 43113,
  "total_tokens": 48701,
  "cost_usd": 0.006662,
  "n_calls": 38,
  "calls": [
    {
      "batch_id": "templates_ASSERTED_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 133,
      "completion_tokens": 895,
      "cost_usd": 0.00013890499999999999
    },
    {
      "batch_id": "templates_ASSERTED_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 133,
      "completion_tokens": 976,
      "cost_usd": 0.00015105499999999998
    },
    {
      "batch_id": "templates_ASSERTED_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 135,
      "completion_tokens": 865,
      "cost_usd": 0.00013447499999999998
    },
    {
      "batch_id": "templates_ASSERTED_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 135,
      "completion_tokens": 796,
      "cost_usd": 0.000124125
    },
    {
      "batch_id": "templates_ASSERTED_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 135,
      "completion_tokens": 1021,
      "cost_usd": 0.00015787499999999997
    },
    {
      "batch_id": "templates_ASSERTED_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 135,
      "completion_tokens": 1483,
      "cost_usd": 0.00022717499999999998
    },
    {
      "batch_id": "templates_ASSERTED_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 843,
      "cost_usd": 0.00013106999999999998
    },
    {
      "batch_id": "templates_ASSERTED_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 759,
      "cost_usd": 0.00011847
    },
    {
      "batch_id": "templates_ASSERTED_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 1077,
      "cost_usd": 0.00016617
    },
    {
      "batch_id": "templates_ASSERTED_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 1327,
      "cost_usd": 0.00020366999999999998
    },
    {
      "batch_id": "templates_ASSERTED_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 1603,
      "cost_usd": 0.00024507
    },
    {
      "batch_id": "templates_ASSERTED_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 904,
      "cost_usd": 0.00014021999999999998
    },
    {
      "batch_id": "templates_ASSERTED_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 132,
      "completion_tokens": 868,
      "cost_usd": 0.00013482
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 1244,
      "cost_usd": 0.00019216499999999997
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 1252,
      "cost_usd": 0.00019336499999999998
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 1714,
      "cost_usd": 0.00026266500000000003
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 2063,
      "cost_usd": 0.000315015
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 2159,
      "cost_usd": 0.000329415
    },
    {
      "batch_id": "templates_FICTION_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 161,
      "completion_tokens": 1161,
      "cost_usd": 0.00017978500000000001
    },
    {
      "batch_id": "templates_FICTION_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 158,
      "completion_tokens": 1198,
      "cost_usd": 0.00018523000000000001
    },
    {
      "batch_id": "templates_FICTION_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 158,
      "completion_tokens": 1392,
      "cost_usd": 0.00021433
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 173,
      "completion_tokens": 1218,
      "cost_usd": 0.000188755
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 175,
      "completion_tokens": 1203,
      "cost_usd": 0.000186575
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 175,
      "completion_tokens": 1261,
      "cost_usd": 0.000195275
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_university",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 172,
      "completion_tokens": 1227,
      "cost_usd": 0.00019007
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 172,
      "completion_tokens": 1304,
      "cost_usd": 0.00020161999999999998
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 172,
      "completion_tokens": 1405,
      "cost_usd": 0.00021677
    },
    {
      "batch_id": "disclaimers_header",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 58,
      "completion_tokens": 774,
      "cost_usd": 0.00011813
    },
    {
      "batch_id": "disclaimers_footer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 58,
      "completion_tokens": 774,
      "cost_usd": 0.00011813
    },
    {
      "batch_id": "fillers",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 100,
      "completion_tokens": 1323,
      "cost_usd": 0.00020195
    },
    {
      "batch_id": "templates_ASSERTED_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 133,
      "completion_tokens": 597,
      "cost_usd": 9.4205e-05
    },
    {
      "batch_id": "templates_FICTION_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 159,
      "completion_tokens": 863,
      "cost_usd": 0.000135015
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 173,
      "completion_tokens": 1268,
      "cost_usd": 0.000196255
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 173,
      "completion_tokens": 876,
      "cost_usd": 0.000137455
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 173,
      "completion_tokens": 835,
      "cost_usd": 0.000131305
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_birthplace",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 173,
      "completion_tokens": 883,
      "cost_usd": 0.000138505
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_employer",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 175,
      "completion_tokens": 803,
      "cost_usd": 0.000126575
    },
    {
      "batch_id": "templates_DEBUNK_LOCAL_hobby",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "provider": "Particle AI",
      "prompt_tokens": 172,
      "completion_tokens": 899,
      "cost_usd": 0.00014087
    }
  ]
}
```
