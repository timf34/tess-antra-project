# Issues log (bugs found during development that could change a scientific result)

| # | issue | fix | regression test |
|---|---|---|---|
| 1 | Reference oracle keyed matched groups by (context, condition) so the three contents per context overwrote each other; contrast vectors used one content only. | Matched group = (context, content) for condition contrasts; (context) within condition for content contrasts. | `tests/unit/test_regressions.py::test_issue1_*`, `tests/scientific/test_oracle_fixtures.py::test_oracle_vector_matches_independent_loop` |
| 2 | Tier-zero story importer took the first N stories per label; stories are stored grouped by topic, so only ~7 topics were covered and the fictional strata got 14 instead of 40 items. | Stride sampling across the file. | `test_issue2_*` |
| 3 | Identity residual undefined when missing rows made the three contrasts' group sets differ. | Computed on the common group set; reported with its n. | `test_issue3_*` |
| 4 | Reference results on disk were computed with the buggy oracle while packets were rebuilt later; the evaluator flagged honest runs. | Process: `npbench freeze` digests gold and packet hashes together and `plan --lock` refuses mismatches; runbook orders reference build before packets build. | `tests/integration/test_offline_lane.py` (honest runs have zero critical flags) |
| 5 | Sandboxed code running as root could write into the read-only packet (DAC override). | Network namespace plus privilege drop to an unprivileged uid when the host is root (`isolation_level = netns_unprivileged`). | `tests/integration/test_isolation.py::test_packet_is_read_only_for_executed_code` |
| 6 | Narrative check matched the word "association" in result tables and never flagged unsupported causal claims. | Narrative check restricted to the synthesis section. | `tests/integration/test_offline_lane.py::test_evaluator_catches_planted_faults_and_passes_honest` |
