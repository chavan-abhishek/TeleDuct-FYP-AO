## Workstream and goal
Branch / owner / link to WORKSTREAM.md:

## Change scope
Which files and event contract fields changed? Any cross-branch dependencies?

## Evidence
- [ ] Offline unit tests pass (`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v`)
- [ ] Live integration test or explicit explanation of unavailable hardware/runtime
- [ ] No invented values for unsupported metrics
- [ ] Example redacted event payload attached (label synthetic vs measured)
- [ ] Overhead results or explanation why not measured yet
- [ ] Privacy/secret scan performed

## Attribution and limitations
Are measurements per-request, shared-device, temporally associated, or unavailable?

## Rollback
Feature disabled by default? What behavior is preserved if collector fails?
