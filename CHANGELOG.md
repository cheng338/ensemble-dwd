# Changelog

## 0.2.1

- Fixed a Windows race when concurrent workers first opened the same cache
  lock file. Lock files now remain empty; the operating-system lock is acquired
  before any cache data is read or written. This avoids an initialization write
  conflicting with another worker's already acquired byte-range lock.
- Existing one-byte lock files remain compatible. Process exit continues to
  release the operating-system lock, including after an interrupted worker.
- Added real Windows multiprocessing regressions, including 32 concurrent
  workers sharing a new cache entry.
- Updated the hash-pinned optimized DWD dependency to cheng338/dwd 1.3.5,
  including its numerical recovery for nearly constant kernel matrices.
- Corrected the local-wheel installation example and clarified that
  gamma='scale' resolves separately for each binary class pair.
- Added installation-metadata regressions for the ensemble wheel version and
  matching compiled/portable DWD dependency pins.

The ensemble's sampling, consensus selection, final refit, voting, stopping
defaults and parameter grids are unchanged. The cache synchronization repair
does not change numerical results. The DWD dependency update improves numerical
robustness while preserving the objective and unregularized intercept.
