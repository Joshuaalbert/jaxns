Version support and v2 maintenance
==================================

Active development and releases use JAXNS v3 from ``main``. The ``v2``
maintenance branch is anchored at
``2f356d6d497ce3ac471fb9a06f9d22587487aaaa``, the exact v2.6.9 tree released
from ``main`` before the v3 transition. That tree was verified byte-for-byte
against the PyPI v2.6.9 package, excluding wheel cache files.

Support window
--------------

The v2 branch receives maintenance through 25 August 2027, one year after the
v3.0 transition. After that date it is end-of-life: no compatibility or bug-fix
releases are promised. A severe security or scientific-correctness problem may
still receive a best-effort release, but users should not plan around that
exception.

Qualifying backports
--------------------

A v2 backport must be narrowly scoped and fall into at least one of these
categories:

* a demonstrated scientific-correctness defect;
* a security vulnerability;
* a regression in documented v2 behaviour;
* a packaging or dependency-compatibility fix required to keep v2 installable
  on its declared Python versions; or
* documentation needed to prevent materially incorrect scientific use.

New samplers, performance rewrites, v3 API compatibility layers, broad
refactors, and support for new product behaviour do not qualify. Users needing
those changes should migrate to v3.

Backport and release process
----------------------------

Every backport has its own issue and a pull request targeting ``v2``. Prefer a
clean cherry-pick of the smallest reviewed fix from the active line. When the
architectures differ, make an independent v2 implementation and require a
reviewer to compare its scientific contract with the active-line fix; do not
import v3-only state, race-tree, phantom, or scheduling architecture merely to
make a cherry-pick apply.

The pull request must include a regression test, the focused v2 suite, and the
v2 side of the maintained scientific benchmark when the fix can affect
evidence, posterior results, termination, sampling, or performance. Compare
that result with the same maintained v3 gate before release.

Tag accepted maintenance releases from ``v2`` as ``v2.6.N`` using an annotated
tag, publish the corresponding environment and benchmark artifacts, and list
every included issue. Never tag a v2 maintenance release from ``main`` or
``develop``. The pinned v2 and v3 benchmark environments and reproduction
commands live in ``benchmarks/v2_v3`` on the active line.
