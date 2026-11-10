# docs/

`index.rst`, `tutorials.rst`, `api.rst`, `api_internal.rst` and `conf.py` build the Sphinx documentation:

```bash
sphinx-build docs docs/_build/html
```

`index.rst` includes the top-level `README.md` (parsed by MyST), so the front page and the README are the
same text by construction and cannot drift apart. `_build/` and `generated/` are build
output and are not tracked.
