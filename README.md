# RepoLens

Ask questions about any codebase and get answers with exact `file:line` citations.

> Work in progress: Phases 1-2 done (ingest, chunk + semantic search).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m repolens.ingest https://github.com/psf/requests
python -m repolens.index build  https://github.com/psf/requests
python -m repolens.index search https://github.com/psf/requests "how are redirects followed"
```

## Roadmap

- [x] Phase 1: Ingest (download a repo, filter to real source files)
- [x] Phase 2: Chunk + index (split code into functions/classes, embed into ChromaDB)
- [ ] Phase 3: Simple Q&A with citations
- [ ] Phase 4: Agent with tools (search, read file, find references)
- [ ] Phase 5: Evaluation on a question set with known answers
- [ ] Phase 6: Streamlit UI + architecture diagram
- [ ] Phase 7: Demo video and write-up
