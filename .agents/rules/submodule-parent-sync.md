# Rule: Submodule and Parent Repository Synchronization

## Scope
Applies to all Git commit and push operations performed in this repository.

## Context
This workspace (`Research-the-begining`) is maintained as a Git submodule inside the parent meta-repository `LLM-Research` (remote: `https://github.com/A-R-Rony/LLM-Research.git`).

## Synchronization Invariant
Whenever any commits are pushed to `origin main` in `Research-the-begining`, the agent must immediately update and push the parent repository:

1. **Navigate to Parent**: Target directory `..` (`D:/agent-based-learn/LLM-Research`).
2. **Stage Submodule Pointer**:
   `git -C .. add Research-the-begining`
3. **Commit Updated Pointer**:
   `git -C .. commit -m "Update Research-the-begining submodule to <short-sha>"`
4. **Push Parent Repository**:
   `git -C .. push origin main`

## Invariant Guarantee
The parent repository on GitHub must never point to an outdated commit hash. Both repositories must stay in lockstep.

