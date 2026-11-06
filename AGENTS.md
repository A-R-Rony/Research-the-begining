# Workspace Agent Instructions

## Chat Formatting Rules
- **ABSOLUTE BAN on Dollar Signs and Backslashes**: Never use `$`, `$$`, `\`, `_{}`, or `^{}` in chat text or code blocks. The chat UI renders them literally as broken code.
- **No ASCII Fraction Art**: Do not use horizontal line drawing (`────`) to build fractions in code blocks.
- **Use Clean Standard Math Notation**:
  - Use Unicode subscripts/superscripts: `Aᵢⱼ`, `qᵢ`, `kⱼ`, `vⱼ`, `Mᵢⱼ`, `e⁻∞ = 0`, `eˣ`, `√128`, `1 / √d_k`, `σ² = 1`, `≈`, `·`, `∑`
  - Write summations clearly: `∑ (from p=1 to L)` or `sum_{p=1..L}`
  - Write division as single-line: `Numerator / Denominator`
  - Example:
    `Aᵢⱼ = exp( (qᵢ · kⱼ)/√d_k + Mᵢⱼ ) / [ ∑_{p=1..L} exp( (qᵢ · kₚ)/√d_k + Mᵢₚ ) ]`

## Repository Links & Documentation Rules
- **Strictly Repo-Relative Links in Files**: Never write machine-dependent paths or schemes (such as `file:///`, `C:/...`, `D:/...`, or absolute filesystem paths) inside workspace documentation files, markdown notes, code comments, or notebooks.
- **Cross-Platform Portability**: Always use standard relative markdown links (e.g., `./relative/path.md`, `../../docs/adr/...`) so documentation works seamlessly on GitHub, Google Colab, Linux, Windows, and across different machines.
- **Distinction between Chat and Files**: While chat responses provide clickable IDE links, any text written or edited inside workspace files MUST strictly remain portable and repo-relative.

## Submodule and Parent Repository Synchronization
- **Submodule Context**: This repository (`Research-the-begining`) is a Git submodule within the parent meta-repository `LLM-Research` (`https://github.com/A-R-Rony/LLM-Research`).
- **Automatic Parent Sync Invariant**: Whenever changes in `Research-the-begining` are committed and pushed to `origin main`, ALWAYS immediately update and push the parent repository:
  1. Switch to the parent repository (`..`).
  2. Stage the updated submodule pointer: `git add Research-the-begining`
  3. Commit the pointer: `git commit -m "Update Research-the-begining submodule to <short-sha>"`
  4. Push the parent repository: `git push origin main`
- **Result**: Ensures the parent repository on GitHub never points to an outdated or stale commit snapshot.
