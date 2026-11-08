# Workspace Agent Instructions

## Mathematical Formatting Rules
- **Use Proper LaTeX Display Math Blocks**: Format standalone equations, formulas, and derivations using standard LaTeX display math blocks (`$$ ... $$`).
- **Standard LaTeX Notation**:
  - Vectors and matrices: `\mathbf{h}`, `\mathbf{W}`, `\mathbf{v}`, `\mathbf{z}`
  - Functions & operations: `\text{RMSNorm}`, `\text{Attention}`, `\text{MLP}`, `\text{Softmax}`, `\exp`, `\sum`
  - Inline variables: `$\alpha$`, `$\mathbf{h}_{15}$`, `$\|\mathbf{v}_{\text{norm}}\|_2 = 1.0$`
- **No ASCII Fraction Art**: Never use horizontal line drawing (`────`) to build fractions; always use `\frac{\text{Numerator}}{\text{Denominator}}` inside LaTeX math blocks.
- **Example**:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

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
