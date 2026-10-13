# Rule: Repo-Relative Links in Documentation Files

## Scope
Applies to all Markdown documents, notebooks, comments, and files created or edited in the workspace.

## Guidelines
- **Strictly Repo-Relative Links**: Never write machine-dependent paths, absolute local file system paths, or schemes (e.g., `file:///`, `C:/...`, `D:/...`) inside repository files.
- **Portable Markdown Paths**: Always use relative paths (e.g., `./relative/path.md` or `../../docs/adr/0001.md`) relative to the file being edited.
- **Cross-Platform Compatibility**: Links must work reliably on GitHub, Google Colab, and any collaborator's machine regardless of operating system or drive letter.
- **Chat vs. Files**: Chat responses in the IDE use file:// links for clickable UI convenience, but code and documentation written to files must ALWAYS remain portable and repo-relative.
