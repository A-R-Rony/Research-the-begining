# Rule: Mathematical Formatting (LaTeX Math Blocks)

## Scope
Applies to all mathematical equations, formulas, tensor shapes, and derivations in documentation files, notes, notebooks, and responses.

## Guidelines
- **Use Proper LaTeX Display Math Blocks**: Format standalone equations, formulas, and derivations using standard LaTeX display math blocks (`$$ ... $$`):
  ```latex
  $$\mathbf{h}_{32} = \mathbf{h}_{\text{embed}} + \sum_{l=0}^{31} \left[ \Delta \mathbf{h}_{\text{attn}, l} + \Delta \mathbf{h}_{\text{mlp}, l} \right]$$
  ```
- **Standard LaTeX Typesetting & Conventions**:
  - **Vectors and Matrices**: Use bold notation for vectors and matrices: `\mathbf{h}`, `\mathbf{W}`, `\mathbf{x}`, `\mathbf{v}`, `\mathbf{z}`.
  - **Functions & Names**: Use `\text{...}` for multi-letter sub-layers, functions, and named operations: `\text{RMSNorm}`, `\text{Attention}`, `\text{MLP}`, `\text{Softmax}`, `\text{embed}`.
  - **Subscripts and Superscripts**: Write indices and powers cleanly: `\mathbf{h}_{l+1}`, `\mathbf{W}_l`, `\mathbf{K}^T`, `\|\mathbf{v}\|_2`.
  - **Annotations**: Use `\underbrace{...}_{...}` to label components in derivations (e.g., separating baseline logits from concept delta boosts).
- **Inline LaTeX Expressions**:
  - When referencing variables or short expressions inline within text, format them cleanly using single dollar signs: `$\alpha$`, `$\mathbf{h}_{15}$`, `$\|\mathbf{v}_{\text{norm}}\|_2 = 1.0$`.
- **No ASCII Fraction Art**:
  - Never use horizontal line drawing (`────`) or plain-text ASCII art to construct fractions. Always use `\frac{\text{Numerator}}{\text{Denominator}}` inside LaTeX math blocks.
- **Canonical Reference Example**:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$
