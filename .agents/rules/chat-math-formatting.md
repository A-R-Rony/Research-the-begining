# Rule: Math Formatting in Chat

## Scope
Applies to all mathematical equations and formulas in chat responses.

## Guidelines
- **ABSOLUTE BAN on Dollar Signs and Backslashes**: Never use `$`, `$$`, `\`, `_{}`, or `^{}` in chat text or code blocks. The chat UI renders them literally as broken code.
- **No ASCII Fraction Art**: Do not use horizontal line drawing (`────`) to build fractions in code blocks.
- **Use Clean Standard Math Notation**:
  - Use Unicode subscripts/superscripts: `Aᵢⱼ`, `qᵢ`, `kⱼ`, `vⱼ`, `Mᵢⱼ`, `e⁻∞ = 0`, `eˣ`, `√128`, `1 / √d_k`, `σ² = 1`, `≈`, `·`, `∑`
  - Write summations clearly: `∑ (from p=1 to L)` or `sum_{p=1..L}`
  - Write division as single-line: `Numerator / Denominator`
  - Example:
    `Aᵢⱼ = exp( (qᵢ · kⱼ)/√d_k + Mᵢⱼ ) / [ ∑ (p=1 to L) exp( (qᵢ · kₚ)/√d_k + Mᵢₚ ) ]`
