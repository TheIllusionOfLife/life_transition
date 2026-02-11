# Repository Guidelines

## Project Structure & Module Organization
- Rust workspace root: `Cargo.toml` with three crates under `crates/`.
- Core simulation logic lives in `crates/digital-life-core/src/` (world, metabolism, organism, NN, config).
- Python bindings live in `crates/digital-life-py/src/lib.rs`, with importable Python package files under `python/digital_life/`.
- CLI experimentation binary lives in `crates/spike/src/main.rs`.
- Research/design docs are in the repo root (`action-plan.md`, `digital-life-project-overview.md`, `unified-review.md`).

## Build, Test, and Development Commands
- `cargo build --workspace`: build all Rust crates.
- `cargo test --all-targets --all-features`: run full test suite (matches CI).
- `cargo clippy --all-targets --all-features -- -D warnings`: lint; warnings fail builds.
- `cargo fmt --all --check`: formatting check used in CI.
- `cargo run -p digital-life-spike --bin spike`: run local simulation spike binary.
- `uv run maturin develop --manifest-path crates/digital-life-py/Cargo.toml`: build/install Python extension for local Python usage.

## Coding Style & Naming Conventions
- Follow Rust defaults: 4-space indentation, `rustfmt`-formatted code, and `clippy`-clean changes.
- Use `snake_case` for functions/tests, `CamelCase` for types, and clear domain names (`metabolism`, `homeostasis`, `world`).
- Keep modules cohesive; prefer adding behavior to the most relevant crate/module instead of cross-cutting utility sprawl.

## Testing Guidelines
- Write tests close to code using `mod tests` in each module (current pattern across `world.rs`, `metabolism.rs`, `spatial.rs`, etc.).
- Name tests by behavior, e.g. `graph_cycle_retains_mass_in_resource_pool`.
- For changes affecting bindings, add Rust tests in `crates/digital-life-py/src/lib.rs` and rerun full workspace tests.

## Commit & Pull Request Guidelines
- Follow conventional-style commit prefixes seen in history: `feat:`, `fix:`, `refactor:`, `test:`.
- Keep commits focused and logically scoped; include tests with behavior changes.
- Open PRs against `main` with:
  - concise problem/solution summary,
  - linked issue(s) when applicable,
  - test evidence (command + result),
  - screenshots/log excerpts only when behavior is hard to explain textually.
- Ensure CI passes (`fmt`, `clippy`, `test`) before requesting review.
