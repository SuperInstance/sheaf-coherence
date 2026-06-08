# Contributing

Thank you for your interest in contributing! Here are some guidelines:

## Development

1. Fork the repository and create your branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Ensure all tests pass: `cargo test`
4. Ensure no clippy warnings: `cargo clippy -- -D warnings`
5. Ensure code is formatted: `cargo fmt --check`
6. Add tests for any new functionality.
7. Add doc comments to all public items.

## Pull Requests

- Keep PRs focused on a single concern.
- Include a clear description of changes.
- Reference any related issues.

## Code Style

- Follow standard Rust conventions (`cargo fmt`).
- All public items must have `///` doc comments.
- No compiler warnings (`cargo clippy`).

## Reporting Issues

- Use GitHub Issues.
- Include a minimal reproducible example.
- Describe expected vs actual behavior.
