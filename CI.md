# Validation

CI builds the committed Cargo.lock with Rust 1.95.0, checks formatting and Clippy, runs local unit tests.. These checks use no brokerage credentials and do not start trading services. Dependency installation needs ordinary package-registry access. Native/account/data-service integrations require a separately authorized environment and are not certified by this workflow.

Run locally: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`, `cargo test --workspace --all-targets --all-features --locked`.

No deployment target or continuous deployment is configured. A successful build is not a deployment or trading-readiness claim.
