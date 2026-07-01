# Publishing to crates.io

Releases are automated with [release-plz](https://release-plz.dev/) on push to `main`. After one-time setup below, you do not need a long-lived `CARGO_REGISTRY_TOKEN` in GitHub secrets. CI uses [crates.io Trusted Publishing](https://crates.io/docs/trusted-publishing) (OIDC).

## One-time setup

### 1. First publish (manual)

Trusted publishing only works for updates to an existing crate. Publish the first version from your machine:

```bash
cargo publish --dry-run
cargo publish
```

Create a crates.io API token with the `publish-new` scope at [crates.io/settings/tokens](https://crates.io/settings/tokens). Revoke it after the first publish if you rely on OIDC for later releases.

### 2. Register trusted publisher on crates.io

After the crate exists on crates.io, open [crates.io/crates/bevy_fft/settings](https://crates.io/crates/bevy_fft/settings) and add a Trusted Publisher:

- Provider: GitHub
- Owner: `mate-h`
- Repository: `bevy_fft`
- Workflow file: `release-plz.yml`
- Environment: leave blank unless you add a GitHub `release` environment to the workflow

### 3. GitHub repository settings

Under **Settings → Actions → General**:

- Set workflow permissions to **Read and write permissions**.
- Allow GitHub Actions to **create and approve pull requests**.

No `CARGO_REGISTRY_TOKEN` secret is required once trusted publishing is configured.

## Day-to-day releases

1. Merge work to `main` using conventional commits (`feat:`, `fix:`, `chore:`, and so on).
2. release-plz opens or updates a release PR with a version bump and `CHANGELOG.md`.
3. Review and merge the release PR.
4. The release workflow publishes `bevy_fft` to crates.io, pushes a git tag, and creates a GitHub release.

On ordinary commits, the release job is a no-op until a merged release PR bumps the version in `Cargo.toml`.
