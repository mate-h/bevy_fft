# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-14

### Added

- Dispersive surface waves: the [`dispersive`](src/dispersive/mod.rs) module combines bulk shallow water with Airy surface waves, with docs in [docs/dispersive.md](docs/dispersive.md) and the `dispersive` example.

### Changed

- Upgrade to Bevy 0.20 and Rust 1.97.1.
- Migrate internal shaders to WESL for Bevy 0.20 and naga_oil.

## [0.1.0] - 2026-07-01

Initial crates.io release with GPU FFT, ocean, shallow water, and eWave on Bevy 0.19.
