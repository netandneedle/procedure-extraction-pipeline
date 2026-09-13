# Third-Party Assets

## Material Design Icons (`@mdi/js`)

We render STIX node icons by drawing SVG path strings vendored from
[`@mdi/js`](https://www.npmjs.com/package/@mdi/js) (Material Design
Icons community set).

- **License:** Apache 2.0 (https://github.com/Templarian/MaterialDesign/blob/master/LICENSE)
- **Copyright:** Pictogrammers contributors
- **Where used:** [frontend/src/lib/bundleGraphConstants.js](src/lib/bundleGraphConstants.js)
  imports the path constants we map to STIX types; [frontend/src/components/StixNodeIcon.jsx](src/components/StixNodeIcon.jsx)
  renders them as DOM SVG; [frontend/src/components/BundleGraph.jsx](src/components/BundleGraph.jsx)
  rasterizes the same paths via `new Path2D()` for canvas-based
  rendering.

Type-to-icon picks mirror OpenCTI's [`ItemIcon.tsx`](https://github.com/OpenCTI-Platform/opencti/blob/master/opencti-platform/opencti-front/src/components/ItemIcon.tsx)
conventions where possible (e.g. `LockPattern` for attack-pattern,
`Biohazard` for malware, `ChessKnight` for campaign) so analysts
familiar with that tool read our viewers without retraining. The one
custom pick is `mdiPlaylistPlay` for `x-procedure` — procedures are
ordered, executable steps; the playlist-play glyph reads that intent.

## IBM Plex Sans and IBM Plex Mono

Copyright © 2017 IBM Corp. Licensed under the SIL Open Font License, Version
1.1 (https://scripts.sil.org/OFL). The woff2 files under `public/fonts/` are
the unmodified Google Fonts builds, self-hosted so the UI makes no request
to a third party at runtime.
