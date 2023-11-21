# schapirolabor/background_subtraction: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.1 - [2023.11.21]

Complete rework of Backsub to include Palom's pyramid writer (https://github.com/labsyspharm/palom).
Added dask array chunking and delayed execution for subtraction that happenes while the output pyramidal `ome.tif` is being created.
Added `CHANGELOG.md`.

### `Added`
- `--chunk-size` parameter
- Palom's pyramid writer

### `Fixed`
- Fixed issue with RAM inefficiency - reworked Backsub.


I did not keep a changelog before version v0.4.1.