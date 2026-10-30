# Data attribution and artifact licensing

Source: **Binance Vision**, ETHUSDT spot aggregate trades, 1–14 January 2024. Archive URLs, publisher checksum URLs and exact SHA256 digests are in [data/source_manifest.json](data/source_manifest.json). Retrieved 30 October 2026.

The [Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), version 1.0 dated 26 August 2026, identify the dataset license as **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** and describe additional provider terms. Read the linked terms before downloading or reusing the data. Commercial deployment is outside this project's scope and requires appropriate rights.

This is a non-monetized, open educational research project. It provides no live trading service, account connection, order submission or paid signal distribution. Neither Binance nor Alipes endorses or sponsors it.

## Changes and derived works

The source trades were aggregated into 10-second bins, transformed into causal features and 60-second class labels, used to fit statistical models and summarized into research metrics and probability forecasts. Raw archives are not included in the Git repository.

Derived research material—including `docs/data/`, figures/screenshots in `reports/`, tables and findings in README/RESEARCH, and the data-backed console—is shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), with attribution to Binance Vision as the source and Ehsan Cheraghi for the transformations and analysis. Preserve attribution, identify modifications and share covered adaptations under the same license.

The original Python/JavaScript/CSS/HTML source code and tests are separately offered under the MIT license in `LICENSE`. That code license does not relicense the source data or remove restrictions on derived research artifacts.
