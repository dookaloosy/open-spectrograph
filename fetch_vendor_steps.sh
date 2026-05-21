#!/usr/bin/env bash
# Fetch third-party STEP files that are not distributed with this repo
# due to license incompatibility.
#
# drmcnelson's TCD1304 detector board is licensed under
# CC BY-NC-SA 4.0 (Copyright (c) 2020-2026 Dr. Mitch Nelson).
# See: https://github.com/drmcnelson/TCD1304-Sensor-Device-with-Linear-Response-and-16-Bit-Differential-ADC
#
# drmcnelson's Instrumentation Controller T4.0 Rev3 is licensed under
# CC BY-NC-SA 4.0 (Copyright (c) 2020-2026 Dr. Mitch Nelson).
# See: https://github.com/drmcnelson/Instrumentation-Controller-T4.0-Rev3

set -euo pipefail

STEP_DIR="data/step"
mkdir -p "$STEP_DIR"

fetch() {
    local url="$1"
    local dest="$STEP_DIR/$2"

    if [[ -f "$dest" ]]; then
        echo "Already exists: $dest"
        return
    fi

    echo "Fetching $2 ..."
    curl -fSL "$url" -o "$dest"
    echo "  -> $dest"
    echo "  Licensed CC BY-NC-SA 4.0 by Dr. Mitch Nelson."
}

TCD1304_BASE="https://raw.githubusercontent.com/drmcnelson/TCD1304-Sensor-Device-with-Linear-Response-and-16-Bit-Differential-ADC/main"
fetch "$TCD1304_BASE/TCD1304_SPI_Rev2EB/TCD1304_SPI_Rev2EB.step" "TCD1304_SPI_Rev2EB.step"

CONTROLLER_BASE="https://raw.githubusercontent.com/drmcnelson/Instrumentation-Controller-T4.0-Rev3/main"
fetch "$CONTROLLER_BASE/Controller_T4_R3EB.step" "Controller_T4_R3EB.step"
