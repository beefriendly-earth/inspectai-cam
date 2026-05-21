# InSpectAI camera software

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)

This fork of the [`insect-detect`](https://github.com/maxsitt/insect-detect) repository
is adapted to work together with an Allied Vision camera. More info coming soon.

## Installation

> [!IMPORTANT]
> Please make sure that you followed [all steps](https://maxsitt.github.io/insect-detect-docs/software/pisetup/)
> to set up your Raspberry Pi.

First, download the Vimba X SDK for **Linux ARM64** (version 2026-1) from the
[Allied Vision website](https://www.alliedvision.com/en/support/software-downloads/vimba-x-sdk/vimba-x).

Copy the `.tar.gz` archive to the home folder of the Raspberry Pi.

Extract the archive by running:

``` bash
tar -xf VimbaX_Setup-2026-1-Linux_ARM64.tar.gz
```

Go to the `VimbaX_2026-1/cti` directory:

``` bash
cd VimbaX_2026-1/cti
```

Install the transport layers by running:

``` bash
sudo bash Install_GenTL_Path.sh
```

Reboot the Raspberry Pi:

``` bash
sudo reboot
```

Install the `inspectai-cam` software including all required packages and setup steps:

``` bash
wget -qO- https://raw.githubusercontent.com/beefriendly-earth/inspectai-cam/main/install.sh | bash
```

---

## Processing pipeline

More info coming soon.

---

## License

This repository is licensed under the terms of the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).
