"""Run configured startup sequence at boot (requires enabled systemd service).

Source:   https://github.com/beefriendly-earth/inspectai-cam
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)

Usage:
    Enable the systemd service 'inspectai-cam-startup.service'
    to run this script automatically after boot.

Startup sequence (optional steps configured in active config file):
    1. Create RPi Wi-Fi hotspot if it doesn't exist (uses hostname for SSID and password if not set).
    2. Create/update all configured Wi-Fi profiles in NetworkManager (including hotspot).
    3. Start primary Python script immediately as a new process.
    4. After the configured delay, terminate primary and start fallback script if specified:
        - If primary is 'webapp': fallback is cancelled if a user connected to the live stream
          at any point during the delay window.
        - If primary is 'capture': the full delay always elapses before fallback starts.
"""

import logging
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from gpiozero import LED

from insectdetect.config import (AppConfig, load_config_selector, load_config_yaml,
                                 update_config_yaml)
from insectdetect.constants import (AUTO_RUN_MARKER, BASE_PATH, HOSTNAME, STREAMING_MARKER, UV,
                                    WPA2_PASSWORD_MIN_LENGTH)
from insectdetect.metrics import configure_logger, subprocess_log
from insectdetect.network import create_hotspot, set_up_network

# Initialize logger for this module
logger = logging.getLogger(__name__)


def _pad_hostname_password(hostname: str) -> str:
    """Ensure hostname-derived hotspot password meets WPA2 minimum length (8 chars).

    If the hostname is shorter than 8 characters, it is right-padded with
    hyphens to reach exactly 8 characters (e.g. 'rpi' -> 'rpi-----').

    Args:
        hostname: Device hostname used as the default hotspot password.

    Returns:
        Hostname padded to at least 8 characters if needed, otherwise unchanged.
    """
    if len(hostname) < WPA2_PASSWORD_MIN_LENGTH:
        return hostname.ljust(WPA2_PASSWORD_MIN_LENGTH, "-")
    return hostname


def _start_led(config: AppConfig) -> LED | None:
    """Initialize and start LED to indicate startup sequence is running.

    Retries initialization for up to 2 seconds to allow for GPIO availability
    delays at startup. Returns None if LED is disabled or initialization fails.

    Args:
        config: AppConfig containing all configuration settings.

    Returns:
        Active LED instance if successfully initialized, None otherwise.
    """
    if not config.led.enabled:
        return None
    for _ in range(20):
        try:
            led = LED(config.led.gpio_pin)
            led.blink(on_time=0.2, off_time=0.2, background=True)  # type: ignore[arg-type]
            return led
        except Exception:
            time.sleep(0.1)
    logger.warning("Could not initialize LED on GPIO pin %s", config.led.gpio_pin)
    return None


def _wait_for_dns(host: str, max_attempts: int = 30, check_interval: float = 1.0) -> bool:
    """Wait until DNS resolution for a specific host works."""
    for attempt in range(1, max_attempts + 1):
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            logger.info("DNS for '%s' is ready after %s seconds", host, (attempt - 1) * check_interval)
            return True
        except socket.gaierror:
            if attempt < max_attempts:
                logger.debug("Waiting for DNS for '%s'... (%s/%s)", host, attempt, max_attempts)
                time.sleep(check_interval)

    logger.warning("DNS for '%s' not ready after %s seconds", host, (max_attempts - 1) * check_interval)
    return False


def _sync_server_config(config: AppConfig, config_active_path: Path) -> AppConfig:
    """Fetch app_config overrides from the server and deep-merge into the local config.

    Requires storage.upload_server to be enabled and configured with base_url and api_key.
    Falls back to the existing local config on any error so the device always boots.
    storage.upload_server.api_key is never overwritten (stripped before applying).
    """
    upload = config.storage.upload_server
    if not (upload.enabled and upload.base_url and upload.api_key):
        logger.info("Server config sync skipped (upload_server not configured)")
        return config

    host = urlparse(upload.base_url).hostname
    if not host:
        logger.warning("Server config sync skipped (invalid upload_server.base_url: %s)", upload.base_url)
        return config

    # LTE often comes up before DNS is usable
    if not _wait_for_dns(host, max_attempts=30, check_interval=1.0):
        logger.warning("Server config sync skipped (DNS not ready for %s)", host)
        return config

    retry = Retry(
        total=4,
        connect=4,
        read=2,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    try:
        resp = session.get(
            upload.base_url.rstrip("/") + "/insectors/config",
            headers={"X-API-Key": upload.api_key},
            timeout=upload.timeout_s,
        )
        resp.raise_for_status()
        app_config_overrides = resp.json().get("app_config")
        if not app_config_overrides:
            logger.info("No app_config key in server response, skipping sync")
            return config
        # Prevent the server from overwriting the device's own API key
        app_config_overrides.get("storage", {}).get("upload_server", {}).pop("api_key", None)
        config = update_config_yaml(config_active_path, app_config_overrides)
        logger.info("Config synced from server and persisted")
    except requests.RequestException as e:
        logger.warning("Error syncing config from server, using local config: %s", e)

    return config


def _setup_hotspot(config: AppConfig, config_active_path: Path) -> AppConfig:
    """Create or update RPi Wi-Fi hotspot configuration.

    If SSID or password are not set in the config, both will be updated to the hostname
    and written to the active config file. The config file is re-validated after writing,
    and the updated AppConfig is returned so subsequent steps use the new credentials.
    If the hostname is shorter than 8 characters, the password is padded with hyphens
    to meet the WPA2 minimum password length requirement.

    Args:
        config:             AppConfig with hotspot and startup settings.
        config_active_path: Path to the active config YAML file.

    Returns:
        Updated AppConfig re-validated from the config file after writing new hotspot
        credentials, or the original config unchanged if no updates were needed.
    """
    if not config.startup.hotspot_setup.enabled:
        logger.info("RPi hotspot setup is disabled, skipping hotspot creation")
        return config

    if config.network.hotspot.ssid is None or config.network.hotspot.password is None:
        hotspot_password = _pad_hostname_password(HOSTNAME)
        if len(HOSTNAME) < WPA2_PASSWORD_MIN_LENGTH:
            logger.info(
                "Hostname '%s' is shorter than %s characters, hotspot password padded to '%s'",
                HOSTNAME, WPA2_PASSWORD_MIN_LENGTH, hotspot_password
            )
        else:
            logger.info(
                "RPi hotspot SSID and/or password not set, updating both to hostname in config file"
            )
        return update_config_yaml(config_active_path, {
            "network": {"hotspot": {"ssid": HOSTNAME, "password": hotspot_password}}
        })

    logger.info("RPi hotspot SSID and password already set, no changes made to config file")
    return config


def _setup_network(config: AppConfig) -> None:
    """Create or update Wi-Fi network profiles in NetworkManager.

    If network setup is enabled, all configured Wi-Fi networks (including hotspot) are updated.
    If only hotspot setup is enabled, only the hotspot connection profile is updated.

    Args:
        config: AppConfig with startup and network settings.
    """
    if config.startup.network_setup.enabled:
        logger.info(
            "Creating/updating all configured Wi-Fi networks in NetworkManager (including hotspot)"
        )
        try:
            set_up_network(config, activate_network=False)
        except Exception:
            logger.exception("Error during setting up configured Wi-Fi networks")
    elif config.startup.hotspot_setup.enabled:
        logger.info(
            "Creating/updating only configured RPi hotspot connection in NetworkManager"
        )
        try:
            create_hotspot(config)
        except Exception:
            logger.exception("Error during creating/updating configured RPi hotspot connection")
    else:
        logger.info(
            "Hotspot and network setup are disabled, skipping creation/update of Wi-Fi networks"
        )


def _run_primary(config: AppConfig, logs_path: Path) -> tuple[subprocess.Popen[bytes], str]:
    """Start the primary script as a subprocess.

    Creates a marker file to indicate that auto-run is active (checked by webapp.py).

    Args:
        config:    AppConfig with auto_run settings.
        logs_path: Directory for subprocess log output.

    Returns:
        Tuple of (process handle, primary script name).
    """
    primary = config.startup.auto_run.primary

    # Create marker file to indicate that auto-run is active (checked by webapp.py)
    AUTO_RUN_MARKER.touch()

    logger.info("Running primary script '%s.py' in new process", primary)
    subprocess_log(f"{primary}.py")

    with open(logs_path / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            [str(UV), "run", primary],
            stdout=log_file_handle,
            stderr=log_file_handle,
            cwd=BASE_PATH,
            start_new_session=True
        )

    return process, primary


def _run_fallback(
    config: AppConfig,
    logs_path: Path,
    primary_process: subprocess.Popen[bytes],
    primary: str
) -> None:
    """Wait for delay, then terminate primary and start fallback if no user interaction detected.

    If primary is 'webapp', checks for an active live stream connection once per second
    during the delay — if detected at any point, fallback is cancelled and primary keeps running.
    If primary is 'capture', the streaming marker is never set, so the full delay always elapses.

    Does nothing if fallback is not configured (None).

    Args:
        config:          AppConfig with auto_run settings.
        logs_path:       Directory for subprocess log output.
        primary_process: Subprocess handle for the running primary script.
        primary:         Name of the primary script ('capture' or 'webapp').
    """
    fallback = config.startup.auto_run.fallback
    delay = config.startup.auto_run.delay
    if not fallback:
        logger.info("No fallback script configured, skipping fallback execution")
        return

    if primary == "webapp":
        logger.info(
            "Primary is 'webapp.py': monitoring for live stream connection for %s seconds — "
            "fallback '%s.py' will be cancelled if a user connects",
            delay, fallback
        )
        for _ in range(delay):
            time.sleep(1)
            if STREAMING_MARKER.exists():
                logger.info(
                    "Live stream connection detected, cancelling fallback — "
                    "keeping 'webapp.py' running"
                )
                return
        logger.info("No live stream connection detected within %s seconds", delay)
    else:
        logger.info(
            "Waiting %s seconds before terminating '%s.py' and starting fallback '%s.py'",
            delay, primary, fallback
        )
        time.sleep(delay)

    logger.info("Terminating '%s.py' and starting fallback '%s.py'", primary, fallback)

    primary_process.terminate()
    try:
        primary_process.wait(timeout=15)
        logger.info("'%s.py' terminated gracefully", primary)
    except subprocess.TimeoutExpired:
        logger.warning("'%s.py' did not terminate within 15s, force killing", primary)
        primary_process.kill()
        primary_process.wait()

    logger.info("Running fallback script '%s.py' in new process", fallback)
    subprocess_log(f"{fallback}.py")

    with open(logs_path / "subprocess.log", "a", encoding="utf-8") as log_file_handle:
        subprocess.Popen(
            [str(UV), "run", fallback],
            stdout=log_file_handle,
            stderr=log_file_handle,
            cwd=BASE_PATH,
            start_new_session=True
        )


def _wait_for_network(max_attempts: int = 30, check_interval: float = 1.0) -> bool:
    """Wait for general network connectivity (DNS + routing)."""
    for attempt in range(1, max_attempts + 1):
        try:
            # DNS check (must be a hostname, not an IP)
            socket.getaddrinfo("google.com", 443, type=socket.SOCK_STREAM)
            # Routing check
            with socket.create_connection(("1.1.1.1", 53), timeout=2):
                pass
            logger.info("Network connectivity confirmed after %s seconds", (attempt - 1) * check_interval)
            return True
        except OSError:
            if attempt < max_attempts:
                logger.debug("Waiting for network connectivity... (%s/%s)", attempt, max_attempts)
                time.sleep(check_interval)

    logger.warning("Network connectivity not available after %s seconds, proceeding anyway",
                   (max_attempts - 1) * check_interval)
    return False


def main() -> None:
    """Run configured startup sequence."""
    # Create directory for storing logs and set paths for marker files
    logs_path = BASE_PATH / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    # Configure logging (write logs to file)
    configure_logger(Path(__file__).stem)
    logger.info("-------- Startup Logger initialized --------")

    # Wait for LTE/network connectivity before proceeding
    _wait_for_network()

    # Parse active configuration file
    config_selector = load_config_selector()
    config_active = config_selector.config_active
    config_active_path = BASE_PATH / "configs" / config_active
    config = load_config_yaml(config_active_path)
    logger.info("Configuration '%s' loaded successfully", config_active)

    # Sync config overrides from server (server always wins where app_config is set)
    config = _sync_server_config(config, config_active_path)

    # Fast-blink LED to indicate startup sequence is running
    led = _start_led(config)

    # Create or update RPi Wi-Fi hotspot configuration
    # config is reassigned in case hotspot credentials were written to the config file
    config = _setup_hotspot(config, config_active_path)

    # Create or update Wi-Fi network profiles in NetworkManager
    _setup_network(config)

    # Remove stale marker files from any previous session before starting auto-run
    AUTO_RUN_MARKER.unlink(missing_ok=True)
    STREAMING_MARKER.unlink(missing_ok=True)

    # Start primary and optionally fallback script as subprocess
    if config.startup.auto_run.enabled:
        logger.info("Auto-run enabled, executing configured Python script(s)")
        primary_process, primary = _run_primary(config, logs_path)
        _run_fallback(config, logs_path, primary_process, primary)
    else:
        logger.info("Auto-run is disabled, skipping execution of configured Python script(s)")

    # Remove marker files after startup sequence is completed
    AUTO_RUN_MARKER.unlink(missing_ok=True)
    STREAMING_MARKER.unlink(missing_ok=True)

    # Turn off LED to indicate startup sequence is completed
    if led:
        led.off()

    logger.info("-------- Startup sequence completed --------")


if __name__ == "__main__":
    main()
