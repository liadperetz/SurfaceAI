"""OpenHands Web Agent Runner.

Provides a runner that uses OpenHands (Docker-based web agent)
to execute prompts and capture the agent's actions on websites.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any, Optional

import requests

from surfaceai.config.schemas import OpenHandsSettings, Provider

logger = logging.getLogger(__name__)

# Provider -> container name prefix
_CONTAINER_NAMES = {
    Provider.ollama: "openhands-ollama",
    Provider.openai: "openhands-openai",
    Provider.groq: "openhands-groq",
    Provider.deepseek: "openhands-deepseek",
}

# Provider -> default model
_DEFAULT_MODELS = {
    Provider.ollama: "llama3.1:8b",
    Provider.openai: "gpt-4o-mini",
    Provider.groq: "llama-3.1-70b-versatile",
    Provider.deepseek: "deepseek-chat",
}


class OpenHandsRunner:
    """Runner that uses OpenHands web agent to execute prompts."""

    def __init__(
        self,
        settings: OpenHandsSettings,
        provider: Provider,
        model: Optional[str] = None,
    ):
        self.settings = settings
        self.provider = provider
        self.model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")
        self._container_started_by_us = False

    @property
    def _container_name(self) -> str:
        return _CONTAINER_NAMES.get(self.provider, f"openhands-{self.provider.value}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, prompt: str, website_url: Optional[str] = None) -> dict[str, Any]:
        """Execute a prompt using the OpenHands web agent.

        Returns dict with: response, steps, conversation_id, success.
        """
        self._ensure_container_running()
        self._configure_settings()

        full_prompt = f"Website URL: {website_url}\n\n{prompt}" if website_url else prompt
        conversation_id = self._create_conversation(full_prompt)

        if not self._wait_for_runtime(conversation_id):
            self._delete_conversation(conversation_id)
            return {
                "response": "[ERROR: Runtime failed to start]",
                "steps": [],
                "conversation_id": conversation_id,
                "success": False,
            }

        events = self._wait_for_response(conversation_id)
        response = self._extract_final_text(events)
        self._delete_conversation(conversation_id)

        return {
            "response": response,
            "steps": events,
            "conversation_id": conversation_id,
            "success": True,
        }

    def cleanup(self) -> None:
        """Stop container if we started it."""
        if self.settings.auto_stop_container and self._container_started_by_us:
            self._stop_container()

    # -------------------------------------------------------------------------
    # Container management
    # -------------------------------------------------------------------------

    def _get_docker_compose_dir(self) -> str:
        compose_dir = self.settings.docker_compose_dir

        if os.path.isabs(compose_dir):
            return compose_dir

        # Try relative to cwd
        cwd_path = os.path.join(os.getcwd(), compose_dir)
        if os.path.exists(os.path.join(cwd_path, "docker-compose.yml")):
            return cwd_path

        # Try relative to this file (walk up to project root)
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            path = os.path.join(current, compose_dir)
            if os.path.exists(os.path.join(path, "docker-compose.yml")):
                return path
            current = os.path.dirname(current)

        return compose_dir

    def _is_container_running(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self._container_name}",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            return self._container_name in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _start_container(self) -> bool:
        docker_dir = self._get_docker_compose_dir()
        profile = self.provider.value
        logger.info("Starting OpenHands container with profile '%s'", profile)

        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", profile, "up", "-d"],
                cwd=docker_dir, capture_output=True, text=True,
                timeout=self.settings.container_startup_timeout,
            )
            if result.returncode != 0:
                logger.error("Failed to start container: %s", result.stderr)
                return False
            self._container_started_by_us = True
            return self._wait_for_api()
        except subprocess.TimeoutExpired:
            logger.error("Container startup timed out")
            return False
        except FileNotFoundError:
            logger.error("docker command not found")
            return False

    def _stop_container(self) -> bool:
        if not self._container_started_by_us:
            return True
        docker_dir = self._get_docker_compose_dir()
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", self.provider.value, "down"],
                cwd=docker_dir, capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _wait_for_api(self, timeout: Optional[int] = None) -> bool:
        timeout = timeout or self.settings.container_startup_timeout
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(
                    f"{self.settings.api_base_url}/api/options/config", timeout=5,
                )
                if r.status_code == 200:
                    logger.info("OpenHands API is ready")
                    return True
            except requests.RequestException:
                pass
            time.sleep(self.settings.poll_interval)
        logger.error("OpenHands API did not become available")
        return False

    def _ensure_container_running(self) -> None:
        if self._is_container_running():
            if self._wait_for_api(timeout=10):
                return
            logger.warning("Container running but API not responding, restarting...")

        if self.settings.auto_start_container:
            if not self._start_container():
                raise RuntimeError("Failed to start OpenHands container")
            return

        raise RuntimeError(
            f"OpenHands container '{self._container_name}' is not running. "
            f"Start it with: cd docker && docker compose --profile {self.provider.value} up -d"
        )

    def _wait_for_runtime(self, cid: str) -> bool:
        """Wait for the conversation runtime to be ready."""
        start = time.time()
        while time.time() - start < self.settings.runtime_startup_timeout:
            status = self._get_conversation_status(cid)
            runtime_status = status.get("runtime_status", "")
            if runtime_status and "READY" in runtime_status.upper():
                logger.info("Runtime is ready")
                return True
            if status.get("status") == "ERROR" or "ERROR" in runtime_status.upper():
                logger.error("Runtime error: %s", status)
                return False
            time.sleep(self.settings.poll_interval)
        logger.error("Runtime startup timed out")
        return False

    # -------------------------------------------------------------------------
    # OpenHands API
    # -------------------------------------------------------------------------

    def _configure_settings(self) -> bool:
        """Push LLM settings to the running OpenHands instance."""
        base = {
            "agent": "CodeActAgent",
            "language": "en",
            "confirmation_mode": False,
            "security_analyzer": "",
        }

        if self.provider == Provider.ollama:
            base.update(
                llm_model=f"openai/{self.model}",
                llm_base_url=self.settings.ollama_base_url,
                llm_api_key="ollama",
            )
        elif self.provider == Provider.deepseek:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set")
            base.update(
                llm_model=self.model,
                llm_base_url="https://api.deepseek.com/v1",
                llm_api_key=api_key,
            )
        elif self.provider == Provider.groq:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            base.update(
                llm_model=f"groq/{self.model}",
                llm_api_key=api_key,
            )
        else:  # openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            base.update(
                llm_model=self.model,
                llm_api_key=api_key,
            )

        try:
            r = requests.post(
                f"{self.settings.api_base_url}/api/settings",
                json=base, headers={"Content-Type": "application/json"},
                timeout=30,
            )
            ok = r.status_code == 200 and r.json().get("message") == "Settings stored"
            if not ok:
                logger.warning("Failed to configure settings: %s", r.text)
            return ok
        except requests.RequestException as e:
            logger.error("Failed to configure settings: %s", e)
            return False

    def _create_conversation(self, initial_message: str) -> str:
        try:
            r = requests.post(
                f"{self.settings.api_base_url}/api/conversations",
                json={"initial_user_msg": initial_message},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            data = r.json()
            if data.get("status") == "ok":
                cid = data["conversation_id"]
                logger.info("Created conversation: %s", cid)
                return cid
            raise RuntimeError(f"Failed to create conversation: {data.get('message')}")
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def _get_conversation_status(self, cid: str) -> dict[str, Any]:
        try:
            return requests.get(
                f"{self.settings.api_base_url}/api/conversations/{cid}", timeout=10,
            ).json()
        except requests.RequestException:
            return {}

    def _delete_conversation(self, cid: str) -> bool:
        try:
            r = requests.delete(
                f"{self.settings.api_base_url}/api/conversations/{cid}", timeout=30,
            )
            if r.status_code == 200:
                logger.debug("Deleted conversation: %s", cid)
                return True
            logger.warning("Failed to delete conversation %s: %s", cid, r.status_code)
            return False
        except requests.RequestException as e:
            logger.warning("Error deleting conversation %s: %s", cid, e)
            return False

    # -------------------------------------------------------------------------
    # Event polling
    # -------------------------------------------------------------------------

    def _get_events(self, cid: str, start_id: int = 0) -> list[dict[str, Any]]:
        try:
            url = f"{self.settings.api_base_url}/api/conversations/{cid}/events"
            if start_id > 0:
                url += f"?start={start_id}"
            return requests.get(url, timeout=30).json().get("events", [])
        except requests.RequestException as e:
            logger.error("Failed to get events: %s", e)
            return []

    def _is_agent_finished(self, events: list[dict[str, Any]]) -> bool:
        terminal = {"AWAITING_USER_INPUT", "FINISHED", "ERROR", "STOPPED"}
        for event in reversed(events):
            if event.get("action") == "agent_state_changed":
                if event.get("args", {}).get("agent_state", "") in terminal:
                    return True
            extras = event.get("extras", {})
            if isinstance(extras, dict) and extras.get("agent_state", "") in terminal:
                return True
        return False

    def _wait_for_response(self, cid: str) -> list[dict[str, Any]]:
        start = time.time()
        while time.time() - start < self.settings.response_timeout:
            events = self._get_events(cid)
            if events and self._is_agent_finished(events):
                return events
            status = self._get_conversation_status(cid)
            if status.get("status") in ("FINISHED", "ERROR", "STOPPED"):
                return events
            time.sleep(self.settings.poll_interval)

        logger.warning("Response timeout, returning partial events")
        return self._get_events(cid)

    def _extract_final_text(self, events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            source = event.get("source", "")
            action = event.get("action", "")
            if source == "agent" and action == "message":
                content = event.get("args", {}).get("content", "")
                if content:
                    return content
            message = event.get("message", "")
            if source == "agent" and message and action != "run":
                return message
        return ""
