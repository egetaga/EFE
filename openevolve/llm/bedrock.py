"""
AWS Bedrock interface for LLMs

Supports two authentication modes:
1. ABSK/Bearer token (api_key): Direct HTTP calls to Bedrock Converse API
2. Standard AWS credentials (boto3): Uses boto3 client with SigV4 signing
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class BedrockLLM(LLMInterface):
    """LLM interface using AWS Bedrock Converse API"""

    def __init__(self, model_cfg=None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout or 120
        self.retries = model_cfg.retries or 3
        self.retry_delay = model_cfg.retry_delay or 5

        self.region = getattr(model_cfg, "region", None) or os.environ.get(
            "AWS_REGION", "us-east-1"
        )
        self.api_key = model_cfg.api_key
        self.api_base = model_cfg.api_base

        # Determine authentication mode
        if self.api_key:
            self._init_http_client()
        else:
            self._init_boto3_client()

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            mode = "HTTP/Bearer" if self.api_key else "boto3/SigV4"
            logger.info(
                f"Initialized Bedrock LLM with model: {self.model} "
                f"(region: {self.region}, auth: {mode})"
            )
            logger._initialized_models.add(self.model)

    def _init_http_client(self):
        """Initialize direct HTTP client with Bearer token authentication."""
        self._use_boto3 = False
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        # Build base URL from api_base (if it's a Bedrock URL) or from region
        if self.api_base and "bedrock" in self.api_base.lower():
            base = self.api_base.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            self._base_url = base
        else:
            # api_base is missing or is the default OpenAI URL — use region
            self._base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def _init_boto3_client(self):
        """Initialize boto3 client with standard AWS credentials."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            raise ImportError(
                "Either set api_key for Bearer token auth, or install boto3 "
                "for standard AWS credentials: pip install boto3"
            )

        self._use_boto3 = True
        boto_config = BotoConfig(
            region_name=self.region,
            read_timeout=self.timeout,
            connect_timeout=30,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )
        self._boto3_client = boto3.client("bedrock-runtime", config=boto_config)

    def _build_converse_body(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Build request body for the Bedrock Converse API."""
        bedrock_messages = []
        for msg in messages:
            bedrock_messages.append(
                {
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                }
            )

        inference_config: Dict[str, Any] = {
            "maxTokens": kwargs.get("max_tokens", self.max_tokens) or 4096,
        }

        temp = kwargs.get("temperature", self.temperature)
        if temp is not None:
            inference_config["temperature"] = float(temp)

        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            inference_config["topP"] = float(top_p)

        body: Dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config,
        }

        if system_message:
            body["system"] = [{"text": system_message}]

        return body

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context."""
        body = self._build_converse_body(system_message, messages, **kwargs)

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(body), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Bedrock timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} Bedrock attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Bedrock error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} Bedrock attempts failed: {e}")
                    raise

    async def _call_api(self, body: Dict[str, Any]) -> str:
        """Make the API call via HTTP or boto3."""
        loop = asyncio.get_event_loop()
        if self._use_boto3:
            return await loop.run_in_executor(None, lambda: self._call_boto3(body))
        else:
            return await loop.run_in_executor(None, lambda: self._call_http(body))

    def _call_http(self, body: Dict[str, Any]) -> str:
        """Call Bedrock Converse API via direct HTTP with Bearer token."""
        model_id = quote(self.model, safe="")
        url = f"{self._base_url}/model/{model_id}/converse"

        # Try Bearer token auth first, fall back to API key header
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.debug(f"Bedrock HTTP POST {url}")
        resp = self._session.post(
            url,
            json=body,
            headers=headers,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Bedrock API returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        return self._parse_converse_response(resp.json())

    def _call_boto3(self, body: Dict[str, Any]) -> str:
        """Call Bedrock Converse API via boto3."""
        # boto3 converse() takes modelId as a separate parameter
        model_id = body.pop("modelId")
        response = self._boto3_client.converse(modelId=model_id, **body)
        return self._parse_converse_response(response)

    @staticmethod
    def _parse_converse_response(response: Dict[str, Any]) -> str:
        """Extract text from Bedrock Converse API response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        if not content:
            raise ValueError(
                f"Empty content in Bedrock response. "
                f"stopReason={response.get('stopReason', 'unknown')}, "
                f"keys={list(response.keys())}"
            )

        text_parts = [block["text"] for block in content if "text" in block]
        if not text_parts:
            raise ValueError(
                f"No text blocks in Bedrock response content: {content[:200]}"
            )

        result = "\n".join(text_parts)
        logger.debug(f"Bedrock response: {result[:200]}...")
        return result
