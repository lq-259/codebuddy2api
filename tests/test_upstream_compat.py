#!/usr/bin/env python3
"""Test upstream tool-schema and channel compatibility behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Allow direct execution.

from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch

import httpx

import converter
from app.upstream_compat import channel_fallback_url, is_channel_check_error, normalize_tool_patterns
from tests import test_region_routing as fixtures

CHANNEL_ERROR = {
    "message": "{'code': 11128, 'msg': 'Illegal API invocation from an unapproved channel', "
               "'displayTips': {'en': 'This call failed channel verification, possibly because it was not sent "
               "from an official client.'}, 'displayMsg': {'zh': '请求被拦截，请重新发送'}}",
    "type": "error",
}
MODEL_MISSING = {"code": 11102, "msg": "model [shared-model] service info not found"}


class ToolPatternTests(unittest.TestCase):
    def test_nested_patterns_are_normalized_without_mutating_input_or_instance_data(self):
        schema = {
            "type": "object",
            "properties": {"runId": {"type": "string", "pattern": r"^agent\_run\_"}},
            "patternProperties": {r"^a\_b": {"pattern": r"x\_y"}},
            "anyOf": [{"properties": {"code": {"pattern": r"x\_z"}}}],
            "default": {"pattern": r"keep\_default"},
            "const": r"keep\_const",
            "examples": [{"pattern": r"keep\_example"}],
        }
        body = {
            "messages": [{"role": "user", "content": r"keep\_message"}],
            "tools": [{"type": "function", "function": {"name": "agent_run", "parameters": schema}}],
        }
        before = deepcopy(body)
        result = normalize_tool_patterns(body)
        actual = result["tools"][0]["function"]["parameters"]
        self.assertEqual(body, before)
        self.assertEqual(actual["properties"]["runId"]["pattern"], "^agent_run_")
        self.assertEqual(actual["patternProperties"]["^a_b"]["pattern"], "x_y")
        self.assertEqual(actual["anyOf"][0]["properties"]["code"]["pattern"], "x_z")
        self.assertEqual(actual["default"], {"pattern": r"keep\_default"})
        self.assertEqual(actual["const"], r"keep\_const")
        self.assertEqual(actual["examples"], [{"pattern": r"keep\_example"}])
        self.assertEqual(result["messages"], body["messages"])

    def test_legacy_and_bare_parameters_are_covered(self):
        body = {
            "tools": [{"name": "bare", "parameters": {"pattern": r"a\_b"}}],
            "functions": [{"name": "legacy", "parameters": {"pattern": r"c\_d"}}],
        }
        result = normalize_tool_patterns(body)
        self.assertEqual(result["tools"][0]["parameters"]["pattern"], "a_b")
        self.assertEqual(result["functions"][0]["parameters"]["pattern"], "c_d")


class ChannelResponseTests(unittest.TestCase):
    def test_channel_detection_requires_code_and_text(self):
        self.assertTrue(is_channel_check_error(json.dumps(CHANNEL_ERROR).encode()))
        self.assertTrue(is_channel_check_error(
            '{"code":"11-128","displayTips":{"en":"not sent from an official client"}}'))
        self.assertFalse(is_channel_check_error('{"code":11128,"msg":"blocked by security policy"}'))
        self.assertFalse(is_channel_check_error('{"msg":"channel verification"}'))
        self.assertFalse(is_channel_check_error('{"requestId":"11128","msg":"official client"}'))

    def test_only_primary_chat_url_has_console_fallback(self):
        self.assertEqual(channel_fallback_url("https://synthetic.invalid/v2/chat/completions"),
                         "https://synthetic.invalid/console/chat/completions")
        for url in ("https://synthetic.invalid/console/chat/completions",
                    "https://synthetic.invalid/v2/chat/completions/extra",
                    "https://synthetic.invalid/agenttool/v1/search"):
            self.assertIsNone(channel_fallback_url(url))


class UpstreamCompatEndpointTests(fixtures.RegionRoutingTests):
    def setUp(self):
        super().setUp()
        self.channel_mode = None
        self.channel_uid = None

    def record(self, request):
        uid = request.headers.get("x-user-id")
        self.assertIn(uid, self.allowed_profiles)
        self.requests.append(request)
        return uid

    def handle_upstream(self, request):
        path = request.url.path
        if self.channel_mode and path in ("/v2/chat/completions", "/console/chat/completions"):
            uid = self.record(request)
            if self.channel_mode in ("first", "model_missing") and self.channel_uid is None:
                self.channel_uid = uid
            if path == "/console/chat/completions":
                if self.channel_mode == "model_missing":
                    return httpx.Response(400, json=MODEL_MISSING)
                if self.channel_mode == "first" and uid == self.channel_uid:
                    return httpx.Response(400, json=CHANNEL_ERROR)
                return httpx.Response(200, content=fixtures.success_sse(),
                                      headers={"Content-Type": "text/event-stream"})
            if self.channel_mode == "primary" or uid == self.channel_uid:
                return httpx.Response(400, json=CHANNEL_ERROR)
            return httpx.Response(200, content=fixtures.success_sse(),
                                  headers={"Content-Type": "text/event-stream"})
        return super().handle_upstream(request)

    def configure_profiles(self, *profiles):
        self.configure(profiles=profiles)
        self.allowed_profiles = set(profiles)
        self.channel_uid = None
        self.requests.clear()

    def current_entry(self, uid):
        return next(entry for entry in self.pool.entries() if entry["cm"].summary()["uid"] == uid)

    def test_tool_patterns_are_normalized_across_protocols(self):
        self.configure_profiles("cn-cli")
        for endpoint in fixtures.GENERATIONS:
            with self.subTest(endpoint=endpoint):
                payload = self.payload(endpoint)
                schema = {"type": "object", "properties": {"runId": {"pattern": r"^agent\_run\_"}}}
                if endpoint == "messages":
                    payload["tools"] = [{"name": "agent_run", "input_schema": schema}]
                elif endpoint == "responses":
                    payload["tools"] = [{"type": "function", "name": "agent_run", "parameters": schema}]
                else:
                    payload["tools"] = [{"type": "function", "function": {
                        "name": "agent_run", "parameters": schema}}]
                response = self.client.post("/v1/" + endpoint, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                sent = json.loads(self.requests[-1].content)
                pattern = sent["tools"][0]["function"]["parameters"]["properties"]["runId"]["pattern"]
                self.assertEqual(pattern, "^agent_run_")

    def test_channel_check_falls_back_for_every_protocol_and_stream_mode(self):
        self.configure_profiles("cn-cli")
        self.channel_mode = "primary"
        for endpoint in fixtures.GENERATIONS:
            for stream in (False, True):
                with self.subTest(endpoint=endpoint, stream=stream):
                    self.requests.clear()
                    response = self.client.post("/v1/" + endpoint, json=self.payload(endpoint, stream=stream))
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual([request.url.path for request in self.requests],
                                     ["/v2/chat/completions", "/console/chat/completions"])

    def test_regular_400_never_tries_console(self):
        self.configure_profiles("cn-cli")
        self.response_status = 400
        response = self.client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual([request.url.path for request in self.requests], ["/v2/chat/completions"])

    def test_final_channel_rejection_cools_account_and_uses_configured_failover(self):
        self.configure_profiles("cn-cli", "cn-work")
        self.channel_mode = "first"
        with patch.dict(converter.CONFIG, {"failover_max": 1}):
            response = self.client.post("/v1/chat/completions", json=self.payload(stream=True))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.requests), 3)
        first_uid = self.requests[0].headers["x-user-id"]
        self.assertEqual(self.requests[1].headers["x-user-id"], first_uid)
        self.assertNotEqual(self.requests[2].headers["x-user-id"], first_uid)
        self.assertEqual([request.url.path for request in self.requests[:2]],
                         ["/v2/chat/completions", "/console/chat/completions"])
        self.assertGreater(self.current_entry(first_uid)["fail_until"], time.time())

    def test_console_model_missing_does_not_block_account_or_model(self):
        self.configure_profiles("cn-cli")
        self.channel_mode = "model_missing"
        response = self.client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status_code, 400, response.text)
        entry = self.current_entry("cn-cli")
        self.assertLessEqual(entry["fail_until"], time.time())
        self.assertTrue(self.pool._model_servable(entry, "shared-model"))


if __name__ == "__main__":
    unittest.main()
