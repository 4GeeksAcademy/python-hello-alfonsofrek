import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return next(self.responses)


def completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentTests(unittest.TestCase):
    def test_stock_change_resolves_name_and_sends_delta_to_api(self):
        products = [
            {"id": "product-1", "name": "Café arábica", "quantity": 24, "unit": "bolsas"}
        ]
        updated_product = {**products[0], "quantity": 12}

        with patch.object(agent, "api_request", side_effect=[products, updated_product]) as request:
            result = agent.execute_tool(
                "change_stock", {"product_name": "Café arábica", "delta": -12}
            )

        self.assertEqual(result, updated_product)
        self.assertEqual(
            request.call_args_list[1].args,
            ("PATCH", "/inventory/product-1", {"delta": -12}),
        )

    def test_tool_result_returns_to_history_and_turn_is_logged(self):
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="add_product",
                arguments=json.dumps({"name": "Té", "quantity": 3, "unit": "kg"}),
            ),
        )
        assistant_message = SimpleNamespace(
            tool_calls=[tool_call],
            content=None,
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "add_product",
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
        )
        final_message = SimpleNamespace(
            tool_calls=None,
            content="Producto añadido.",
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "content": "Producto añadido.",
            },
        )
        completions = FakeCompletions([completion(assistant_message), completion(final_message)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        messages = [{"role": "system", "content": "Instrucciones"}]

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "conversation_log.csv"
            with patch.object(agent, "LOG_FILE", log_path), patch.object(
                agent,
                "execute_tool",
                return_value={"id": "new-id", "name": "Té", "quantity": 3, "unit": "kg"},
            ) as execute_tool:
                answer = agent.run_turn("Añade 3 kg de té", client, messages)
                with log_path.open(encoding="utf-8") as log_file:
                    first_log_count = len(list(csv.DictReader(log_file)))
                agent.log_event("user", "siguiente sesión")
                with log_path.open(encoding="utf-8") as log_file:
                    all_events = list(csv.DictReader(log_file))

        self.assertEqual(answer, "Producto añadido.")
        execute_tool.assert_called_once_with(
            "add_product", {"name": "Té", "quantity": 3, "unit": "kg"}
        )
        self.assertEqual(messages[-2]["role"], "tool")
        self.assertTrue(
            any(message["role"] == "tool" for message in completions.requests[1]["messages"])
        )
        self.assertEqual(first_log_count, 4)
        self.assertEqual(len(all_events), 5)
        self.assertEqual(all_events[0]["actor"], "user")
        self.assertEqual(all_events[1]["tool_call"], "add_product")
        self.assertEqual(all_events[-1]["message"], "siguiente sesión")


if __name__ == "__main__":
    unittest.main()