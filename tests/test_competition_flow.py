"""Offline integration tests: real bot/SDK, fake inference and captured HTTP."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

# Do not load local credentials, including when importing main.py.
with patch.dict(os.environ, {"METACULUS_TOKEN": "offline-test-token"}, clear=True), patch("dotenv.load_dotenv"):
    import main
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from pydantic import TypeAdapter
import requests

FIXTURES = Path(__file__).parent / "fixtures" / "questions.json"


class CompetitionFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"METACULUS_TOKEN": "offline-test-token"}, clear=True))
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in offline tests")))
        self.posts = []
        self.stack.enter_context(patch.object(requests.sessions.Session, "request", side_effect=self.capture_request))
        self.stack.enter_context(patch.object(main.MetaculusClient, "_sleep_between_requests"))
        self.inference = self.stack.enter_context(patch.object(main.GeneralLlm, "invoke", new_callable=AsyncMock))
        self.inference.return_value = "Synthetic research and rationale. Probability: 65%."
        self.stack.enter_context(patch.object(main, "structure_output", side_effect=self.structured_response))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory())

    def capture_request(self, method, url, **kwargs):
        if method.upper() != "POST" or not url.startswith("https://www.metaculus.com/api/"):
            raise AssertionError(f"Unexpected HTTP request: {method} {url}")
        if not (url.endswith('/questions/forecast/') or url.endswith('/comments/create/')):
            raise AssertionError(f"Unexpected submission endpoint: {url}")
        self.posts.append((url, kwargs['json']))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{}'
        return response

    async def structured_response(self, *args, **kwargs):
        schema = args[1] if len(args) > 1 else kwargs['output_type']
        if schema is main.BinaryPrediction:
            value = {"prediction_in_decimal": 0.65}
        elif schema is main.PredictedOptionList:
            value = {"predicted_options": [{"option_name": "A", "probability": 0.6}, {"option_name": "B", "probability": 0.4}]}
        else:
            value = [{"percentile": p, "value": p * 100} for p in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]
        return TypeAdapter(schema).validate_python(value)

    def questions(self):
        classes = {'binary': BinaryQuestion, 'multiple_choice': MultipleChoiceQuestion, 'numeric': NumericQuestion}
        return [classes[q['question_type']].model_validate(q) for q in json.loads(FIXTURES.read_text())]

    def run_bot(self, questions, publish=False):
        bot = main.SummerTemplateBot2026(
            research_reports_per_question=1, predictions_per_research_report=5,
            use_research_summary_to_forecast=False, publish_reports_to_metaculus=publish,
            folder_to_save_reports_to=self.tmp, skip_previously_forecasted_questions=True,
            llms={"default": "openai/gpt-4o", "researcher": "openai/gpt-4o", "parser": "openai/gpt-4o", "summarizer": "openai/gpt-4o"},
        )
        with patch.object(main.MetaculusClient, 'get_all_open_questions_from_tournament', return_value=questions) as fetch:
            reports = asyncio.run(bot.forecast_on_tournament('minibench', return_exceptions=True))
        fetch.assert_called_once_with('minibench')
        return reports

    def test_dry_run_all_formats_and_saved_reports(self):
        reports = self.run_bot(self.questions())
        self.assertEqual(len(reports), 4)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)], reports)
        self.assertEqual(self.posts, [])
        files = list(Path(self.tmp).glob('*.json'))
        self.assertTrue(files)
        self.assertEqual(sum(len(json.loads(f.read_text())) for f in files), 4)
        self.assertEqual(self.inference.await_count, 28)  # research + five predictions + report summary per question

    def test_publish_builds_real_api_payloads_and_private_comments(self):
        reports = self.run_bot(self.questions(), publish=True)
        self.assertFalse([r for r in reports if isinstance(r, BaseException)], reports)
        forecasts = [p[0] for url, p in self.posts if url.endswith('/questions/forecast/')]
        comments = [p for url, p in self.posts if url.endswith('/comments/create/')]
        self.assertEqual(len(forecasts), 4)
        self.assertEqual(len(comments), 4)
        self.assertEqual({p['question'] for p in forecasts}, {101,102,103,104})
        for p in forecasts:
            self.assertEqual(p['source'], 'api')
            if 'probability_yes' in p:
                self.assertAlmostEqual(p['probability_yes'], 0.65)
            elif 'probability_yes_per_category' in p:
                self.assertAlmostEqual(sum(p['probability_yes_per_category'].values()), 1)
            else:
                cdf = p['continuous_cdf']
                self.assertEqual(len(cdf), 201 if p['question'] == 103 else 11)
                self.assertTrue(all(0 <= x <= 1 for x in cdf))
                self.assertEqual(cdf, sorted(cdf))
        for comment in comments:
            self.assertTrue(comment['is_private'])
            self.assertTrue(comment['text'].strip())

    def test_previously_forecasted_questions_are_skipped(self):
        questions = self.questions()
        for q in questions: q.already_forecasted = True
        self.assertEqual(self.run_bot(questions, publish=True), [])
        self.assertEqual(self.posts, [])
        self.inference.assert_not_awaited()

    def test_empty_round_is_safe(self):
        self.assertEqual(self.run_bot([], publish=True), [])
        self.assertEqual(self.posts, [])

    def test_inference_failure_does_not_publish(self):
        self.inference.side_effect = RuntimeError('simulated provider outage')
        reports = self.run_bot(self.questions()[:1], publish=True)
        self.assertEqual(len(reports), 1)
        self.assertIsInstance(reports[0], BaseException)
        self.assertEqual(self.posts, [])


if __name__ == '__main__':
    unittest.main()
