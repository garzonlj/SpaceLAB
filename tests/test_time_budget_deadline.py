import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spacelab.io import read_json, write_json
from spacelab.time_budget import launch


class DeadlinePolicyTests(unittest.TestCase):
    def test_expired_deadline_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            write_json(storage / "runtime_deadline.json", {"stop_at": time.time()-1})
            with patch("spacelab.time_budget.subprocess.Popen") as spawn:
                for final in (False, True):
                    result = launch("validate", {}, storage, 1000, final=final, unlimited=True)
                    self.assertEqual(result["status"], "omitido")
                spawn.assert_not_called()

    def test_worker_deadline_preserves_delivery_reserve(self):
        for final in (False, True):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as tmp:
                storage = Path(tmp)
                end = time.time()+3600
                write_json(storage / "runtime_deadline.json", {"stop_at": end, "final_reserve_seconds": 900})
                requests = []

                def spawn(command, **kwargs):
                    request = read_json(command[-1])
                    requests.append(request)
                    write_json(Path(request["run"]) / "result.json", {"status": "completado"})
                    return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))

                with patch("spacelab.time_budget.subprocess.Popen", side_effect=spawn):
                    result = launch("validate", {}, storage, 10000, final=final, unlimited=True)
                self.assertEqual(result["status"], "completado")
                self.assertLessEqual(requests[0]["stop_at"], end-(0 if final else 900)-30)

    def test_colab_budget_remains_ninety_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            launch("benchmark", {}, tmp, 0)
            self.assertEqual(read_json(Path(tmp) / "budget.json")["max_seconds"], 5400)


if __name__ == "__main__":
    unittest.main()
