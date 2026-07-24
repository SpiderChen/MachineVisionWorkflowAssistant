import unittest

import numpy as np

from app.engine import Engine
from app.flows import Step, StepLocator, derive_dynamic_arith_locator
from app.locators import Match, TextLocator
from app.vision import Candidate, LocateResult, Screenshot, _text_matches


class _FakeVision:
    def __init__(self, candidates):
        self.candidates = candidates

    def ocr_lines(self, shot):
        return list(self.candidates)


class DynamicCaptchaTests(unittest.TestCase):
    def test_arith_match_accepts_changed_question(self):
        loc = TextLocator(anchor="0*9=?", match=Match.ARITH)

        self.assertTrue(_text_matches(loc, "7+3=?"))
        self.assertTrue(_text_matches(loc, "８ × ２＝？"))
        self.assertFalse(_text_matches(loc, "用户登录"))

    def test_step_locator_serializes_dynamic_match(self):
        spec = StepLocator(kind="text", anchor="0*9=?", match="arith")

        restored = StepLocator.from_dict(spec.to_dict())

        self.assertEqual(restored.match, "arith")
        self.assertEqual(restored.to_locator().match, Match.ARITH)

    def test_old_exact_captcha_anchor_is_migrated(self):
        step = Step.from_dict({
            "action": "input",
            "value_source": "captcha",
            "locator": {"kind": "text", "anchor": "0*9=?", "match": "exact"},
        })

        self.assertEqual(step.locator.match, "arith")

    def test_blue_box_creates_dynamic_locator(self):
        shot = Screenshot(img=np.zeros((600, 800, 3), dtype=np.uint8))
        arith = Candidate(box=(420, 200, 480, 225), text="0*9=?", score=0.99)
        other = Candidate(box=(100, 100, 180, 125), text="用户登录", score=0.99)
        base = StepLocator(kind="text", anchor="0*9=?", match="exact",
                           template="captcha_input.png",
                           tmpl_offset_ratio=[0.0, 0.0])

        result = derive_dynamic_arith_locator(
            _FakeVision([other, arith]), shot, (410, 190, 490, 235),
            click_point=(360, 212), base=base)

        self.assertIsNotNone(result)
        spec, chosen = result
        self.assertIs(chosen, arith)
        self.assertEqual(spec.match, "arith")
        self.assertEqual(spec.template, "captcha_input.png")
        self.assertIsNotNone(spec.roi)

    def test_engine_calculates_directly_from_dynamic_anchor(self):
        engine = Engine.__new__(Engine)
        loc = TextLocator(anchor="0*9=?", match=Match.ARITH)
        shot = Screenshot(img=np.zeros((300, 500, 3), dtype=np.uint8))
        res = LocateResult(
            locator=loc, ok=True,
            chosen=Candidate(box=(300, 120, 360, 145), text="7+3=?", score=0.99),
            shot=shot,
        )
        step = Step(action="input", value_source="captcha",
                    captcha_ratio=[-0.5, -0.5, 0.5, 0.5])

        self.assertEqual(engine._solve_captcha(step, res), "10")


if __name__ == "__main__":
    unittest.main()
