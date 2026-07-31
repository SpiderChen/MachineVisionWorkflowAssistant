import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.captcha import parse_alnum, parse_arith, solve_alnum, solve_arith
from app.engine import Engine, StepError
from app.flows import (
    Step, StepLocator, derive_dynamic_arith_locator, relative_box_ratio,
)
from app.input_ctl import InputController, InputError, _DummyBackend
from app.locators import Match, TextLocator
from app.vision import Candidate, LocateResult, Screenshot, _text_matches


class _FakeVision:
    def __init__(self, candidates):
        self.candidates = candidates

    def ocr_lines(self, shot):
        return list(self.candidates)


class DynamicCaptchaTests(unittest.TestCase):
    def test_parse_arith_recovers_common_ocr_confusions(self):
        self.assertEqual(parse_arith("B 十 S = ?"), ("13", "8+5"))
        self.assertEqual(parse_arith("O：S"), ("0", "0/5"))
        self.assertEqual(parse_arith("9–4"), ("5", "9-4"))

    def test_solver_reorders_split_ocr_candidates_left_to_right(self):
        class _CaptchaVision:
            def ocr_image(self, image):
                return [
                    Candidate(box=(70, 0, 90, 20), text="5=?", score=0.95),
                    Candidate(box=(10, 0, 40, 20), text="3+", score=0.96),
                ]

        img = np.full((30, 100, 3), 255, dtype=np.uint8)

        self.assertEqual(solve_arith(_CaptchaVision(), img), ("8", "3+5"))

    def test_solver_continues_after_one_ocr_variant_errors(self):
        class _FlakyCaptchaVision:
            def __init__(self):
                self.calls = 0

            def ocr_image(self, image):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary OCR failure")
                return [Candidate(box=(0, 0, 50, 20), text="6×2=?", score=0.9)]

        vision = _FlakyCaptchaVision()
        img = np.full((30, 100, 3), 255, dtype=np.uint8)

        self.assertEqual(solve_arith(vision, img), ("12", "6*2"))
        self.assertEqual(vision.calls, 2)

    def test_parse_alnum_normalizes_fullwidth_and_preserves_case(self):
        self.assertEqual(parse_alnum("Ａ b-3 ７"), "Ab37")
        self.assertEqual(parse_alnum("xY9"), "xY9")
        self.assertIsNone(parse_alnum("A7"))
        self.assertIsNone(parse_alnum("ABCDEFGHI"))

    def test_alnum_solver_reorders_split_candidates_before_accepting_fragment(self):
        class _CaptchaVision:
            def ocr_image(self, image):
                return [
                    Candidate(box=(55, 0, 90, 20), text="7dK", score=0.95),
                    Candidate(box=(5, 0, 40, 20), text="aB", score=0.96),
                ]

        img = np.full((30, 100, 3), 255, dtype=np.uint8)

        self.assertEqual(solve_alnum(_CaptchaVision(), img), ("aB7dK", "aB7dK"))

    def test_alnum_solver_rejects_low_confidence_guess(self):
        class _CaptchaVision:
            def ocr_image(self, image):
                return [Candidate(box=(5, 0, 90, 20), text="aB7d", score=0.2)]

        img = np.full((30, 100, 3), 255, dtype=np.uint8)

        self.assertIsNone(solve_alnum(_CaptchaVision(), img))

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

    def test_captcha_locator_roundtrips_separately_from_input_locator(self):
        base = StepLocator(kind="text", anchor="用户名", match="exact")
        captcha = StepLocator(kind="text", anchor="0*9=?", match="arith")
        step = Step(action="input", value_source="captcha",
                    locator=base, captcha_locator=captcha)

        restored = Step.from_dict(step.to_dict())

        self.assertEqual(restored.locator.anchor, "用户名")
        self.assertEqual(restored.captcha_locator.anchor, "0*9=?")
        self.assertEqual(restored.captcha_locator.match, "arith")

    def test_captcha_mode_roundtrips_and_old_config_defaults_to_arith(self):
        alnum = Step(action="input", value_source="captcha", captcha_mode="alnum")

        restored = Step.from_dict(alnum.to_dict())
        legacy = Step.from_dict({"action": "input", "value_source": "captcha"})

        self.assertEqual(restored.captcha_mode, "alnum")
        self.assertEqual(legacy.captcha_mode, "arith")

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

    def test_captcha_crop_ratio_is_relative_to_answer_field(self):
        answer_field = Candidate(box=(100, 100, 300, 140), text="", score=0.99)

        ratio = relative_box_ratio((320, 100, 380, 140), answer_field)

        self.assertEqual(ratio, [0.6, -0.5, 0.9, 0.5])

    def test_engine_calculates_directly_from_dynamic_anchor(self):
        engine = Engine.__new__(Engine)
        input_loc = TextLocator(anchor="用户名", match=Match.EXACT)
        captcha_loc = StepLocator(kind="text", anchor="0*9=?", match="arith")
        shot = Screenshot(img=np.zeros((300, 500, 3), dtype=np.uint8))
        res = LocateResult(
            locator=input_loc, ok=True,
            chosen=Candidate(box=(100, 120, 160, 145), text="用户名", score=0.99),
            shot=shot,
        )
        step = Step(action="input", value_source="captcha",
                    locator=StepLocator(kind="text", anchor="用户名", match="exact"),
                    captcha_locator=captcha_loc,
                    captcha_ratio=[-0.5, -0.5, 0.5, 0.5])

        engine.vision = type("V", (), {
            "locate": lambda self, loc, shot: LocateResult(
                locator=loc, ok=True,
                chosen=Candidate(box=(300, 120, 360, 145), text="7+3=?", score=0.99),
                shot=shot,
            )
        })()

        self.assertEqual(engine._solve_captcha(step, res), "10")

    def test_engine_uses_answer_relative_crop_when_dynamic_anchor_misses(self):
        engine = Engine.__new__(Engine)
        shot_img = np.zeros((300, 500, 3), dtype=np.uint8)
        shot_img[100:140, 320:380] = 255
        shot = Screenshot(img=shot_img)
        input_res = LocateResult(
            locator=TextLocator(anchor="验证码", match=Match.EXACT), ok=True,
            chosen=Candidate(box=(100, 100, 300, 140), text="", score=0.99),
            click_point=(200, 120), shot=shot,
        )
        step = Step(
            action="input", value_source="captcha",
            locator=StepLocator(kind="text", anchor="验证码", match="exact"),
            captcha_locator=StepLocator(kind="text", anchor="0*9=?", match="arith"),
            captcha_ratio=[0.6, -0.5, 0.9, 0.5],
        )
        engine.vision = type("V", (), {
            "locate": lambda self, loc, current_shot: LocateResult(
                locator=loc, ok=False, shot=current_shot),
            "capture": lambda self: shot,
        })()

        def _solve(_vision, crop):
            self.assertEqual(crop.shape[:2], (40, 60))
            self.assertTrue(np.all(crop == 255))
            return "12", "6+6"

        with patch("app.captcha.solve_arith", side_effect=_solve):
            self.assertEqual(engine._solve_captcha(step, input_res), "12")

    def test_engine_uses_alnum_solver_and_ignores_arith_dynamic_anchor(self):
        engine = Engine.__new__(Engine)
        shot_img = np.zeros((300, 500, 3), dtype=np.uint8)
        shot_img[100:140, 320:380] = 255
        shot = Screenshot(img=shot_img)
        input_res = LocateResult(
            locator=TextLocator(anchor="验证码", match=Match.EXACT), ok=True,
            chosen=Candidate(box=(100, 100, 300, 140), text="", score=0.99),
            click_point=(200, 120), shot=shot,
        )
        step = Step(
            action="input", value_source="captcha", captcha_mode="alnum",
            locator=StepLocator(kind="text", anchor="验证码", match="exact"),
            captcha_locator=StepLocator(kind="text", anchor="0*9=?", match="arith"),
            captcha_ratio=[0.6, -0.5, 0.9, 0.5],
        )
        engine.vision = type("V", (), {})()

        with patch("app.captcha.solve_alnum",
                   return_value=("aB7d", "aB7d")) as solve_alnum_mock, \
                patch("app.captcha.solve_arith") as solve_arith_mock:
            self.assertEqual(engine._solve_captcha(step, input_res), "aB7d")

        solve_alnum_mock.assert_called_once()
        solve_arith_mock.assert_not_called()

    def test_engine_clicks_captcha_to_refresh_and_retries(self):
        engine = Engine.__new__(Engine)
        first_shot = Screenshot(img=np.zeros((300, 500, 3), dtype=np.uint8))
        second_shot = Screenshot(img=np.ones((300, 500, 3), dtype=np.uint8))
        answer = Candidate(box=(100, 100, 300, 140), text="", score=0.99)
        input_res = LocateResult(
            locator=TextLocator(anchor="验证码", match=Match.EXACT), ok=True,
            chosen=answer, click_point=(200, 120), shot=first_shot,
        )
        refreshed_res = LocateResult(
            locator=input_res.locator, ok=True, chosen=answer,
            click_point=(200, 120), shot=second_shot,
        )
        step = Step(
            action="input", value_source="captcha",
            locator=StepLocator(kind="text", anchor="验证码", match="exact"),
            captcha_ratio=[0.6, -0.5, 0.9, 0.5],
        )

        class _Vision:
            def __init__(self, case):
                self.case = case

            def capture(self):
                return second_shot

            def locate(self, loc, shot):
                self.case.assertIs(shot, second_shot)
                return refreshed_res

            def to_screen(self, shot, point):
                return point

        class _Inputs:
            def __init__(self):
                self.clicks = []

            def click(self, x, y):
                self.clicks.append((x, y))

        engine.vision = _Vision(self)
        engine.inputs = _Inputs()

        with patch("app.captcha.solve_arith",
                   side_effect=[None, ("12", "6+6")]) as solve, \
                patch("app.engine.time.sleep"):
            self.assertEqual(engine._solve_captcha(step, input_res), "12")

        self.assertEqual(engine.inputs.clicks, [(350, 120)])
        self.assertEqual(solve.call_count, 2)

    def test_engine_refreshes_captcha_up_to_ten_times(self):
        engine = Engine.__new__(Engine)
        shot = Screenshot(img=np.zeros((300, 500, 3), dtype=np.uint8))
        answer = Candidate(box=(100, 100, 300, 140), text="", score=0.99)
        input_res = LocateResult(
            locator=TextLocator(anchor="验证码", match=Match.EXACT), ok=True,
            chosen=answer, click_point=(200, 120), shot=shot,
        )
        step = Step(
            action="input", value_source="captcha",
            locator=StepLocator(kind="text", anchor="验证码", match="exact"),
            captcha_ratio=[0.6, -0.5, 0.9, 0.5],
        )

        class _Vision:
            def capture(self):
                return shot

            def locate(self, loc, current_shot):
                return input_res

            def to_screen(self, current_shot, point):
                return point

        class _Inputs:
            def __init__(self):
                self.clicks = []

            def click(self, x, y):
                self.clicks.append((x, y))

        engine.vision = _Vision()
        engine.inputs = _Inputs()

        with patch("app.captcha.solve_arith", return_value=None) as solve, \
                patch("app.engine.time.sleep"):
            with self.assertRaisesRegex(StepError, "连续刷新 10 次"):
                engine._solve_captcha(step, input_res)

        self.assertEqual(len(engine.inputs.clicks), 10)
        self.assertEqual(solve.call_count, 11)

    def test_input_step_clicks_answer_field_then_types_calculated_result(self):
        engine = Engine.__new__(Engine)
        shot = Screenshot(img=np.zeros((300, 500, 3), dtype=np.uint8))
        input_res = LocateResult(
            locator=TextLocator(anchor="验证码", match=Match.EXACT), ok=True,
            chosen=Candidate(box=(100, 120, 260, 150), text="", score=0.99),
            click_point=(180, 135), shot=shot,
        )
        captcha_res = LocateResult(
            locator=TextLocator(anchor="0*9=?", match=Match.ARITH), ok=True,
            chosen=Candidate(box=(300, 120, 360, 145), text="7+3=?", score=0.99),
            shot=shot,
        )

        class _Vision:
            def locate(self, loc, current_shot):
                return captcha_res

            def to_screen(self, current_shot, point):
                return point

        class _Inputs:
            def __init__(self):
                self.calls = []

            def click(self, x, y):
                self.calls.append(("click", x, y))

            def select_all_clear(self):
                self.calls.append(("clear",))

            def type_text(self, value):
                self.calls.append(("type", value))

        engine.vision = _Vision()
        engine.inputs = _Inputs()
        engine.s = SimpleNamespace(engine=SimpleNamespace(verify_input=False))
        engine._wait_step = lambda step, loc, timeout: input_res
        engine._emit = lambda *args, **kwargs: None
        step = Step(
            action="input", value_source="captcha", clear_first=True,
            locator=StepLocator(kind="text", anchor="验证码", match="exact"),
            captcha_locator=StepLocator(kind="text", anchor="0*9=?", match="arith"),
            captcha_ratio=[-0.5, -0.5, 0.5, 0.5],
        )

        engine._input_step(step, step.build_locator(), 3.0, {})

        self.assertEqual(engine.inputs.calls, [
            ("click", 180, 135),
            ("clear",),
            ("type", "10"),
        ])

    def test_dummy_backend_fails_instead_of_reporting_fake_success(self):
        inputs = InputController.__new__(InputController)
        inputs.backend = _DummyBackend()

        with self.assertRaisesRegex(InputError, "不要从 WSL 启动"):
            inputs.click(100, 200)


if __name__ == "__main__":
    unittest.main()
