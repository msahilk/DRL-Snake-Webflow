import time
from typing import Dict, Any, Optional, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

ACTIONS = [
    "home","start","type_first","type_last","next",
    "type_email","type_password","submit","finish"
]
PAGE_DEPTH = {"home": 0, "signup1": 1, "signup2": 2, "confirm": 3}

class WebFlowEnv(gym.Env):
    """
    Real Playwright-driven env for the 2-step signup flow.

    Modes:
      - completer / bug_hunter: as before
      - fuzzer (NAIVE): mirrors the mock naive shaping to encourage banners
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000/",
        reward_mode: str = "completer",
        seed: Optional[int] = None,
        step_limit: int = 120,
        headless: bool = True,
        fast_mode: bool = True,
        curriculum: bool = True
    ):
        super().__init__()
        assert reward_mode in ("completer","fuzzer","bug_hunter")
        self.base_url = base_url
        self.reward_mode = reward_mode
        self.step_limit = step_limit
        self.headless = headless
        self.fast_mode = fast_mode
        self.curriculum = curriculum
        self.np_random = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._pw = None
        self.browser = None
        self.page = None

        self._typed_flags = {"first": False, "last": False, "email": False, "password": False}
        self.invalid_clicks = 0
        self.pingpong_count = 0

        self.steps = 0
        self.clicked_unique: Set[str] = set()
        self.latency = 0.0
        self.last_status_500 = 0
        self.validation_errors = 0
        self.success = 0
        self.no_progress = 0

        self._visited_pages: Set[str] = set()
        self._seen_errors: Set[str] = set()
        self._max_depth: int = 0

        self._last_action: Optional[str] = None
        self._stale_error_streak: int = 0

    # -------- browser helpers --------
    def _start_browser(self):
        self._pw = sync_playwright().start()
        launch_args = dict(headless=self.headless)
        if self.fast_mode:
            launch_args["args"] = [
                "--disable-gpu","--disable-dev-shm-usage","--no-sandbox",
                "--disable-background-networking","--disable-renderer-backgrounding",
                "--disable-features=IsolateOrigins,site-per-process",
            ]
        self.browser = self._pw.chromium.launch(**launch_args)
        self.page = self.browser.new_page()
        if self.fast_mode:
            self.page.set_default_timeout(500)
            def _route(route):
                if route.request.resource_type in {"image","stylesheet","font"}:
                    return route.abort()
                return route.continue_()
            self.page.route("**/*", _route)

    def _stop_browser(self):
        try:
            if self.page: self.page.close()
            if self.browser: self.browser.close()
            if self._pw: self._pw.stop()
        except:
            pass
        self.page = self.browser = self._pw = None

    def _goto(self, url: str):
        t0 = time.perf_counter()
        try:
            resp = self.page.goto(url, wait_until="domcontentloaded" if self.fast_mode else "load")
        except Exception:
            resp = None
        self.latency = time.perf_counter() - t0
        self.last_status_500 = 1 if (resp and resp.status == 500) else 0

    def _click(self, sel: str):
        if self.page.query_selector(sel):
            self.clicked_unique.add(sel)
            t0 = time.perf_counter()
            try:
                self.page.click(sel, timeout=300 if self.fast_mode else 1000)
            except:
                pass
            self.latency = time.perf_counter() - t0
        else:
            self.invalid_clicks += 1

    def _type(self, sel: str, text: str):
        el = self.page.query_selector(sel)
        if el:
            self.clicked_unique.add(sel)
            try:
                el.fill("")
                el.type(text, delay=0)
            except:
                pass
        else:
            self.invalid_clicks += 1

    def _where(self) -> str:
        url = (self.page.url or "")
        if url.endswith("/"): url = url[:-1]
        if url.endswith("confirm"): return "confirm"
        if url.endswith("signup2"): return "signup2"
        if url.endswith("signup1"): return "signup1"
        return "home"

    def _err_banner(self) -> int:
        return 1 if self.page.locator(".err").count() > 0 else 0

    def _inputs_count(self) -> int:
        return self.page.locator("input").count()

    def _obs(self) -> np.ndarray:
        page = self._where()
        onehot = {"home":[1,0,0,0], "signup1":[0,1,0,0], "signup2":[0,0,1,0], "confirm":[0,0,0,1]}[page]
        err = float(self._err_banner())
        inputs = min(1.0, self._inputs_count()/6.0)
        clicked = min(1.0, len(self.clicked_unique)/16.0)
        lat = min(1.0, self.latency/1.0)
        stepn = min(1.0, self.steps/max(1, self.step_limit))
        return np.array([*onehot, err, inputs, clicked, lat, stepn], dtype=np.float32)

    # -------- gym api --------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.page is None:
            self._start_browser()

        self.steps = 0
        self.clicked_unique.clear()
        self.latency = 0.0
        self.last_status_500 = 0
        self.validation_errors = 0
        self.success = 0
        self.no_progress = 0
        self.pingpong_count = 0
        self._typed_flags = {"first": False, "last": False, "email": False, "password": False}
        self._visited_pages.clear()
        self._seen_errors.clear()
        self._max_depth = 0
        self.invalid_clicks = 0
        self._last_action = None
        self._stale_error_streak = 0

        # light curriculum start
        start_p = self.np_random.random() if self.curriculum else 0.0
        if start_p > 0.85:
            self._goto(self.base_url + "signup2")
        elif start_p > 0.60:
            self._goto(self.base_url + "signup1")
        else:
            self._goto(self.base_url + "")
        self._visited_pages.add(self._where())
        self._max_depth = PAGE_DEPTH[self._where()]

        return self._obs(), self._info()

    def step(self, action: int):
        self.steps += 1
        a = ACTIONS[int(action)]

        prev_err = self._err_banner()
        prev_page = self._where()
        prev_clicked = len(self.clicked_unique)
        prev_flags = self._typed_flags.copy()
        prev_invalids = self.invalid_clicks
        self.latency = 0.0

        # ----- actions -----
        if a == "home":
            self._goto(self.base_url + "")
        elif a == "start":
            self._click("#start")
        elif a == "type_first":
            self._type("#first", "Alex")
            self._typed_flags["first"] = True
        elif a == "type_last":
            self._type("#last", "Smith")
            self._typed_flags["last"] = True
        elif a == "next":
            self._click("#next1")
        elif a == "type_email":
            self._type("#email", "alex@example.com")
            self._typed_flags["email"] = True
        elif a == "type_password":
            self._type("#password", "secret123")
            self._typed_flags["password"] = True
        elif a == "submit":
            self._click("#submit")
        elif a == "finish":
            self._click("#finish")

        # ----- outcomes -----
        page = self._where()
        err = self._err_banner()
        if err and not prev_err:
            self.validation_errors += 1
        success = int(page == "confirm")
        if success:
            self.success = 1

        first_visit = False
        if page not in self._visited_pages:
            self._visited_pages.add(page)
            first_visit = True

        delta_clicks = max(0, len(self.clicked_unique) - prev_clicked)
        delta_invalids = max(0, self.invalid_clicks - prev_invalids)
        newly_filled = sum(int(self._typed_flags[k] and not prev_flags[k]) for k in ("first","last","email","password"))

        stale_error = (err == 1 and prev_err == 1 and page == prev_page and a in ("next","submit") and delta_clicks == 0)
        self._stale_error_streak = self._stale_error_streak + 1 if stale_error else 0

        # ----- rewards -----
        reward = 0.0
        forward_pairs = {("home","signup1"), ("signup1","signup2"), ("signup2","confirm")}
        backward_pairs = {("signup1","home"), ("signup2","signup1")}
        pair = (prev_page, page)

        if self.reward_mode == "completer":
            reward -= 0.002
            if page != prev_page:
                if pair in forward_pairs:   reward += 0.5
                elif pair in backward_pairs: reward -= 0.2

            if a == "type_first"    and self._typed_flags["first"]:    reward += 0.1
            if a == "type_last"     and self._typed_flags["last"]:     reward += 0.1
            if a == "type_email"    and self._typed_flags["email"]:    reward += 0.1
            if a == "type_password" and self._typed_flags["password"]: reward += 0.1

            if success: reward += 10.0
            if a == "finish" and prev_page != "confirm": reward -= 0.05
            if err and not prev_err: reward -= 0.2

        elif self.reward_mode == "fuzzer":
            # NAIVE: reward banners & poking around
            reward -= 0.005

            prev_depth = PAGE_DEPTH[prev_page]
            depth = PAGE_DEPTH[page]
            if pair in forward_pairs and depth > self._max_depth:
                reward += 0.2
                self._max_depth = depth
            if pair in backward_pairs:
                reward -= 0.1

            reward += 0.05 * delta_clicks
            if first_visit: reward += 0.2

            if err and not prev_err:
                site = f"{page}:err"
                if site not in self._seen_errors:
                    self._seen_errors.add(site)
                    reward += 0.5

            if newly_filled > 0:
                reward += 0.1 * newly_filled

            if (prev_err == 1 and err == 0):
                reward += 0.1

            if stale_error:
                reward -= (0.01 + 0.005 * min(10, self._stale_error_streak))

            repeated_same = (a == self._last_action and page == prev_page and delta_clicks == 0 and not (err and not prev_err))
            if repeated_same:
                reward -= 0.005

        if delta_invalids > 0:
            reward -= 0.15 * delta_invalids

        is_forward = pair in forward_pairs
        new_error  = (err and not prev_err)
        found_500  = bool(self.last_status_500)

        if pair in backward_pairs:
            self.pingpong_count += 1
        elif is_forward:
            self.pingpong_count = 0

        progress = (PAGE_DEPTH[page] > PAGE_DEPTH[prev_page]) or new_error or found_500 or (newly_filled > 0)
        self.no_progress = 0 if progress else (self.no_progress + 1)

        terminated = bool(success) or found_500
        truncated = (
            self.steps >= self.step_limit
            or self.no_progress > 12
            or self.pingpong_count > 12
        )

        self._last_action = a
        obs = self._obs()
        info = self._info()
        return obs, reward, terminated, truncated, info

    def _info(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "page_home": int(self._where()=="home"),
            "page_signup1": int(self._where()=="signup1"),
            "page_signup2": int(self._where()=="signup2"),
            "page_confirm": int(self._where()=="confirm"),
            "error_banner": self._err_banner(),
            "clicked_unique": len(self.clicked_unique),
            "latency_ms": int(self.latency*1000),
            "status_500": self.last_status_500,
            "validation_errors": self.validation_errors,
            "success": self.success,
            "no_progress": self.no_progress,
            "invalid_clicks": self.invalid_clicks,
            "pingpong_count": self.pingpong_count,
        }

    def render(self) -> str:
        return f"URL={self.page.url if self.page else ''} steps={self.steps}"

    def close(self):
        self._stop_browser()
