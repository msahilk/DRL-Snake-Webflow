from typing import Dict, Any, Optional, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTIONS = [
    "home","start","type_first","type_last","next",
    "type_email","type_password","submit","finish"
]
PAGE_DEPTH = {"home": 0, "signup1": 1, "signup2": 2, "confirm": 3}

class MockWebFlowEnv(gym.Env):
    """
    Offline mock of the 2-step signup flow.

    Modes:
      - completer: finish efficiently/correctly
      - bug_hunter: surface server/validation errors
      - fuzzer (NAIVE): encourages creating validation errors; very light penalties,
        mild exploration rewards, minimal guidance toward completion.
        Expect: many validation_errors, low success.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        reward_mode: str = "completer",
        seed: Optional[int] = None,
        step_limit: int = 100,
        curriculum: bool = True
    ):
        super().__init__()
        assert reward_mode in ("completer","fuzzer","bug_hunter")
        self.reward_mode = reward_mode
        self.step_limit = step_limit
        self.curriculum = curriculum
        self.np_random = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        # simulation state
        self._page = "home"
        self._err = 0
        self._typed = {"first":"", "last":"", "email":"", "password":""}
        self._typed_flags = {"first": False, "last": False, "email": False, "password": False}

        # metrics/state
        self.steps = 0
        self.clicked_unique: Set[str] = set()
        self.latency = 0.0
        self.validation_errors = 0
        self.status_500 = 0
        self.success = 0
        self.no_progress = 0
        self.pingpong_count = 0

        # exploration helpers
        self._visited_pages: Set[str] = set()
        self._seen_errors: Set[str] = set()
        self._max_depth: int = 0

        # guards
        self._last_action: Optional[str] = None
        self._stale_error_streak: int = 0

    # ---------- helpers ----------
    def _where(self): return self._page
    def _err_banner(self): return self._err
    def _inputs_count(self): return {"home":0,"signup1":2,"signup2":2,"confirm":0}[self._page]

    def _obs(self):
        page = self._where()
        onehot = {"home":[1,0,0,0],"signup1":[0,1,0,0],"signup2":[0,0,1,0],"confirm":[0,0,0,1]}[page]
        err = float(self._err_banner())
        inputs = min(1.0,self._inputs_count()/6.0)
        clicked = min(1.0,len(self.clicked_unique)/16.0)
        lat = min(1.0,self.latency/1.0)
        stepn = min(1.0,self.steps/max(1,self.step_limit))
        return np.array([*onehot,err,inputs,clicked,lat,stepn],dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.steps = 0
        self.clicked_unique.clear()
        self._typed = {"first":"", "last":"", "email":"", "password":""}
        self._typed_flags = {k: False for k in self._typed_flags}
        self.latency = 0.0
        self.validation_errors = 0
        self.status_500 = 0
        self.success = 0
        self.no_progress = 0
        self.pingpong_count = 0
        self._visited_pages.clear()
        self._seen_errors.clear()
        self._max_depth = 0
        self._last_action = None
        self._stale_error_streak = 0

        # simple curriculum start (home/signup1/signup2)
        r = self.np_random.random() if self.curriculum else 0.0
        if r < 0.6:
            self._page, self._err = "home", 0
        elif r < 0.85:
            self._page, self._err = "signup1", 0
        else:
            self._page, self._err = "signup2", 0

        self._visited_pages.add(self._page)
        self._max_depth = PAGE_DEPTH[self._page]
        return self._obs(), self._info()

    def step(self, action:int):
        self.steps += 1
        a = ACTIONS[int(action)]
        prev_err, prev_page = self._err_banner(), self._where()
        prev_clicked = len(self.clicked_unique)
        prev_flags = self._typed_flags.copy()

        invalid=False
        def require(page_name:str):
            nonlocal invalid
            if self._page!=page_name: invalid=True

        # ----- UI simulation -----
        if a=="home": self._page="home"
        elif a=="start":
            require("home")
            if not invalid:
                self.clicked_unique.add("#start")
                self._page="signup1"; self._err=0
        elif a=="type_first":
            require("signup1")
            if not invalid:
                self.clicked_unique.add("#first")
                self._typed["first"]="Alex"; self._typed_flags["first"]=True
        elif a=="type_last":
            require("signup1")
            if not invalid:
                self.clicked_unique.add("#last")
                self._typed["last"]="Smith"; self._typed_flags["last"]=True
        elif a=="next":
            require("signup1")
            if not invalid:
                self.clicked_unique.add("#next1")
                ok = self._typed["first"] and self._typed["last"]
                self._err = 0 if ok else 1
                if ok: self._page="signup2"
        elif a=="type_email":
            require("signup2")
            if not invalid:
                self.clicked_unique.add("#email")
                self._typed["email"]="alex@example.com"
                self._typed_flags["email"]=True
        elif a=="type_password":
            require("signup2")
            if not invalid:
                self.clicked_unique.add("#password")
                self._typed["password"]="secret"
                self._typed_flags["password"]=True
        elif a=="submit":
            require("signup2")
            if not invalid:
                self.clicked_unique.add("#submit")
                ok = self._typed["email"] and self._typed["password"]
                self._err = 0 if ok else 1
                if ok: self._page="confirm"; self.success=1
        elif a=="finish":
            require("confirm")
            if not invalid:
                self.clicked_unique.add("#finish")
                self._page="home"; self._err=0

        # ----- deltas -----
        page=self._where()
        err=self._err_banner()
        delta_clicks=max(0,len(self.clicked_unique)-prev_clicked)
        newly_filled=sum(int(self._typed_flags[k] and not prev_flags[k])
                         for k in ("first","last","email","password"))

        if err and not prev_err: self.validation_errors+=1
        success=int(page=="confirm")
        if success: self.success=1

        # stale-error detection (keeps clicking next/submit under same banner)
        stale_error=(err==1 and prev_err==1 and page==prev_page and a in ("next","submit") and delta_clicks==0)
        self._stale_error_streak = self._stale_error_streak+1 if stale_error else 0

        # ----- reward shaping -----
        reward=0.0
        forward_pairs={("home","signup1"),("signup1","signup2"),("signup2","confirm")}
        backward_pairs={("signup1","home"),("signup2","signup1")}
        pair=(prev_page,page)

        if self.reward_mode=="completer":
            reward-=0.002
            if page!=prev_page:
                if pair in forward_pairs: reward+=0.5
                elif pair in backward_pairs: reward-=0.2
            if a=="type_first" and self._typed_flags["first"]: reward+=0.1
            if a=="type_last" and self._typed_flags["last"]: reward+=0.1
            if a=="type_email" and self._typed_flags["email"]: reward+=0.1
            if a=="type_password" and self._typed_flags["password"]: reward+=0.1
            if success: reward+=10.0
            if a=="finish" and prev_page!="confirm": reward-=0.05
            if err and not prev_err: reward-=0.2

        elif self.reward_mode=="fuzzer":
            # NAIVE fuzzing: reward seeing errors & poking the UI; minimal penalties.
            reward -= 0.005  # small living cost

            # Mild depth incentive (but not strong—won't push to finish often)
            prev_depth, depth = PAGE_DEPTH[prev_page], PAGE_DEPTH[page]
            if pair in forward_pairs and depth > self._max_depth:
                reward += 0.2
                self._max_depth = depth
            if pair in backward_pairs:
                reward -= 0.1

            # Exploration & interaction
            reward += 0.05 * delta_clicks
            if page not in self._visited_pages:
                self._visited_pages.add(page)
                reward += 0.2

            # Validation error discovery (big driver)
            if err and not prev_err:
                site = f"{page}:err"
                if site not in self._seen_errors:
                    self._seen_errors.add(site)
                    reward += 0.5   # large bump → encourages creating banners

            # Very small bonus for first-time field fills (so it sometimes moves forward)
            if newly_filled > 0:
                reward += 0.1 * newly_filled

            # Fixing validation? tiny credit only (keeps “error creation” attractive)
            if (prev_err == 1 and err == 0):
                reward += 0.1

            # Soft stale-error penalty so it can keep poking a bit
            if stale_error:
                reward -= (0.01 + 0.005 * min(10, self._stale_error_streak))

            # Repeating same action exact state → tiny penalty
            repeated_same=(a==self._last_action and page==prev_page and delta_clicks==0 and not (err and not prev_err))
            if repeated_same:
                reward -= 0.005


        if invalid: reward-=0.15

        # ----- termination -----
        if pair in backward_pairs: self.pingpong_count+=1
        elif pair in forward_pairs: self.pingpong_count=0
        progress=(PAGE_DEPTH[page]>PAGE_DEPTH[prev_page]) or (err and not prev_err) or (newly_filled>0)
        self.no_progress=0 if progress else self.no_progress+1
        terminated=bool(success)
        truncated=(self.steps>=self.step_limit or self.no_progress>12 or self.pingpong_count>12)

        self._last_action=a
        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self)->Dict[str,Any]:
        return {
            "steps":self.steps,
            "page_home":int(self._where()=="home"),
            "page_signup1":int(self._where()=="signup1"),
            "page_signup2":int(self._where()=="signup2"),
            "page_confirm":int(self._where()=="confirm"),
            "error_banner":self._err_banner(),
            "clicked_unique":len(self.clicked_unique),
            "latency_ms":int(self.latency*1000),
            "status_500":self.status_500,
            "validation_errors":self.validation_errors,
            "success":self.success,
            "no_progress":self.no_progress,
            "pingpong_count":self.pingpong_count,
        }

    def render(self): return f"page={self._page} steps={self.steps} err={self._err}"
