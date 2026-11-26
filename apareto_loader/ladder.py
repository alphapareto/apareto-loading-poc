"""Think-end ladder controller: IncNStar and LadderController.

Torch-free, so an offline replay harness and the engine wrapper (apareto_loader/vllm027/ladder.py) run the same
class.
"""
from collections import Counter, deque

M1, B1 = (1 << 61) - 1, 131_542_391
M2, B2 = (1 << 59) - 55, 916_132_831


class IncNStar:
    """Longest n-gram repeated >= `repeats` times, censored at `maxn`, maintained incrementally.

    n* is monotone non-decreasing in the prefix, so this never searches: it holds the count table
    for window length n*+1 only, adds the single new window per push, and escalates when a count
    reaches `repeats`. An escalation rebuilds in O(len), bounded by `maxn` times per request.
    Exact-equal to the batch and tuple-Counter forms of the same computation.
    """

    __slots__ = ("repeats", "maxn", "n", "seq", "cur_m", "counts", "h1", "h2", "p1", "p2")

    def __init__(self, repeats=4, maxn=320):
        self.repeats, self.maxn = repeats, maxn
        self.n = 0
        self.seq = []
        self.cur_m = None
        self.counts = Counter()
        self.h1 = self.h2 = 0
        self.p1 = self.p2 = 0

    def _rebuild(self, m):
        self.cur_m = m
        self.counts = Counter()
        seq = self.seq
        if len(seq) < m:
            self.h1 = self.h2 = 0
            return 0
        self.p1, self.p2 = pow(B1, m - 1, M1), pow(B2, m - 1, M2)
        h1 = h2 = 0
        for x in seq[:m]:
            h1 = (h1 * B1 + x) % M1
            h2 = (h2 * B2 + x) % M2
        self.counts[(h1, h2)] = 1
        best = 1
        for i in range(m, len(seq)):
            h1 = ((h1 - seq[i - m] * self.p1) * B1 + seq[i]) % M1
            h2 = ((h2 - seq[i - m] * self.p2) * B2 + seq[i]) % M2
            k = (h1, h2)
            c = self.counts[k] + 1
            self.counts[k] = c
            if c > best:
                best = c
        self.h1, self.h2 = h1, h2
        return best

    def push(self, x):
        self.seq.append(x)
        while True:
            m = self.n + 1
            if m > self.maxn or len(self.seq) < m:
                return self.n
            if self.cur_m != m:
                best = self._rebuild(m)
            else:
                seq = self.seq
                i = len(seq) - 1
                self.h1 = ((self.h1 - seq[i - m] * self.p1) * B1 + seq[i]) % M1
                self.h2 = ((self.h2 - seq[i - m] * self.p2) * B2 + seq[i]) % M2
                k = (self.h1, self.h2)
                best = self.counts[k] + 1
                self.counts[k] = best
            if best >= self.repeats:
                self.n = m
                self.cur_m = None
                continue
            return self.n


class LadderController:
    """One request's ladder. Torch-free on purpose: the offline replay harness runs this same class.

    cfg keys (all token counts are GENERATED tokens):
      fixed        [[at_tok, bar, inj_ids], ...]   bar applies from at_tok until the next entry
      ramp         {start, thr0, decay, per, floor, inj_ids, max_fires}
      final        {at_tok, inj_ids}               unconditional
      tag_id       disarm-forever token (248069)
      ends_nl      callable(token_id) -> bool   (built from ends_nl_ids over the wire)
      ends_nl_ids  token ids whose text ends in a newline -- the line-boundary set
      nl_id        token forced when the wait cap expires
      wait_cap     decode steps to wait for a natural boundary before forcing nl_id
      final_wait   the same cap for the unconditional close, which cannot eat the answer reserve
      fence_ids       token ids whose text contains a 3+ backtick run (fence tokens)
      ws_ids          token ids whose text is spaces/tabs only (keep "line start" alive; a bare
                      fence may be followed by spaces before its newline)
      fence_close_ids ids for "```\\n", emitted before a forced close if the fence is open
      safe_final      require a boundary OUTSIDE code for the unconditional close (default True)
      safe_nudge      nudges wait for a boundary outside code up to nudge_wait (default True)
      nudge_wait      tokens a safe nudge waits before falling back to the any-line-end rule (1500)
      fence0          fence state at request start for checkpointed runs (default 0 = closed)
      t0              generated-token count already elapsed (checkpointed runs); default 0
      fired           indices of fixed gates already consumed in earlier checkpoints
      ramp_fires      ramp nudges already spent, against ramp.max_fires
      repeats, maxn
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ns = IncNStar(cfg.get("repeats", 4), cfg.get("maxn", 320))
        # t0/fired/fires carry gate state ACROSS a checkpointed run. A checkpointed run ends a
        # request every I generated tokens and starts a new one, so without these the gate clock
        # would reset at every checkpoint and, with I below the first gate, NO gate could ever fire.
        # They default to a cold start, so an ordinary single-request run is unaffected.
        self.t = int(cfg.get("t0", 0))
        self.fires = int(cfg.get("ramp_fires", 0))
        # bar(t) is continuous, so a crossed FIXED gate would re-arm and re-fire every step until
        # the next gate, so each fixed gate is one-shot.
        self.fixed_fired = set(cfg.get("fired") or ())
        self.armed_at = None
        self.armed_inj = None
        self.armed_is_final = False
        self.armed_gate = None
        self.pending = deque()
        self.disarmed = False
        self.events = []
        self.fence = int(cfg.get("fence0", 0))
        self._fence_ids = frozenset(cfg.get("fence_ids") or ())
        self._ws_ids = frozenset(cfg.get("ws_ids") or ())
        # A fence token has been seen at line start; the NEXT non-blank token decides opener/closer.
        self._fence_pending = False
        self._line_start = True
        self.fence_events = 0

    def _log(self, event, d):
        rec = dict(d, event=event)
        self.events.append(rec)
        sid = self.cfg.get("session")
        if sid is not None:
            print(f"[ladder] {sid} {event} " +
                  " ".join(f"{k}={v}" for k, v in d.items()), flush=True)

    def _set_fence(self, state):
        self._fence_pending = False
        if state != self.fence:
            self.fence = state
            self.fence_events += 1
            self._log("fence", {"at_tok": self.t, "state": state})

    def bar(self):
        c = self.cfg
        if self.t >= c["final"]["at_tok"]:
            return None
        r = c.get("ramp")
        if r and self.t >= r["start"]:
            return max(r.get("floor", 0),
                       r["thr0"] - r.get("decay", 1) * ((self.t - r["start"]) // r.get("per", 1000)))
        cur = None
        for at, b, _ in c.get("fixed", []):
            if self.t >= at:
                cur = b
        return cur

    def _inj_for_bar(self):
        """Returns (inj_ids, is_ramp, gate_index). gate_index is None for the ramp."""
        c = self.cfg
        r = c.get("ramp")
        if r and self.t >= r["start"]:
            return r["inj_ids"], True, None
        cur, gi = None, None
        for i, (at, _, ids) in enumerate(c.get("fixed", [])):
            if self.t >= at:
                cur, gi = ids, i
        return cur, False, gi

    def observe(self, tok):
        """One emitted token. Returns the token id to FORCE next, or None to let the model sample."""
        self.t += 1
        if self.pending:
            # Advance unconditionally: one token was emitted for the one we forced, so the queue
            # moves on even if they differ. Popping only on a match stalls forever on any
            # divergence (another processor, a stop condition), re-forcing the same id every step.
            want = self.pending.popleft()
            if want != tok:
                self._log("force_miss", {"at_tok": self.t, "wanted": want, "got": tok})
        if tok == self.cfg["tag_id"]:
            self.disarmed = True
            self.pending.clear()
            return None
        if self.disarmed:
            return None
        at_nl = self.cfg["ends_nl"](tok)
        if self._fence_pending:
            if at_nl:
                self._set_fence(0)
            elif tok in self._ws_ids:
                pass
            else:
                self._set_fence(1)
        elif tok in self._fence_ids and self._line_start:
            if at_nl:
                self._set_fence(0)
            else:
                self._fence_pending = True
        self._line_start = at_nl or (self._line_start and tok in self._ws_ids)
        self.ns.push(tok)

        if self.pending:
            return self.pending[0]

        if self.armed_at is None:
            c = self.cfg
            if self.t >= c["final"]["at_tok"]:
                self.armed_at, self.armed_inj, self.armed_is_final = self.t, c["final"]["inj_ids"], True
                self._log("arm", {"at_tok": self.t, "nstar": self.ns.n, "bar": None,
                                  "kind": "final"})
            else:
                b = self.bar()
                inj, is_ramp, gi = self._inj_for_bar()
                if b is not None and inj and self.ns.n >= b:
                    if is_ramp and self.fires >= c.get("ramp", {}).get("max_fires", 1):
                        return None
                    if not is_ramp and gi in self.fixed_fired:
                        return None
                    self.armed_gate = gi
                    self.armed_at, self.armed_inj, self.armed_is_final = self.t, inj, False
                    self._log("arm", {"at_tok": self.t, "nstar": self.ns.n, "bar": b,
                                      "kind": "ramp" if is_ramp else "fixed"})
            if self.armed_at is None:
                return None

        assert self.armed_inj, "armed with no injection"
        waited = self.t - self.armed_at
        # The fence state is updated in observe() BEFORE this runs, so a closing ``` already reads 0
        # and is itself a legal boundary once its newline lands. A fence token whose opener/closer
        # verdict is still pending is never a boundary.
        outside = self.fence == 0 and not self._fence_pending
        forced_fence = False
        phase = "any"
        if self.armed_is_final:
            cap = self.cfg["final_wait"]
            want_safe = self.cfg.get("safe_final", True)
            boundary = at_nl and (outside or not want_safe)
            if boundary:
                self.pending.extend(self.armed_inj)
            elif waited >= cap:
                if want_safe and not outside:
                    self.pending.extend(self.cfg["fence_close_ids"])
                    forced_fence = True
                else:
                    self.pending.append(self.cfg["nl_id"])
                self.pending.extend(self.armed_inj)
            else:
                return None
            phase = "safe" if want_safe else "any"
        else:
            safe = self.cfg.get("safe_nudge", True)
            nudge_wait = int(self.cfg.get("nudge_wait", 1500))
            cap = self.cfg["wait_cap"]
            if safe and waited < nudge_wait:
                # phase 1: a line end OUTSIDE code, no forcing of any kind
                boundary = at_nl and outside
                if not boundary:
                    return None
                self.pending.extend(self.armed_inj)
                phase = "safe"
            else:
                # phase 2: unsafe nudge -- any line end, then a forced newline after wait_cap
                waited2 = waited - (nudge_wait if safe else 0)
                boundary = at_nl
                if boundary:
                    self.pending.extend(self.armed_inj)
                elif waited2 >= cap:
                    self.pending.append(self.cfg["nl_id"])
                    self.pending.extend(self.armed_inj)
                else:
                    return None
                phase = "any" if boundary else "forced"
        self._log("inject", {"at_tok": self.t, "nstar": self.ns.n, "waited": waited,
                             "forced_nl": bool(not boundary and not forced_fence),
                             "forced_fence_close": forced_fence, "fence_open": bool(self.fence),
                             "phase": phase,
                             "kind": "final" if self.armed_is_final else "nudge"})
        if not self.armed_is_final:
            self.fires += 1
            if self.armed_gate is not None:
                self.fixed_fired.add(self.armed_gate)
        self.armed_at = None
        self.armed_gate = None
        self.armed_inj = None
        return self.pending[0]


