"""Reproducible example for Section 5.8.

Compares four execution strategies for deadline-constrained GPU tasks on
revocable shared GPUs under identical assumptions:

  P   placement only: one copy; on reclaim the task restarts from scratch on a
      free eligible GPU (classical restart, no feasibility check)
  R   replication: copies chosen with the covering constraint (14) using the
      lower bound (11) on an *estimated* reclaim rate, at most one copy per
      failure domain (15), at most K_MAX copies
  C   checkpoint recovery using only the cost/capacity constraints (16)-(19):
      checkpoints on the provider's local disk; on reclaim the task migrates to
      any eligible free GPU; the checkpoint survives only if it can be
      transferred within the reclaim notice, otherwise the task restarts
  Cr  as C, but checkpoints are also written to storage outside the provider
      (isolates the effect of a recoverable state, r = 1 in (19a))
  C+  checkpoint recovery with the deadline-aware conditions (19a)-(19b):
      checkpoints are also written to storage outside the provider; a task is
      migrated only if the recovery can finish before the deadline, otherwise it
      is abandoned immediately and its capacity is released

Discrete-event simulation; all random draws use numpy Generators seeded per run.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    horizon_h: float = 48.0          # simulated hours of task arrivals
    n_domains: int = 6               # laboratories (failure domains)
    gpus_per_domain: int = 4
    speeds: tuple = (1.0, 1.5, 2.0)  # relative speeds, cycled over GPUs
    bigmem_frac: float = 2 / 3       # fraction of GPUs eligible for large-memory tasks
    arrival_per_h: float = 5.0       # external task arrival rate (light load)
    work_min: tuple = (20.0, 60.0)   # work in minutes on a speed-1.0 GPU
    slack: tuple = (1.6, 3.0)        # deadline = arrival + slack * work / 1.5
    state_gb: tuple = (5.0, 40.0)    # checkpoint (model + optimizer) size
    bigmem_task_frac: float = 0.5
    bw_gbps: float = 1.0             # inter-provider bandwidth (Gbit/s)
    restore_fixed_min: float = 0.5   # container start + attestation
    ckpt_interval_min: float = 10.0
    ckpt_overhead: float = 0.03      # slowdown from periodic checkpointing
    reclaim_per_h: float = 1.0       # total reclaim intensity per GPU (per hour)
    corr: float = 0.25               # share of reclaim intensity that is domain-wide
    notice_min: float = 2.0          # reclaim notice granted by the provider
    local_busy_min: float = 20.0     # mean duration of the provider's local job
    local_tol_min: tuple = (0.5, 5.0)  # tolerated start delay of a local job
    est_err: float = 0.0             # relative error of the scheduler's reclaim-rate estimate
    margin: float = 0.0              # safety margin: inflate estimated rate by (1+margin)
    eps: float = 0.1                 # tolerated failure probability in (14)
    k_max: int = 4
    kappa: int | None = 1            # replicas per failure domain in (15); None = no limit
    power_kw: float = 0.3            # GPU power while running external work


@dataclass
class Task:
    tid: int
    arrival: float
    work: float          # minutes at speed 1.0
    deadline: float      # absolute minutes
    state_gb: float
    bigmem: bool
    done: float = 0.0    # progress preserved (min @ speed 1) - for checkpoint strategies
    finished: bool = False
    lost: bool = False
    copies: dict = field(default_factory=dict)  # gpu -> (start_time, start_progress)
    last_ckpt_progress: float = 0.0


class Sim:
    def __init__(self, p: Params, strategy: str, seed: int):
        self.p, self.strategy = p, strategy
        self.rng = np.random.default_rng(seed)
        n = p.n_domains * p.gpus_per_domain
        self.n = n
        self.domain = np.repeat(np.arange(p.n_domains), p.gpus_per_domain)
        self.speed = np.array([p.speeds[i % len(p.speeds)] for i in range(n)])
        bigmem = np.zeros(n, bool)
        bigmem[: int(round(p.bigmem_frac * n))] = True
        self.rng.shuffle(bigmem)
        self.bigmem = bigmem
        self.busy_by = [None] * n       # tid of external task occupying the GPU
        self.local_until = np.zeros(n)  # GPU unavailable (local work) until
        self.events: list = []
        self.seq = 0
        self.tasks: list[Task] = []
        self.waiting: list[int] = []
        # metrics
        self.alloc_gpu_min = 0.0   # wall-clock GPU minutes held by external work
        self.alloc_work = 0.0      # the same, weighted by GPU speed
        self.useful_work = 0.0     # work of tasks completed on time
        self.migrations = 0
        self.local_jobs = 0
        self.local_miss = 0
        self.transfer_gb = 0.0

    # ---------- event helpers ----------
    def push(self, t, kind, *data):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, data))

    def bw_gb_per_min(self):
        return self.p.bw_gbps / 8.0 * 60.0

    # ---------- workload generation ----------
    def generate(self):
        p, rng = self.p, self.rng
        t, tid = 0.0, 0
        H = p.horizon_h * 60
        while True:
            t += rng.exponential(60.0 / p.arrival_per_h)
            if t > H:
                break
            w = rng.uniform(*p.work_min)
            sl = rng.uniform(*p.slack)
            task = Task(tid, t, w, t + sl * w / 1.5, rng.uniform(*p.state_gb),
                        rng.random() < p.bigmem_task_frac)
            self.tasks.append(task)
            self.push(t, "arrive", tid)
            tid += 1
        # reclaim processes: independent per GPU and domain-wide
        lam_ind = p.reclaim_per_h * (1 - p.corr) / 60.0
        lam_dom = p.reclaim_per_h * p.corr / 60.0
        end = H + 24 * 60
        for j in range(self.n):
            tt = 0.0
            while lam_ind > 0:
                tt += rng.exponential(1 / lam_ind)
                if tt > end:
                    break
                self.push(tt, "reclaim", (j,))
        for d in range(p.n_domains):
            tt = 0.0
            while lam_dom > 0:
                tt += rng.exponential(1 / lam_dom)
                if tt > end:
                    break
                self.push(tt, "reclaim", tuple(np.where(self.domain == d)[0]))

    # ---------- placement ----------
    def eligible_free(self, task, now):
        ok = []
        for j in range(self.n):
            if self.busy_by[j] is not None or self.local_until[j] > now:
                continue
            if task.bigmem and not self.bigmem[j]:
                continue
            ok.append(j)
        return ok

    def exec_time(self, task, j, progress):
        slow = 1 + (self.p.ckpt_overhead if self.strategy in ("C", "Cr", "C+") else 0.0)
        return (task.work - progress) / self.speed[j] * slow

    def survival_lb(self, j, horizon_min):
        """a_j(D) under the scheduler's (possibly wrong) rate estimate, eq. (11)."""
        lam = self.p.reclaim_per_h / 60.0 * (1 + self.p.est_err) * (1 + self.p.margin)
        return math.exp(-lam * max(horizon_min, 0.0))

    def try_start(self, tid, now, recovery=False):
        task = self.tasks[tid]
        if task.finished or task.lost:
            return True
        if now >= task.deadline:
            task.lost = True
            return True
        cands = self.eligible_free(task, now)
        prep = self.p.restore_fixed_min + (
            task.state_gb / self.bw_gb_per_min() if task.done > 0 else 0.0)
        if recovery and self.strategy in ("P", "C", "Cr"):
            # restart / cost-only recovery (16)-(19): any eligible survivor with capacity
            feas = list(cands)
        else:
            feas = [j for j in cands
                    if now + prep + self.exec_time(task, j, task.done) <= task.deadline]
        if not feas:
            if not cands:
                return False  # keep waiting for a GPU
            if recovery and self.strategy in ("P", "C", "Cr"):
                return False
            # free GPUs exist but none can meet the deadline any more
            if all(now + prep + self.exec_time(task, j, task.done) > task.deadline
                   for j in range(self.n) if not (task.bigmem and not self.bigmem[j])):
                task.lost = True
                return True
            return False
        feas.sort(key=lambda j: now + prep + self.exec_time(task, j, task.done))
        if self.strategy == "R":
            chosen, weight, per_dom = [], 0.0, {}
            target = math.log(1 / self.p.eps)
            for j in feas:
                d = self.domain[j]
                if self.p.kappa is not None and per_dom.get(d, 0) >= self.p.kappa:
                    continue  # failure-domain limit (15)
                s_lb = self.survival_lb(j, task.deadline - now)
                chosen.append(j)
                per_dom[d] = per_dom.get(d, 0) + 1
                weight += math.log(1 / max(1 - s_lb, 1e-12))
                if weight >= target or len(chosen) >= self.p.k_max:
                    break
            if not task.copies and not hasattr(task, "pred"):
                task.pred = 1 - math.exp(-weight)  # success predicted by (13) under independence
        else:
            chosen = [feas[0]]
        for j in chosen:
            self.busy_by[j] = tid
            task.copies[j] = (now, now + prep, task.done)
            self.push(now + prep + self.exec_time(task, j, task.done), "finish", tid, j, now)
        return True

    def schedule_waiting(self, now):
        still = []
        for tid in self.waiting:
            if not self.try_start(tid, now, recovery=getattr(self.tasks[tid], "_recovering", False)):
                still.append(tid)
        self.waiting = still

    def release(self, j, now, task):
        placed, start, prog0 = task.copies.pop(j)
        held = max(now - placed, 0.0)
        self.alloc_gpu_min += held
        self.alloc_work += held * self.speed[j]
        self.busy_by[j] = None
        return held

    # ---------- main loop ----------
    def run(self):
        self.generate()
        while self.events:
            now, _, kind, data = heapq.heappop(self.events)
            if kind == "arrive":
                (tid,) = data
                if not self.try_start(tid, now):
                    self.waiting.append(tid)
            elif kind == "finish":
                tid, j, placed = data
                task = self.tasks[tid]
                if task.finished or task.lost or j not in task.copies:
                    continue
                if task.copies[j][0] != placed:
                    continue  # stale event from an earlier copy on this GPU
                self.release(j, now, task)
                task.finished = now <= task.deadline + 1e-9
                if not task.finished:
                    task.lost = True
                else:
                    self.useful_work += task.work
                for k in list(task.copies):  # cancel the other replicas
                    self.release(k, now, task)
                self.schedule_waiting(now)
            elif kind == "reclaim":
                gpus = data[0]
                for j in gpus:
                    self.reclaim(j, now)
                self.schedule_waiting(now)
            elif kind == "local_done":
                self.schedule_waiting(now)
        return self.metrics()

    def reclaim(self, j, now):
        p, rng = self.p, self.rng
        busy = rng.exponential(p.local_busy_min)
        self.local_jobs += 1
        tol = rng.uniform(*p.local_tol_min)
        tid = self.busy_by[j]
        delay = 0.0
        if tid is not None:
            task = self.tasks[tid]
            placed, start, prog0 = task.copies[j]
            elapsed = max(now - start, 0.0)
            slow = 1 + (p.ckpt_overhead if self.strategy in ("C", "Cr", "C+") else 0.0)
            progress = min(prog0 + elapsed * self.speed[j] / slow, task.work)
            if self.strategy in ("C", "Cr", "C+"):
                # write a final checkpoint during the notice period
                write_t = task.state_gb / self.bw_gb_per_min()
                delay = min(p.notice_min, write_t)
                if self.strategy in ("Cr", "C+"):
                    # periodic checkpoints already outside the provider; final one if it fits
                    periodic = math.floor(progress / p.ckpt_interval_min) * p.ckpt_interval_min
                    task.done = progress if write_t <= p.notice_min else max(periodic, prog0)
                    self.transfer_gb += task.state_gb
                else:
                    # local checkpoints only: survive only if transferable within the notice
                    task.done = progress if write_t <= p.notice_min else 0.0
            else:
                task.done = 0.0
            self.release(j, now, task)
            if not task.copies and not task.finished and not task.lost:
                self.recover(task, now + delay)
        if delay > tol:
            self.local_miss += 1
        self.local_until[j] = max(self.local_until[j], now + delay + busy)
        self.push(now + delay + busy, "local_done")

    def recover(self, task, now):
        if self.strategy == "C+":
            # deadline-aware recovery feasibility (19a)-(19b): check before migrating
            prep = self.p.restore_fixed_min + task.state_gb / self.bw_gb_per_min()
            fastest = max(self.speed[j] for j in range(self.n)
                          if not (task.bigmem and not self.bigmem[j]))
            if now + prep + (task.work - task.done) / fastest * (1 + self.p.ckpt_overhead) > task.deadline:
                task.lost = True
                return
        if task.done > 0:
            self.migrations += 1
        task._recovering = True
        if not self.try_start(task.tid, now, recovery=True):
            self.waiting.append(task.tid)

    def metrics(self):
        n = len(self.tasks)
        fin = sum(t.finished for t in self.tasks)
        return {
            "dsr": fin / n,
            "useful_ratio": self.useful_work / max(self.alloc_work, 1e-9),
            "energy_kwh_per_done": self.alloc_gpu_min / 60 * self.p.power_kw / max(fin, 1),
            "local_miss": self.local_miss / max(self.local_jobs, 1),
            "gpu_h": self.alloc_gpu_min / 60,
            "migrations_per_task": self.migrations / n,
            "pred_minus_real": (float(np.mean([t.pred - t.finished for t in self.tasks
                                               if hasattr(t, "pred")]))
                                if any(hasattr(t, "pred") for t in self.tasks) else 0.0),
        }


STRATEGIES = ("P", "R", "C", "Cr", "C+")


def run_many(p: Params, strategy: str, seeds=range(30)):
    rows = [Sim(p, strategy, s).run() for s in seeds]
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = (v.mean(), 1.96 * v.std(ddof=1) / math.sqrt(len(v)))
    return out
