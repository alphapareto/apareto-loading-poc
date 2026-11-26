"""Host-RAM safety for servers that pin a host tier.

Why: the bank engine pins the whole layer store (18.16 GB measured) in host RAM, which the kernel can neither
reclaim nor swap. On a 31.7 GB machine this can push the user session into heavy memory pressure, and
systemd-oomd may then kill the whole terminal scope the server was launched from. So a server launch:
- refuses to start unless MemAvailable covers the pinned tier plus a reserve for the engine itself;
- runs in its own systemd user scope with a hard MemoryMax and no swap, so running out of memory kills
  only the server, never the shell or agent that launched it;
- is stopped by the harness if MemAvailable falls below a floor while it runs.
"""
import os
from pathlib import Path
import subprocess

GIB = 1 << 30


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


class HostMemoryGuard:
    def __init__(self, unit, pinned_bytes, reserve_bytes=8 * GIB, floor_bytes=2 * GIB, headroom_bytes=3 * GIB):
        self.unit, self.floor = unit, floor_bytes
        available = mem_available()
        if available < pinned_bytes + reserve_bytes:
            raise RuntimeError(f'Host RAM: MemAvailable {available / GIB:.1f} GiB < pinned tier '
                               f'{pinned_bytes / GIB:.1f} GiB + reserve {reserve_bytes / GIB:.1f} GiB')
        # Everything above the cap stays with the desktop, the launching shell and the agent.
        self.limit = available - headroom_bytes
        self.record = {'mem_available_at_launch_bytes': available, 'scope_memory_max_bytes': self.limit,
                       'pinned_tier_bytes': pinned_bytes, 'min_mem_available_bytes': available,
                       'peak_scope_memory_bytes': 0}

    def wrap(self, command):
        return ['systemd-run', '--user', '--scope', '--quiet', '--unit', self.unit,
                '-p', f'MemoryMax={self.limit}', '-p', 'MemorySwapMax=0', *command]

    def _scope_current(self):
        path = Path(f'/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{self.unit}.scope/memory.current')
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            return 0

    def stop(self):
        """Stop everything in the server's scope. A SIGTERM to the process group is not always honoured
        within the harness's wait; systemd stops the scope fully."""
        subprocess.run(['systemctl', '--user', 'stop', f'{self.unit}.scope'], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def check(self):
        available = mem_available()
        self.record['min_mem_available_bytes'] = min(self.record['min_mem_available_bytes'], available)
        self.record['peak_scope_memory_bytes'] = max(self.record['peak_scope_memory_bytes'], self._scope_current())
        if available < self.floor:
            raise RuntimeError(f'Host RAM: MemAvailable {available / GIB:.2f} GiB below the {self.floor / GIB:.1f} GiB floor')
