"""Eventos locais por execução e sinais agregados do modelo."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


class _MemoryCounters(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('page_faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
            'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile',
        )
    ]


def resident_bytes():
    """Memória residente atual do processo em bytes, quando disponível."""
    try:
        if os.name == 'nt':
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            current_process = kernel.GetCurrentProcess
            current_process.argtypes = []
            current_process.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            read_memory = psapi.GetProcessMemoryInfo
            read_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
            read_memory.restype = wintypes.BOOL
            counters = _MemoryCounters()
            counters.size = ctypes.sizeof(counters)
            if read_memory(current_process(), ctypes.byref(counters), counters.size):
                return int(counters.working_set)
            return None
        stat = Path('/proc/self/statm').read_text(encoding='ascii')
        return int(stat.split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, AttributeError, ValueError, IndexError):
        return None


def prediction_summary(scores, threshold):
    """Histograma em dez faixas fixas e fração de predições positivas."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Probabilidades precisam ser um vetor válido e não vazio')
    counts, edges = np.histogram(scores, bins=np.arange(11) / 10)
    positives = int((scores >= threshold).sum())
    return {
        'threshold': float(threshold),
        'positive_count': positives,
        'positive_rate': positives / len(scores),
        'probability_histogram': {'edges': edges.tolist(), 'counts': counts.tolist()},
    }


class RunLog:
    """Um arquivo JSONL por execução. Somente campos agregados entram nos eventos."""

    def __init__(self, directory, run_id, scenario):
        self.path = Path(directory) / 'events.jsonl'
        self.path.touch(exist_ok=False)
        self.run_id = run_id
        self.scenario = scenario
        self.current_stage = None
        self.alert_count = 0

    def _emit(self, event):
        entry = {'run_id': self.run_id, 'scenario': self.scenario,
                 'timestamp_utc': datetime.now(UTC).isoformat(), **event}
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(entry, allow_nan=False, sort_keys=True) + '\n')

    @contextmanager
    def stage(self, name, rows=None):
        self.current_stage = name
        start_wall, start_cpu = perf_counter(), process_time()
        event = {'event': 'stage', 'stage': name, 'status': 'ok', 'rows': rows}
        try:
            yield event
        except Exception:
            event['status'] = 'failed'
            raise
        finally:
            cpu = process_time() - start_cpu
            wall = perf_counter() - start_wall
            event.update(duration_seconds=wall, cpu_seconds=cpu,
                         cpu_percent_one_core=100 * cpu / wall if wall > 0 else None,
                         rss_bytes=resident_bytes())
            self._emit(event)

    def alert(self, *, kind, severity, action, rule=None, feature=None,
              value=None, threshold=None, status='open'):
        self._emit({'event': 'alert', 'stage': self.current_stage, 'type': kind,
                    'severity': severity, 'status': status, 'rule': rule,
                    'feature': feature, 'value': value, 'threshold': threshold,
                    'action': action})
        self.alert_count += 1
