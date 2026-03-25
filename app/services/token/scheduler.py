"""Token 刷新与自动删除调度器"""

import asyncio
import time
from typing import Optional

from app.core.logger import logger
from app.core.storage import get_storage, StorageError, RedisStorage
from app.services.token.manager import get_token_manager


class TokenRefreshScheduler:
    """Token 自动维护调度器"""

    def __init__(
        self,
        interval_hours: int = 8,
        refresh_enabled: bool = True,
        tick_seconds: int = 60,
    ):
        self.interval_hours = interval_hours
        self.interval_seconds = max(60, int(interval_hours * 3600))
        self.refresh_enabled = refresh_enabled
        self.tick_seconds = max(30, int(tick_seconds))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._next_refresh_at = 0.0

    def configure(
        self,
        interval_hours: Optional[int] = None,
        refresh_enabled: Optional[bool] = None,
    ):
        if interval_hours is not None:
            self.interval_hours = interval_hours
            self.interval_seconds = max(60, int(interval_hours * 3600))
        if refresh_enabled is not None:
            self.refresh_enabled = refresh_enabled

    async def _refresh_loop(self):
        """刷新循环"""
        logger.info(
            "Scheduler: started "
            f"(refresh_enabled={self.refresh_enabled}, "
            f"refresh_interval={self.interval_hours}h, tick={self.tick_seconds}s)"
        )

        while self._running:
            try:
                storage = get_storage()
                lock_acquired = False
                lock = None
                lock_timeout = max(self.tick_seconds + 30, 60)

                if isinstance(storage, RedisStorage):
                    # Redis: non-blocking lock to avoid multi-worker duplication
                    lock_key = "grok2api:lock:token_refresh"
                    lock = storage.redis.lock(
                        lock_key, timeout=lock_timeout, blocking_timeout=0
                    )
                    lock_acquired = await lock.acquire(blocking=False)
                else:
                    try:
                        async with storage.acquire_lock("token_refresh", timeout=1):
                            lock_acquired = True
                    except StorageError:
                        lock_acquired = False

                if not lock_acquired:
                    await asyncio.sleep(self.tick_seconds)
                    continue

                try:
                    manager = await get_token_manager()
                    cleanup = await manager.cleanup_auto_delete_tokens()
                    if cleanup["deleted"] > 0:
                        logger.info(
                            "Scheduler: auto delete completed - "
                            f"checked={cleanup['checked']}, "
                            f"deleted={cleanup['deleted']}"
                        )

                    should_refresh = self.refresh_enabled and (
                        self._next_refresh_at == 0.0
                        or time.monotonic() >= self._next_refresh_at
                    )
                    if should_refresh:
                        logger.info("Scheduler: starting token refresh...")
                        result = await manager.refresh_cooling_tokens()
                        self._next_refresh_at = time.monotonic() + self.interval_seconds

                        logger.info(
                            f"Scheduler: refresh completed - "
                            f"checked={result['checked']}, "
                            f"refreshed={result['refreshed']}, "
                            f"recovered={result['recovered']}, "
                            f"expired={result['expired']}"
                        )
                finally:
                    if lock is not None and lock_acquired:
                        try:
                            await lock.release()
                        except Exception:
                            pass

                await asyncio.sleep(self.tick_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler: refresh error - {e}")
                await asyncio.sleep(self.tick_seconds)

    def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("Scheduler: already running")
            return

        self._running = True
        self._next_refresh_at = 0.0
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info("Scheduler: enabled")

    def stop(self):
        """停止调度器"""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Scheduler: stopped")


# 全局单例
_scheduler: Optional[TokenRefreshScheduler] = None


def get_scheduler(
    interval_hours: int = 8, refresh_enabled: Optional[bool] = None
) -> TokenRefreshScheduler:
    """获取调度器单例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TokenRefreshScheduler(
            interval_hours, refresh_enabled=refresh_enabled if refresh_enabled is not None else True
        )
    else:
        _scheduler.configure(
            interval_hours=interval_hours, refresh_enabled=refresh_enabled
        )
    return _scheduler


__all__ = ["TokenRefreshScheduler", "get_scheduler"]
