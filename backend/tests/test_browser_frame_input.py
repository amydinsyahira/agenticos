"""Manual input follows actual JPEG density and preserves command order."""

import asyncio
from base64 import b64encode
from collections import deque
from io import BytesIO
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from app.agents.browser_events import BrowserCommand, BrowserEvent
from app.agents.capabilities.browser_use._loop import BrowseSession, held
from app.agents.capabilities.browser_use._page import CdpPage, PagePolicy

pytestmark = pytest.mark.anyio


def jpeg(width: int, height: int) -> str:
    with BytesIO() as output:
        Image.new("RGB", (width, height)).save(output, format="JPEG")
        return b64encode(output.getvalue()).decode()


class Client:
    def __init__(self) -> None:
        self.send = self
        self.register = self
        self.Input = self
        self.Emulation = self
        self.Page = self
        self.dispatchMouseEvent = AsyncMock(return_value={})
        self.setDeviceMetricsOverride = AsyncMock(return_value={})
        self.captureScreenshot = AsyncMock(return_value={})
        self.screencastFrame = Mock()
        self.startScreencast = AsyncMock(return_value={})
        self.stopScreencast = AsyncMock(return_value={})
        self.screencastFrameAck = AsyncMock(return_value={})


@pytest.mark.parametrize("density", [0.5, 1, 2])
async def test_retina_clicks_land_at_the_visible_button_for_each_frame_density(
    density: float,
) -> None:
    client = Client()
    page = CdpPage(
        client, "tab", PagePolicy(allowed_domains=None, candidate_cap=60, preview_width=1000)
    )
    await page.resize(1000, 600, 2)
    client.captureScreenshot.return_value = {
        "data": jpeg(round(1000 * density), round(600 * density))
    }
    assert await page.screenshot()
    await page.click_at(800 * density, 420 * density)
    events = [call.kwargs["params"] for call in client.dispatchMouseEvent.await_args_list]
    assert [(event["type"], event["x"], event["y"]) for event in events] == [
        ("mousePressed", 800, 420),
        ("mouseReleased", 800, 420),
    ]


async def test_switching_from_screencast_to_screenshot_keeps_the_same_click_target() -> None:
    client = Client()
    page = CdpPage(
        client, "tab", PagePolicy(allowed_domains=None, candidate_cap=60, preview_width=1000)
    )
    await page.resize(1000, 600, 2)
    painted = asyncio.Event()

    async def deliver(image: str) -> None:
        assert image.startswith("data:image/jpeg;base64,")
        painted.set()

    async with page.stream(deliver):
        callback = client.screencastFrame.call_args.args[0]
        callback({"sessionId": 1, "data": jpeg(1000, 600)}, "other-tab")
        callback({"sessionId": 2, "data": jpeg(1000, 600)}, "tab")
        await asyncio.wait_for(painted.wait(), timeout=1)
        await page.click_at(800, 420)
    client.captureScreenshot.return_value = {"data": jpeg(2000, 1200)}
    await page.screenshot()
    await page.click_at(1600, 840)
    assert [
        (call.kwargs["params"]["x"], call.kwargs["params"]["y"])
        for call in client.dispatchMouseEvent.await_args_list
    ] == [(800, 420)] * 4


class Control:
    def __init__(self, commands: list[BrowserCommand]) -> None:
        self.commands = deque(commands)

    async def poll(self) -> BrowserCommand | None:
        return self.commands.popleft() if self.commands else None

    async def wait(self, timeout: float) -> BrowserCommand | None:
        result = await self.poll()
        if result is None:
            await asyncio.sleep(timeout)
        return result


class ManualPage:
    streaming = False

    def __init__(self) -> None:
        self.actions: list[tuple[str, float, float]] = []

    async def location(self) -> tuple[str, str]:
        return "https://example.com/", "Example"

    async def screenshot(self) -> None:
        return None

    async def wheel(self, dy: float) -> None:
        self.actions.append(("wheel", dy, 0))

    async def click_at(self, x: float, y: float) -> None:
        self.actions.append(("click", x, y))


def command(
    kind: Literal["scroll", "give_back"], *, dy: float | None = None, owner: str = "holder"
) -> BrowserCommand:
    return BrowserCommand(type=kind, dy=dy, by_user_id=owner)


async def ignore_frames(event: BrowserEvent) -> bool:
    return False


@pytest.mark.security
async def test_trackpad_bursts_preserve_click_order_and_reject_other_viewers() -> None:
    page = ManualPage()
    control = Control(
        [
            command("scroll", dy=20),
            command("scroll", dy=900, owner="other"),
            command("give_back", owner="other"),
            command("scroll", dy=30),
            command("scroll", dy=40),
            BrowserCommand(type="click", x=800, y=420, by_user_id="holder"),
            command("scroll", dy=-10),
            command("give_back"),
        ]
    )
    await held(
        BrowseSession(page=page),
        control,
        ignore_frames,
        call_id="test",
        by="Holder",
        by_user_id="holder",
        remember=None,
    )
    assert page.actions == [("wheel", 90, 0), ("click", 800, 420), ("wheel", -10, 0)]


async def test_large_scroll_bursts_stay_bounded_and_hand_back_still_finishes() -> None:
    page = ManualPage()
    control = Control([command("scroll", dy=1) for _ in range(40)] + [command("give_back")])
    await held(
        BrowseSession(page=page),
        control,
        ignore_frames,
        call_id="test",
        by="Holder",
        by_user_id="holder",
        remember=None,
    )
    assert page.actions == [("wheel", 32, 0), ("wheel", 8, 0)]
