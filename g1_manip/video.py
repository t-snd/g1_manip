"""斜め上の視点のカメラと、その視点の mp4 の書き出し（eval・gui で共通）。"""

from pathlib import Path

import mujoco

from g1_manip import config


def overview_camera(model) -> mujoco.MjvCamera:
    """机の向こう側（ロボットの左前）の斜め上から、ロボットと作業領域を見るカメラ（VIDEO_*）。"""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = config.VIDEO_LOOKAT
    cam.distance = config.VIDEO_DISTANCE
    cam.azimuth = config.VIDEO_AZIMUTH
    cam.elevation = config.VIDEO_ELEVATION
    return cam


class VideoWriter:
    """mujoco.Renderer で描いた画像を mp4（H.264）に書き出す。"""

    def __init__(self, path: Path, model):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=config.VIDEO_HEIGHT, width=config.VIDEO_WIDTH)
        self.camera = overview_camera(model)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=config.FPS)
        self.stream.width, self.stream.height = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self._frame_cls = av.VideoFrame

    def add(self, data) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        frame = self._frame_cls.from_ndarray(self.renderer.render(), format="rgb24")
        self.container.mux(self.stream.encode(frame))

    def close(self) -> None:
        self.container.mux(self.stream.encode())
        self.container.close()
        self.renderer.close()
