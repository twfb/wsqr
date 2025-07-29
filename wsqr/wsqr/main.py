#!python3
# -*- coding: utf-8 -*-
from hashlib import md5
import os
import time
import cv2
import shutil
import tempfile
import pathlib
import warnings
import typer
import numpy as np
import multiprocessing
import threading
import psutil
import requests

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.prompt import Confirm
from datetime import datetime

from typing import (
    List,
    Dict,
    Callable,
    Iterable,
    Annotated,
)
from functools import partial

warnings.simplefilter("ignore")

# 初始化全局组件
console = Console()
app = typer.Typer(
    help="[bold magenta]🚀 哇塞二维码 - 终极二维码解码利器[/]",
    rich_markup_mode="rich",
    add_completion=False,
)
MAX_QREADER_COUNT = psutil.virtual_memory().available // (1024 * 1024 * 1024 * 2) - 1
DEBUG = False


def log(string, style=None):
    global DEBUG
    if DEBUG:
        console.log(string, style=style)
    else:
        console.print(
            f'[dim cyan][{datetime.now().strftime("%H:%M:%S")}][/]', string, style=style
        )


class Image:
    """图像封装类，支持延迟加载和资源释放"""

    def __init__(self, shape, path, flag=cv2.IMREAD_COLOR):
        self.shape = shape  # 图像尺寸
        self.path = path  # 图像路径
        self.flag = flag  # 读取标志
        self._data = None  # 图像数据（延迟加载）
        if not path:
            raise ValueError("缺少图像路径")

    def __str__(self):
        return str(self.path.absolute())

    @property
    def data(self):
        """延迟加载图像数据"""
        if self._data is None:
            self._data = cv2.imread(str(self.path), self.flag)
            if self.shape is None:
                self.shape = self._data.shape
        return self._data

    def release(self):
        """释放图像内存"""
        self._data = None


class DecodeResult:
    """解码结果容器类"""

    def __init__(self, bytes=None, text=None, engine=None, shape=None, path=None):
        self.bytes = bytes  # 原始字节数据
        self.text = text  # 解码文本
        self.engine = engine  # 使用的解码引擎
        self.shape = shape  # 图像尺寸
        self.path = path  # 图像路径


class Decoder:
    """解码器基类"""

    name = "base_decoder"  # 解码器名称
    pip_name = None  # pip安装包名
    rank = 0  # 解码器优先级

    def __str__(self):
        return self.name

    def decode(self, img) -> List[DecodeResult]:
        """解码入口方法"""
        result = []
        # 调用具体解码实现
        for bytes, text in self._decode(img) or []:
            # 记录解码日志
            log(
                f"[cyan]{self.name}[/]从[bright_white] {img.path} [/]解析到: "
                f"[yellow on black]{text}[/]"
            )
            # 封装解码结果
            result.append(
                DecodeResult(
                    bytes=bytes,
                    text=text,
                    engine=self.name,
                    shape=img.shape,
                    path=img.path,
                )
            )
        return result

    def _decode(self, img: Image) -> List[DecodeResult]:
        """具体解码实现（子类必须重写）"""
        raise NotImplementedError

    @classmethod
    def _test(cls):
        """测试解码器是否可用（子类必须重写）"""
        return True

    @classmethod
    def test(cls):
        """测试解码器可用性并提供安装指引"""
        result = cls._test()
        if result == True:
            return None
        elif result == False:
            time.sleep(3)
            # 用户交互：是否继续
            break_now = (
                console.input(
                    f"\n\n未安装 [red]{cls.name}[/], 安装命令: [yellow]pip install {cls.pip_name}[/], 是否继续[Y/n]"
                )
                .lower()
                .startswith("n")
            )
            if break_now:
                return "break"  # 终止程序
            else:
                return "continue"  # 跳过当前解码器
        else:
            return result  # 自定义处理结果


class OpenCVDecoder(Decoder):
    """OpenCV二维码解码器"""

    name = "OpenCV"
    pip_name = "opencv-python"
    rank = 3  # 优先级（数值越小优先级越高）

    @classmethod
    def _test(cls):
        try:
            import cv2

            return True
        except ModuleNotFoundError:
            return False

    def _decode(self, img: Image) -> List[DecodeResult]:
        """使用OpenCV进行多二维码检测和解码"""
        return [
            [i.encode(), i]
            for i in cv2.QRCodeDetector().detectAndDecodeMulti(img.data)[1]
            if i
        ]


class PyzbarDecoder(Decoder):
    """Pyzbar解码器（基于ZBar库）"""

    name = "pyzbar"
    pip_name = "pyzbar"
    rank = 2

    @classmethod
    def _test(cls):
        try:
            import pyzbar

            return True
        except ModuleNotFoundError:
            return False

    def _decode(self, img: Image) -> List[DecodeResult]:
        """使用Pyzbar解码"""
        from pyzbar.pyzbar import decode

        return [
            [result.data, result.data.decode().encode("latin-1").decode("utf-8")]
            for result in decode(image=img.data)
        ]


class ZxingCppDecoder(Decoder):
    """ZXing C++ 解码器（高性能）"""

    name = "ZxingCpp"
    pip_name = "zxingcpp"
    rank = 1

    @classmethod
    def _test(cls):
        try:
            import zxingcpp

            return True
        except ModuleNotFoundError:
            return False

    def _decode(self, img: Image) -> List[DecodeResult]:
        import zxingcpp

        return [
            [result.bytes, result.text] for result in zxingcpp.read_barcodes(img.data)
        ]


class QReaderDecoder(Decoder):
    """基于深度学习的QReader解码器"""

    name = "QReader"
    rank = 999  # 最低优先级（资源消耗大）
    pip_name = "qreader"

    @classmethod
    def _test(cls):
        """特殊处理：检查并下载所需模型文件"""
        try:
            from qreader import QReader
            import qrdet
        except ModuleNotFoundError:
            return False
        model_dir = pathlib.Path(qrdet.__file__).parent / ".model"
        model_dir.mkdir(exist_ok=True)
        src_model_file = model_dir / "qrdet-l.pt"
        release_file = model_dir / "current_release.txt"
        if release_file.exists():
            url = open(release_file).read().strip() + "/qrdet-l.pt"
        else:
            url = "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release"
            open(release_file, "w").write(url)
            url += "/qrdet-l.pt"
        try:
            log(f"[dim]测试QReader模型文件[/]")
            QReader("l", 0)
            return True
        except Exception as e:
            log(f"[dim]QReader模型文件不存在或损坏")
            # raise e
        except ConnectionError:
            pass

        log(f"[dim]正在下载QReader模型文件: {url} 到 {src_model_file}[/]")
        try:
            with requests.get(url, stream=True, timeout=60, verify=False) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with Progress(
                    TextColumn(f"[bold blue]正在下载"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    "•",
                    DownloadColumn(),
                    "•",
                    TransferSpeedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task("downloading", total=total_size)
                    with open(src_model_file, "wb") as f:
                        downloaded = 0
                        for chunk in r.iter_content(chunk_size=1024):
                            downloaded += len(chunk)
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                            progress.start()
            log(f"[dim]QReader模型文件下载完成: {src_model_file}[/]")
            return True

        except ConnectionError:
            pass
        if Confirm.ask("模型文件未下载, 是否继续", default=True):
            return "continue"  # 跳过当前解码器
        else:
            return "break"  # 终止程序

    def _decode(self, img: Image) -> List[DecodeResult]:
        """使用QReader解码（基于深度学习）"""
        from qreader import QReader

        # 调整大尺寸图像
        img_data = img.data
        while img_data.shape[:2] > (1024, 1024):

            img_data = cv2.resize(
                src=img_data,
                dsize=None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_CUBIC,
            )

        img.shape = img_data.shape
        # 创建解码器实例
        readers = [QReader("l", 0)]
        results = []
        for reader in readers:
            detections = reader.detect(img_data)
            for detection in detections:
                content = reader.decode(img_data, detection)
                if content:
                    results.append(
                        [
                            content.encode("utf-8"),
                            content.encode("latin-1").decode("utf-8"),
                        ]
                    )
                    break
        return results


class ImageProcessor:
    """图像预处理管道"""

    MIN_QR_SIZE = 21  # 最小二维码尺寸（模块数）
    base_image = None
    origin_images = []
    variants = {}
    name = None
    path = None
    temp_dir = None

    def __init__(self, share_manager, input_path, temp_dir):
        self.path = input_path
        self.share_manager = share_manager
        self.temp_dir = temp_dir / input_path.name
        if self.temp_dir.exists():
            self.temp_dir = self.temp_dir / (int(time.time()))
        os.makedirs(self.temp_dir)
        self.set_origin_images()  # 初始化原始图像

    def is_similar(self, img1, img2, min_diff_percent=0.5):
        """判断两帧图像是否相似（用于GIF去重）"""
        if (not isinstance(img1, np.ndarray)) or (not isinstance(img2, np.ndarray)):
            return False
        min_diff = np.prod(img1.shape[:2]) * min_diff_percent / 100
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        diff_pixels = np.sum(diff > 25)  # 25为灰度差异阈值
        return diff_pixels <= min_diff

    def filter_similar_gif_frames(self, min_diff_percent=0.5):
        """
        GIF去重：提取关键帧
        参数说明：
        min_diff: 最小像素差异阈值(0.5%)
        """
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError("无法打开GIF文件")

        prev_frame = None
        unique_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.is_similar(frame, prev_frame, min_diff_percent):
                continue
            else:
                path = self.temp_dir / f"{len(unique_frames)}.png"
                cv2.imwrite(path, frame)
                unique_frames.append(Image(shape=frame.shape, path=path))
            prev_frame = frame
        cap.release()
        return unique_frames

    def set_origin_images(self):
        """设置原始图像（支持GIF和多帧处理）"""
        if not self.path.exists():
            raise FileNotFoundError(f"文件不存在: {self.path}")
        # 处理GIF或多帧图像
        result = []
        if self.path.name.lower().endswith("gif"):
            result = self.filter_similar_gif_frames()
        else:
            result = [Image(shape=None, path=self.path)]
        self.origin_images = result
        if not result:
            raise ValueError(f"无法读取图像文件: {self.path}")

    def set_base_image(self) -> np.ndarray:
        """选择最佳基础图像（包含最多矩形特征）"""
        last_count = 0
        base_img = None
        for parent_img in self.origin_images:
            # 尝试原始图像和反转图像
            for img in [parent_img.data, 255 - parent_img.data]:
                if min(img.shape[:2]) < self.MIN_QR_SIZE:
                    continue
                img = self.clear_image(img)  # 图像增强
                count = self.get_rectangle_count(img)  # 矩形特征计数
                if count > last_count:
                    base_img = img
                    last_count = count
        # 若未设置base_image, 默认为第一帧
        if not isinstance(base_img, np.ndarray):
            base_img = parent_img.data
        path = self.temp_dir / "base.png"
        cv2.imwrite(str(path), base_img)
        self.base_image = Image(shape=base_img.shape, path=path)

        return base_img

    def get_rectangle_count(self, roi_img) -> int:
        """计算图像中矩形特征的数量"""
        # 灰度转换
        if len(roi_img.shape) == 2:  # 单通道
            gray = roi_img.copy()
        elif roi_img.shape[2] == 3:  # 三通道
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image format")
        # 边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blurred, 60, 60)
        # 轮廓检测
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 矩形计数
        rectangle_count = 0
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:  # 四边形判定
                rectangle_count += 1
        return rectangle_count

    def clear_image(self, img):
        """优化版二维码增强处理函数，自动适应不同质量二维码"""
        # 1. 图像预处理
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )
        # 2. 自动特征检测
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = np.sum(bin_img == 0) / bin_img.size  # 黑色像素密度
        contrast = np.std(gray)  # 对比度
        edges = cv2.Canny(gray, 100, 200)
        noise = np.sum(edges > 0) / edges.size  # 噪声水平
        # 3. 动态参数计算（确保block_size为奇数）
        line_thickness = gray.shape[1] * (0.02 if density > 0.35 else 0.01)
        block_size = max(3, min(31, int(line_thickness * 2.5)))
        block_size = block_size + 1 if block_size % 2 == 0 else block_size  # 确保为奇数
        clip_limit = 3.0 if contrast < 30 else 1.5  # CLAHE参数
        # 4. 自适应处理流程
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 5. 二值化处理
        if density > 0.35:  # 粗线条模式
            _, thresh = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            kernel_size = max(3, int(line_thickness * 0.8))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:  # 细线条模式
            adaptive = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,  # 使用修正后的奇数参数
                max(2, int(noise * 10)),  # 动态C值
            )
            result = cv2.medianBlur(adaptive, 3) if noise > 0.15 else adaptive

        return result

    def generate_variants(self):
        """生成图像变体（尺寸缩放、边框、模糊等）"""
        h, w = self.base_image.shape[:2]
        padding_args = {
            "top": h // 2,
            "bottom": h // 2,
            "left": w // 2,
            "right": w // 2,
            "borderType": cv2.BORDER_CONSTANT,
        }
        max_dim = max(h, w)
        # 动态缩放因子（大图用较小缩放）
        if max_dim > 2000:
            scale_factors = [1, 0.5, 2, 0.25]
        else:
            scale_factors = [1, 0.5, 2, 0.25, 4]

        # 基础变体：正常和反转
        # image: qreader: 2 + 4,  other: 2*5*9 + 2*5  + 2*2= 104
        for idx, variant in [
            ["normal", self.base_image.data],
            ["reverse", 255 - self.base_image.data],
        ]:
            skip_qreader = False
            for scale_factor in scale_factors:

                if (
                    not all(25 < axis < 1024 for axis in variant.shape[:2])
                    and scale_factor != 1
                ):
                    # 跳过无效尺寸
                    self.share_manager.total_tasks.value -= 1
                    continue
                # 尺寸缩放
                rescaled_image = cv2.resize(
                    src=variant,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )

                # 保存原始缩放图像
                name = f"{idx}_scale{scale_factor}.png"
                path = self.temp_dir / name
                cv2.imwrite(str(path), rescaled_image)
                yield Image(shape=rescaled_image.shape, path=path), skip_qreader

                # 原始比例图像添加边框
                if scale_factor == 1:
                    for color in [0, 255]:  # 黑白边框
                        bordered = cv2.copyMakeBorder(
                            rescaled_image, **padding_args, value=[color] * 3
                        )
                        name = f"{idx}_scale{scale_factor}_border{color}.png"
                        path = self.temp_dir / name
                        cv2.imwrite(str(path), bordered)
                        yield Image(shape=bordered.shape, path=path), skip_qreader
                    skip_qreader = True  # 后续变体跳过QReader
                # 高斯模糊变体
                for i in range(3, 21, 2):  # 不同核大小
                    name = f"{idx}_scale{scale_factor}_ksize{i}.png"
                    path = self.temp_dir / name
                    blur = cv2.GaussianBlur(src=rescaled_image, ksize=(i, i), sigmaX=0)
                    cv2.imwrite(str(path), blur)

                    yield Image(shape=blur.shape, path=path), skip_qreader

    def process(self):
        """处理管道生成器"""
        # 1. 原始图像
        # len(self.origin_images)
        for img in self.origin_images:
            yield img, False
        # 2. 基础图像变体
        # other:104 qreader: 4
        self.set_base_image()
        for img, skip_qreader in self.generate_variants():
            yield img, skip_qreader


class ResultManager:
    """结果管理器"""

    def __init__(self, stop_event):
        self.results: List[DecodeResult] = []
        self.stop_event = stop_event  # 停止标志

    def add_results(self, results: Iterable[DecodeResult]):
        """添加结果"""
        for result in results:
            self.results.append(result)

    def save(self, output_dir: pathlib.Path) -> pathlib.Path:
        """保存结果到文件"""
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_image(output_dir)
        return self._save_txt(output_dir)

    def _save_image(self, output_dir: pathlib.Path) -> pathlib.Path:
        """复制结果图像到输出目录"""
        for i, result in enumerate(self.results):
            path = output_dir / result.path.name
            shutil.copy(result.path, path)
            self.results[i].path = path

    def _save_txt(self, output_dir: pathlib.Path) -> pathlib.Path:
        """保存文本结果"""
        file_path = output_dir / "results.txt"
        with file_path.open("w", encoding="utf-8") as f:
            for i, result in enumerate(self.results, 1):
                f.write(
                    f"结果 #{i} ({result.engine}) {result.path.absolute()} "
                    f"{result.shape[0]}x{result.shape[1]}:\n"
                    f"Text: {repr(result.text)}\n"
                    f"Bytes: {result.bytes}\n\n"
                )
        return file_path


class ShareManager:
    """多进程共享状态管理"""

    def __init__(self, manager):
        self.stop_event = manager.Event()  # 停止事件
        self.completed_tasks = manager.Value("i", 0)  # 已完成任务计数
        self.total_tasks = manager.Value("i", 0)  # 总任务数
        self.lock = manager.Lock()  # 线程锁

    def safe_increment_completed(self, value=1):
        """线程安全的计数增加"""
        with self.lock:
            self.completed_tasks.value += value


def process_worker(decoder_cls, img, share_manager, debug=DEBUG):
    """工作进程任务函数"""
    if share_manager.stop_event.is_set():
        if debug:
            log(f"[dim]停止任务: {decoder_cls.name} {img}[/]")
        return []
    log(
        f"[dim]正在使用{decoder_cls().name}解码 {img} 进度:{share_manager.completed_tasks.value}/{share_manager.total_tasks.value}"
    )
    data = []
    try:
        decoder = decoder_cls()
        data = decoder.decode(img)
    except KeyboardInterrupt:
        share_manager.stop_event.set()
        exit(1)
    except Exception as e:
        log(f"{decoder_cls.name}解码 {img} 发生错误:{e}")
    finally:
        img.release()
    if not data:
        log(f"[dim]{decoder_cls().name}解码 {img} 完成, 未解析到内容:(")
    share_manager.safe_increment_completed()
    log(
        f"解析进度: {share_manager.completed_tasks.value}/{share_manager.total_tasks.value}"
    )

    return data


def result_callback(results, result_manager, stop_event, pool, stop_immediately):
    """结果回调函数（用于提前终止）"""
    if results:
        if stop_immediately and not stop_event.is_set():
            # 检测到有效结果，立即终止所有进程
            result_manager.add_results(results)
            stop_event.set()
            log("[bold]检测到有效结果，终止所有进程![/]")
        elif not stop_event.is_set():
            # 普通结果添加
            result_manager.add_results(results)


def monitor_workers(pool, share_manager):
    """监控工作进程的内存使用"""
    all_sleep_times = 0
    while True:
        time.sleep(0.5)  # 每0.5秒检查一次
        worker_pids = []
        # 获取所有工作进程的PID
        for worker in pool._pool:
            if worker.is_alive():
                worker_pids.append(worker.pid)
        # 检查每个工作进程的内存
        all_sleeping = True
        for pid in worker_pids:
            try:
                proc = psutil.Process(pid)
                all_sleeping = all_sleeping and proc.status() == "sleeping"
            except psutil.NoSuchProcess:
                pass
        # 所有进程休眠超时处理
        if all_sleeping:
            all_sleep_times += 1
            if all_sleep_times > 3:
                with share_manager.lock:
                    share_manager.completed_tasks.value += len(worker_pids) - 1
                share_manager.stop_event.set()
                log(
                    f"解析进度: {share_manager.completed_tasks.value}/{share_manager.total_tasks.value}"
                )
                return


@app.command()
def main(
    input_path: pathlib.Path = typer.Argument(
        ...,
        help="[magenta]📂 输入图像路径[/]",
        exists=True,
        dir_okay=False,
        readable=True,
        metavar="IMAGE",
    ),
    engines: Annotated[
        List[str],
        typer.Option(
            "--engine",
            "-e",
            help="[cyan]🔧 选择解码引擎 (pyzbar/zxingcpp/opencv/qreader)[/]",
            case_sensitive=False,
        ),
    ] = ["pyzbar", "qreader"],
    complete_all_process: bool = typer.Option(
        False,
        "--complete",
        "-c",
        help="[red]🚫 即使检测到内容也等待所有解码任务完成[/]",
    ),
    output_dir: pathlib.Path = typer.Option(
        pathlib.Path("results") / pathlib.Path(time.strftime("%y%m%d_%H%M%S")),
        "--output",
        "-o",
        help="[green]💾 结果输出目录[/]",
        file_okay=False,
        writable=True,
        metavar="DIR",
    ),
    parallel: int = typer.Option(
        MAX_QREADER_COUNT, "--parallel", "-p", help="并行工作线程数"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="启用调试模式, 将会保留中间文件"
    ),
):
    """
    [bold cyan]🔍 高级二维码解码工具[/]

    [underline]支持功能：[/]
    [blue]• 多解码引擎并行处理[/]
    [green]• 智能图像预处理[/]
    [yellow]• 多种输出格式支持[/]

    [bold]示例：[/]
    [dim]qrdetector *.jpg -e qreader -e opencv -o scan_results --parallel 8[/]
    """
    global MAX_QREADER_COUNT
    global DEBUG
    DEBUG = debug
    DECODERS: Dict[str, Callable[[], Decoder]] = {
        "zxingcpp": ZxingCppDecoder,
        "qreader": QReaderDecoder,
        "pyzbar": PyzbarDecoder,
        "opencv": OpenCVDecoder,
    }
    # QReader内存限制检查
    if "qreader" in engines:
        if MAX_QREADER_COUNT == 0:
            log(f"[red]内存不足使用QReader, 并发数应为1![/]")
            MAX_QREADER_COUNT = 1
        elif parallel > MAX_QREADER_COUNT:
            if parallel != os.cpu_count():
                log(f"[red]当使用QReader时, 并发数应小于等于{MAX_QREADER_COUNT}![/]")
            else:
                log(f"当使用QReader时, 并发数应小于等于{MAX_QREADER_COUNT}, 已修改")
                parallel = MAX_QREADER_COUNT

    # 初始化多进程管理器
    with multiprocessing.Manager() as manager:
        share_manager = ShareManager(manager)
        temp_dir_obj = tempfile.TemporaryDirectory(delete=False)
        temp_dir = pathlib.Path(temp_dir_obj.name)
        log(f"[dim]创建临时目录: {temp_dir}[/]")
        result_manager = ResultManager(share_manager.stop_event)
        qreader_enable = False
        decoders = []
        # 验证解码器
        for e in engines:
            d = DECODERS.get(e.lower())
            if not d:
                log(f"引擎{e}不存在", style="bold red")
                exit(1)
            decoders.append(d)
            if d == QReaderDecoder:
                qreader_enable = True
        total_tasks = 0
        # 计算总任务数
        processor = ImageProcessor(share_manager, input_path, temp_dir)
        # other:104 qreader: 6
        if qreader_enable:
            total_tasks += (len(decoders) - 1) * (
                104 + len(processor.origin_images)
            ) + (6 + len(processor.origin_images))
        else:
            total_tasks += len(decoders) * (104 + len(processor.origin_images))
        share_manager.total_tasks.set(total_tasks)

        try:
            # 创建进程池（使用spawn上下文）
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=parallel)
            async_results = []
            # 遍历解码器
            # for decoder in sorted(decoders, key=lambda x: x.rank):
            for decoder in decoders:
                if share_manager.stop_event.is_set():  # 检查停止标志
                    break
                break_now = decoder.test()

                if break_now == "break":
                    share_manager.stop_event.set()
                    break
                elif break_now == "continue":
                    continue

                if share_manager.stop_event.is_set():  # 检查停止标志
                    break
                log(
                    f"[bright_blue]🔍 正在使用{decoder.name}处理: [bold underline cyan]{processor.path.name}[/]"
                )
                # 遍历图像变体
                for img, skip_qreader in processor.process():
                    if skip_qreader and decoder.name == QReaderDecoder.name:
                        continue
                    if share_manager.stop_event.is_set():  # 检查停止标志
                        break
                    # 提交任务到进程池
                    async_results.append(
                        pool.apply_async(
                            process_worker,
                            args=(decoder, img, share_manager, debug),
                            callback=partial(
                                result_callback,
                                result_manager=result_manager,
                                stop_event=share_manager.stop_event,
                                pool=pool,
                                stop_immediately=not complete_all_process,
                            ),
                        )
                    )
            # 启动进程监控线程
            pool.close()
            monitor_thread = threading.Thread(
                target=monitor_workers, daemon=True, args=(pool, share_manager)
            )
            monitor_thread.start()

            # 等待任务完成
            while True:
                if share_manager.stop_event.is_set():  # 检查停止标志
                    break
                try:
                    for i in async_results:
                        if share_manager.stop_event.is_set():  # 检查停止标志
                            break
                        if (
                            share_manager.completed_tasks.value
                            >= share_manager.total_tasks.value
                        ):
                            share_manager.stop_event.set()
                            break
                        i.get(1)
                except multiprocessing.TimeoutError:
                    pass
                except KeyboardInterrupt:
                    log(
                        f"⚠ 用户中断",
                        style="bold red",
                    )
                    share_manager.stop_event.set()
                    pool.terminate()
                    break
        except KeyboardInterrupt:
            log(
                f"[red]⚠ 用户中断",
                style="bold red",
            )
            share_manager.stop_event.set()
            pool.terminate()
        except Exception as e:
            if debug:
                raise
            log(
                f"[red]⚠ 处理 {input_path.name} 时发生错误:[/] {repr(e)}",
                style="bold red",
            )
        finally:
            # 安全关闭进程池
            pool.terminate()

    # 保存结果
    if result_manager.results:
        output_file = result_manager.save(output_dir)

        # 显示结果摘要
        table = Table(
            title="[bold magenta]解码结果摘要[/]",
            box=box.ROUNDED,
            caption=f"📁 输出文件: [dim cyan]{output_dir.absolute()}[/]",
            show_lines=False,
            header_style="bold cyan",
            title_style="bold italic",
            caption_style="not bold dim",
        )
        table.add_column("#", style="bright_white")
        table.add_column("内容摘要", style="bright_yellow")
        table.add_column("引擎", justify="center", style="bright_cyan")
        table.add_column("尺寸", justify="left", style="bright_green")
        table.add_column("位置", justify="left", style="dim")

        for idx, result in enumerate(result_manager.results, 1):
            engine_color = {
                "zxingcpp": "bright_cyan",
                "pyzbar": "bright_magenta",
                "opencv": "bright_cyan",
                "qreader": "bright_green",
            }.get(result.engine.lower(), "white")

            text = Text(result.text)
            table.add_row(
                Text(f"{idx}", style="bright_white"),
                text,
                Text(result.engine, style=engine_color),
                Text(f"{result.shape[0]}×{result.shape[1]}", style="dim bright_green"),
                Text(str(result.path), style="dim"),
            )

        log(table)
    else:
        log("[bold yellow]⚠ 未检测到有效二维码[/]")
    # 清理临时文件
    if debug:
        log(f"[yellow]调试模式保留临时目录: {temp_dir}[/]")
    else:
        log(f"[dim]清理临时目录: {temp_dir}[/]")
        temp_dir_obj.cleanup()  # 清理临时目录


if __name__ == "__main__":
    app()
