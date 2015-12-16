import click
import camtones

from camtones.procs.motion import MotionDetectProcess, MotionExtractProcess, MotionExtractEDLProcess
from camtones.procs.face import FaceDetectProcess, FaceExtractProcess
from camtones.ocv import api as ocv


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("Version {}.{}.{}".format(*camtones.__version__))
    ctx.exit()


@click.group()
@click.option("--debug", is_flag=True, default=False, help="debug enabled")
@click.option("--version", is_flag=True, is_eager=True, default=False, help="print version", callback=print_version)
@click.pass_context
def cli(ctx, debug, version):
    ctx.obj['DEBUG'] = debug


@cli.command(help="Show a video or camera device highlighting the detected motion")
@click.argument("video-or-device")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--resize", type=int, help="resize the image before detect")
@click.option("--blur", type=int, help="blur the mask before detect")
@click.option(
    "--subtractor",
    type=click.Choice(ocv.get_supported_subtractors().keys()),
    default="MOG2" if "MOG2" in ocv.get_supported_subtractors() else "KNN",
    help="select background subtractor"
)
@click.pass_context
def motion_detect(ctx, video_or_device, exclude, resize, blur, subtractor):
    MotionDetectProcess(video_or_device, ctx.obj['DEBUG'], exclude, resize, blur, subtractor).run()


@cli.command(help="Generate video file from a video or camera device without the no-motion sections")
@click.argument("video-or-device")
@click.argument("output-file")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--progress", is_flag=True, help="print progress")
@click.option("--resize", type=int, help="resize the image before detect")
@click.option("--blur", type=int, help="blur the mask before detect")
@click.option("--show-time", is_flag=True, help="show the current time")
@click.option(
    "--subtractor",
    type=click.Choice(ocv.get_supported_subtractors().keys()),
    default="MOG2" if "MOG2" in ocv.get_supported_subtractors() else "KNN",
    help="select background subtractor"
)
@click.pass_context
def motion_extract(ctx, video_or_device, output_file, exclude, progress, resize, blur, show_time, subtractor):
    MotionExtractProcess(video_or_device, ctx.obj['DEBUG'], exclude, output_file, progress, resize, blur, show_time, subtractor).run()


@cli.command(help="Generate an EDL file from a video or camera device to skip no-motion sections")
@click.argument("video-or-device")
@click.argument("output-file")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--progress", is_flag=True, help="print progress")
@click.option("--resize", type=int, help="resize the image before detect")
@click.option("--blur", type=int, help="blur the mask before detect")
@click.option(
    "--subtractor",
    type=click.Choice(ocv.get_supported_subtractors().keys()),
    default="MOG2" if "MOG2" in ocv.get_supported_subtractors() else "KNN",
    help="select background subtractor"
)
@click.pass_context
def motion_extract_edl(ctx, video_or_device, output_file, exclude, progress, resize, blur, subtractor):
    MotionExtractEDLProcess(video_or_device, ctx.obj['DEBUG'], exclude, output_file, progress, resize, blur, subtractor).run()


@cli.command(help="Show a video or camera device highlighting the detected faces")
@click.argument("video-or-device")
@click.option("--classifier", help="path to the cascade classifier file", required=True)
@click.pass_context
def face_detect(ctx, video_or_device, classifier):
    FaceDetectProcess(video_or_device, ctx.obj['DEBUG'], classifier).run()


@cli.command(help="List the available face classifiers")
def classifiers():
    for classifier in ocv.get_stock_classifiers():
        click.echo(" - {}".format(classifier))

@cli.command(help="Generate a set of images from a video or camera device with the detected faces")
@click.argument("video-or-device")
@click.argument("output-directory")
@click.option("--classifier", help="path to the cascade classifier file", required=True)
@click.option("--progress", is_flag=True, help="print progress")
@click.pass_context
def face_extract(ctx, video_or_device, output_directory, classifier, progress):
    FaceExtractProcess(video_or_device, ctx.obj['DEBUG'], output_directory, classifier, progress).run()


if __name__ == '__main__':
    cli(obj={})
