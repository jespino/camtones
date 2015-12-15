import click
import camtones

from camtones.procs.motion import MotionDetectProcess, MotionExtractProcess, MotionExtractEDLProcess
from camtones.procs.face import FaceDetectProcess, FaceExtractProcess


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


@cli.command()
@click.argument("video-or-device")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--resize", type=int, help="print progress")
@click.option("--blur", type=int, help="print progress")
@click.pass_context
def motion_detect(ctx, video_or_device, exclude, resize, blur):
    MotionDetectProcess(video_or_device, ctx.obj['DEBUG'], exclude, resize, blur).run()


@cli.command()
@click.argument("video-or-device")
@click.argument("output-file")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--progress", is_flag=True, help="print progress")
@click.option("--resize", type=int, help="print progress")
@click.option("--blur", type=int, help="print progress")
@click.option("--show-time", is_flag=True, help="show the current time")
@click.pass_context
def motion_extract(ctx, video_or_device, output_file, exclude, progress, resize, blur, show_time):
    MotionExtractProcess(video_or_device, ctx.obj['DEBUG'], exclude, output_file, progress, resize, blur, show_time).run()


@cli.command()
@click.argument("video-or-device")
@click.argument("output-file")
@click.option("--exclude", help="python expression to exclude results")
@click.option("--progress", is_flag=True, help="print progress")
@click.option("--resize", type=int, help="print progress")
@click.option("--blur", type=int, help="print progress")
@click.pass_context
def motion_extract_edl(ctx, video_or_device, output_file, exclude, progress, resize, blur):
    MotionExtractEDLProcess(video_or_device, ctx.obj['DEBUG'], exclude, output_file, progress, resize, blur).run()


@cli.command()
@click.argument("video-or-device")
@click.option("--classifier", help="path to the cascade classifier file", required=True)
@click.pass_context
def face_detect(ctx, video_or_device, classifier):
    FaceDetectProcess(video_or_device, ctx.obj['DEBUG'], classifier).run()


@cli.command()
@click.argument("video-or-device")
@click.argument("output-directory")
@click.option("--classifier", help="path to the cascade classifier file", required=True)
@click.pass_context
def face_extract(ctx, video_or_device, output_directory, classifier):
    FaceExtractProcess(video_or_device, ctx.obj['DEBUG'], output_directory, classifier).run()


if __name__ == '__main__':
    cli(obj={})
