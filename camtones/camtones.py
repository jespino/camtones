import click
import numpy as np
import cv2
import time
import imutils
import datetime
import os

from imutils.object_detection import non_max_suppression
from imutils import paths


@click.group()
def cli():
    pass

@click.command()
@click.option("--video", help="path to the video file")
@click.option("--min-area", type=int, default=500, help="minimum area size")
@click.option("--debug", is_flag=True, default=False, help="debug enabled")
def motion_detect(video, min_area, debug):
    if video:
        camera = cv2.VideoCapture(video)
    else:
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        (cnts, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Motion extract", frame)
        if debug:
            cv2.imshow("FGMASK", fgmask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

@click.command()
@click.option("--video", help="path to the video file")
@click.option("--min-area", type=int, default=500, help="minimum area size")
@click.option("--output", help="path to the output video file")
def motion_extract(video, min_area, output):
    if video:
        camera = cv2.VideoCapture(video)
    else:
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)

    if not output:
        click.echo("Invalid output")
        return

    fps = camera.get(cv2.CAP_PROP_FPS)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output, fourcc, fps, size)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        (grabbed, frame) = camera.read()
        motion_detected = False

        if not grabbed:
            break

        miniframe = imutils.resize(frame, width=500)
        fgmask = fgbg.apply(miniframe)
        fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        (cnts, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue

            motion_detect = True
            break

        if motion_detected:
            output.write(frame)

    camera.release()
    cv2.destroyAllWindows()


@click.command()
@click.option("--video", help="path to the video file")
@click.option("--classifier", help="path to the cascade classifier file")
@click.option("--debug", is_flag=True, default=False, help="debug enabled")
def face_detect(video, classifier, debug):
    if video:
        camera = cv2.VideoCapture(video)
    else:
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)

    if output is not None and not os.path.exists(output):
        click.echo("Invalid directory")
        return

    faceCascade = cv2.CascadeClassifier(classifier)

    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        milisecond = camera.get(cv2.CAP_PROP_POS_MSEC)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.HAAR_SCALE_IMAGE
        )

        counter = 0;
        for (x, y, w, h) in faces:
            counter += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face detect", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


@click.command()
@click.option("--video", help="path to the video file")
@click.option("--output", help="path to the faces output directory")
@click.option("--classifier", help="path to the cascade classifier file")
def face_extract(video, output, classifier):
    if video:
        camera = cv2.VideoCapture(video)
    else:
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)

    if output is not None and not os.path.exists(output):
        click.echo("Invalid directory")
        return

    faceCascade = cv2.CascadeClassifier(classifier)

    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        milisecond = camera.get(cv2.CAP_PROP_POS_MSEC)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            # scaleFactor=1.1,
            # minNeighbors=5,
            # minSize=(30, 30),
            # flags = cv2.HAAR_SCALE_IMAGE
        )

        counter = 0;
        for (x, y, w, h) in faces:
            counter += 1
            if output is None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                croped = frame[y:y + h, x:x + w]
                cv2.imwrite("{}/{}-{}.png".format(output, milisecond, counter), croped)

        if output is None:
            print("HOLA")
            cv2.imshow("Faces found", frame)
            key = cv2.waitKey(1) & 0xFF
            # cv2.waitKey(0)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cli.add_command(motion_detect)
    cli.add_command(motion_extract)
    cli.add_command(face_extract)
    cli()

