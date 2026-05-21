"""Utility functions for OAK camera pipeline creation and metadata conversion.

Source:   https://github.com/beefriendly-earth/inspectai-cam
License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)

Functions:
    convert_cm_lens_position(): Convert closest available centimeter value to OAK lens position value.
    deletterbox_bbox():         De-letterbox bounding box from NN-normalized letterboxed space
                                to frame-normalized space.
    create_pipeline():          Create and configure depthai pipeline for OAK camera.
"""

import logging
from datetime import timedelta
from typing import cast

import depthai as dai

from insectdetect.config import AppConfig
from insectdetect.constants import MODELS_PATH, RESOLUTION_PRESETS, SENSOR_RES, SENSOR_ROI

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Mapping of object-to-camera distances (8-80 cm) to OAK lens position values (255-120)
CM_LENS_POSITIONS: dict[int, int] = {
    8: 255, 9: 210, 10: 200, 11: 190, 12: 180, 13: 175, 14: 170, 15: 165, 16: 162, 17: 160,
    18: 158, 19: 156, 20: 154, 21: 152, 22: 150, 23: 148, 24: 146, 25: 144, 26: 142, 27: 141,
    28: 140, 29: 139, 30: 138, 31: 137, 32: 136, 34: 135, 36: 134, 38: 133, 40: 132, 42: 131,
    45: 130, 48: 129, 52: 128, 56: 127, 60: 126, 64: 125, 68: 124, 72: 123, 76: 122, 80: 120,
}
CM_KEYS: tuple[int, ...] = tuple(CM_LENS_POSITIONS.keys())


def convert_cm_lens_position(distance_cm: int) -> int:
    """Convert closest available centimeter value to OAK lens position value.

    Args:
        distance_cm: Object-to-camera distance in centimeters.

    Returns:
        OAK lens position value (120-255) corresponding to the given distance.
    """
    if distance_cm in CM_KEYS:
        return CM_LENS_POSITIONS[distance_cm]
    closest_cm = min(CM_KEYS, key=lambda k: abs(k - distance_cm))
    return CM_LENS_POSITIONS[closest_cm]


def deletterbox_bbox(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    nn_w: int,
    nn_h: int,
) -> tuple[float, float, float, float]:
    """De-letterbox bounding box from NN-normalized letterboxed space to frame-normalized space.

    Args:
        bbox:    Bounding box (x_min, y_min, x_max, y_max) in NN-normalized letterboxed space.
        frame_w: Active frame width in pixels.
        frame_h: Active frame height in pixels.
        nn_w:    NN input width in pixels.
        nn_h:    NN input height in pixels.

    Returns:
        Bounding box (x_min, y_min, x_max, y_max) in frame-normalized space [0, 1].
    """
    scale = min(nn_w / frame_w, nn_h / frame_h)
    scaled_w = frame_w * scale
    scaled_h = frame_h * scale
    pad_x = (nn_w - scaled_w) / 2 / nn_w
    pad_y = (nn_h - scaled_h) / 2 / nn_h
    content_x = scaled_w / nn_w
    content_y = scaled_h / nn_h
    return (
        max(0.0, (bbox[0] - pad_x) / content_x),
        max(0.0, (bbox[1] - pad_y) / content_y),
        min(1.0, (bbox[2] - pad_x) / content_x),
        min(1.0, (bbox[3] - pad_y) / content_y),
    )


def create_pipeline(
    config: AppConfig,
    stream: bool = False
) -> tuple[
    dai.Pipeline,
    dai.MessageQueue,
    dai.MessageQueue,
    dai.MessageQueue,
    dai.InputQueue,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    list[str],
]:
    """Create and configure depthai pipeline for OAK camera.

    Args:
        config: AppConfig containing all configuration settings.
        stream: If True, request ISP output at stream resolution for webapp.py.
                If False, request ISP output at HQ resolution for capture.py.

    Returns:
        Tuple of (pipeline, q_frames, q_tracks, q_syslog, q_camctrl,
                  frame_size, nn_input_size, sensor_roi, labels).
    """
    pipeline = dai.Pipeline()

    # Get sensor and output resolution
    sensor_w, sensor_h = SENSOR_RES
    image_w, image_h, stream_w, stream_h = RESOLUTION_PRESETS[config.camera.image.resolution]
    out_w, out_h = (image_w, image_h)  # always use full resolution, even for webapp stream

    # Create Camera node and set initial control options
    cam = pipeline.create(dai.node.Camera).build(
        sensorResolution=(sensor_w, sensor_h),
        sensorFps=config.camera.fps
    )

    cam.initialControl.setSharpness(config.camera.isp.sharpness)
    cam.initialControl.setLumaDenoise(config.camera.isp.luma_denoise)
    cam.initialControl.setChromaDenoise(config.camera.isp.chroma_denoise)

    if config.camera.focus.mode == "manual":
        # Set manual focus using either object-to-camera distance (cm) or lens position (0-255)
        cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        if config.camera.focus.type == "distance":
            cam.initialControl.setManualFocus(
                convert_cm_lens_position(config.camera.focus.manual.distance)
            )
        elif config.camera.focus.type == "lens_pos":
            cam.initialControl.setManualFocus(config.camera.focus.manual.lens_pos)
    elif config.camera.focus.mode == "range":
        # Set auto focus range using either object-to-camera distance (cm) or lens position (0-255)
        # Note: higher distance -> lower lens position, so min/max are intentionally swapped
        if config.camera.focus.type == "distance":
            cam.initialControl.setAutoFocusLensRange(
                convert_cm_lens_position(config.camera.focus.range.distance.max),
                convert_cm_lens_position(config.camera.focus.range.distance.min)
            )
        elif config.camera.focus.type == "lens_pos":
            cam.initialControl.setAutoFocusLensRange(
                config.camera.focus.range.lens_pos.min,
                config.camera.focus.range.lens_pos.max,
            )

    # Set auto exposure and focus ROI to the defined sensor ROI
    cam.initialControl.setAutoExposureRegion(*SENSOR_ROI)
    if config.camera.focus.mode != "manual":
        cam.initialControl.setAutoFocusRegion(*SENSOR_ROI)

    # Request camera output with configured resolution
    cam_out = cam.requestOutput(
        size=(out_w, out_h),
        type=dai.ImgFrame.Type.NV12,
        resizeMode=dai.ImgResizeMode.CROP,
        fps=config.camera.fps
    )

    # Calculate cropping parameters to crop to the defined sensor ROI
    roi_x, roi_y, roi_w, roi_h = SENSOR_ROI
    scale_x = out_w / sensor_w
    scale_y = out_h / sensor_h
    crop_x = round(roi_x * scale_x)
    crop_y = round(roi_y * scale_y)
    crop_w = round(roi_w * scale_x) // 32 * 32
    crop_h = round(roi_h * scale_y) // 32 * 32

    # After 90° CW rotation, width and height are swapped for downstream nodes
    target_w = crop_h
    target_h = crop_w

    # Create ImageManip node to crop to the defined sensor ROI and rotate frames 90°
    crop_rot_manip = pipeline.create(dai.node.ImageManip)
    crop_rot_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    crop_rot_manip.initialConfig.addCrop(crop_x, crop_y, crop_w, crop_h)
    crop_rot_manip.initialConfig.addRotateDeg(90)
    crop_rot_manip.setMaxOutputFrameSize(crop_w * crop_h * 3 // 2)
    crop_rot_manip.inputImage.setBlocking(False)
    cam_out.link(crop_rot_manip.inputImage)
    cam_frames = crop_rot_manip.out

    # Create VideoEncoder node for MJPEG-encoding of HQ frames
    encoder = pipeline.create(dai.node.VideoEncoder).build(
        input=cam_frames,
        frameRate=1,
        profile=dai.VideoEncoderProperties.Profile.MJPEG,
        quality=config.webapp.stream.quality if stream else config.camera.image.quality
    )

    # Load the specified detection model from the models directory and get input size and type
    nn_archive = dai.NNArchive(MODELS_PATH / config.detection.model)
    nn_input_size = nn_archive.getInputSize()
    nn_input_type_str = nn_archive.getConfig().model.inputs[0].preprocessing.daiType or "BGR888p"
    nn_input_type = cast("dai.ImgFrame.Type", getattr(dai.ImgFrame.Type, nn_input_type_str))
    if nn_input_size is None:
        raise ValueError(f"Could not retrieve input size from NN archive: {config.detection.model}")
    frame_aspect = target_w / target_h
    nn_aspect = nn_input_size[0] / nn_input_size[1]
    aspect_diff = abs(frame_aspect - nn_aspect)
    if aspect_diff > 0.1:
        logger.warning(
            "Aspect ratio mismatch between frame size (%dx%d) and NN input size (%dx%d). "
            "Letterboxing will add %.1f%% padding.", target_w, target_h,
            nn_input_size[0], nn_input_size[1], aspect_diff / max(frame_aspect, nn_aspect) * 100
        )

    # Create ImageManip node to resize (Letterbox) and format frames for detection network input
    nn_manip = pipeline.create(dai.node.ImageManip)
    nn_manip.initialConfig.setFrameType(nn_input_type)
    nn_manip.initialConfig.setOutputSize(nn_input_size[0], nn_input_size[1],
                                         dai.ImageManipConfig.ResizeMode.LETTERBOX)
    nn_manip.setMaxOutputFrameSize(nn_input_size[0] * nn_input_size[1] * 3)
    nn_manip.inputImage.setBlocking(False)
    cam_frames.link(nn_manip.inputImage)

    # Create and configure DetectionNetwork node
    nn_det = pipeline.create(dai.node.DetectionNetwork).build(
        input=nn_manip.out,
        nnArchive=nn_archive
    )
    nn_det.setNNArchive(nn_archive, numShaves=config.detection.num_shaves)  # type: ignore[call-arg]
    nn_det.setConfidenceThreshold(config.detection.conf_threshold)
    nn_det.setNumInferenceThreads(2)
    nn_det.input.setBlocking(False)
    labels = nn_det.getClasses()
    if labels is None:
        raise ValueError(f"Could not retrieve class labels from NN archive: {config.detection.model}")

    # Create and configure ObjectTracker node
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setOcclusionRatioThreshold(0.5)  # threshold to filter out overlapping tracklets (default: 0.3)
    tracker.setTrackletBirthThreshold(20)    # frames to consider tracklet as TRACKED (default: 3)
    tracker.setTrackletMaxLifespan(300)      # frames before a LOST tracklet is removed (default: 120)
    nn_det.passthrough.link(tracker.inputTrackerFrame)
    nn_det.passthrough.link(tracker.inputDetectionFrame)
    nn_det.out.link(tracker.inputDetections)

    # Create Sync node to synchronize HQ frames with tracker output
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=100))
    encoder.bitstream.link(sync.inputs["frames"])
    tracker.out.link(sync.inputs["tracks"])

    # Create MessageDemux node and output queues for frames and tracks
    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)
    q_frames = demux.outputs["frames"].createOutputQueue(maxSize=2, blocking=False)
    q_tracks = demux.outputs["tracks"].createOutputQueue(maxSize=2, blocking=False)

    # Create SystemLogger node to periodically get system information from OAK device
    syslog = pipeline.create(dai.node.SystemLogger)
    syslog.setRate(1)
    q_syslog = syslog.out.createOutputQueue(maxSize=2, blocking=False)

    # Create input queue to send camera control commands (e.g. AE region, focus control)
    q_camctrl = cam.inputControl.createInputQueue(maxSize=1, blocking=False)

    return (pipeline, q_frames, q_tracks, q_syslog, q_camctrl,
            (target_w, target_h), nn_input_size, SENSOR_ROI, labels)
