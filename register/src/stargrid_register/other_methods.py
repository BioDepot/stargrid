"""Other registration measures, kept for later work; not used by the command-line tool.

Phase correlation, OpenCV ECC (translation or affine) and SimpleITK Mattes mutual information
(translation, affine or cubic B-spline). In the tests behind the STARgrid manuscript (frozen
settings of 2026-10-07, below) none of them met all criteria: phase correlation and ECC did not
recover a planted displacement on the calibration slide; optimised mutual information met the
ovarian check and predicted the regional drift on the SPATCH slide but failed the colorectal
control and the capture test; the B-spline stage failed its zero-displacement control.

Inputs are the band-passed mask and counts of `methods.prepare`. Displacements are (dy, dx), mask
to counts: OpenCV and SimpleITK use the mask as fixed image and the counts as moving image, so
their sampling transforms already have this direction. They need the optional dependencies
`opencv-python-headless` and `SimpleITK` (pip install 'stargrid-register[other]').
"""
from __future__ import annotations

import numpy as np

# The frozen settings of the manuscript's registration-methods test (2026-10-07).
FROZEN_2026_10_07 = {
    "seed": 20261007, "threads": 8, "radius": 12, "bandpass_sigma": 8,
    "mi_bins": 32, "mi_sampling": 0.2, "mi_shrink": [2, 1], "mi_smoothing": [1, 0],
    "mi_optimizer": "gradient", "mi_learning_rate": 1.0, "mi_min_step": 0.01, "mi_iterations": 150,
    "bspline_mesh": [6, 6], "ecc_iterations": 150, "ecc_epsilon": 1e-05, "ecc_gaussian_size": 3,
    "phase_window": False,
}


def phase(a, b, tissue, window=True):
    import cv2
    first = np.asarray(a * tissue, np.float32).copy()
    second = np.asarray(-b * tissue, np.float32).copy()
    hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F) if window else None
    (dx, dy), response = cv2.phaseCorrelate(first, second, hann)
    return dict(status='ok', dy=float(dy), dx=float(dx), metric=float(response))


def ecc(a, b, tissue, config, start=(0, 0), affine=False, *, fixed_mask=None,
        moving_mask=None, initial_matrix=None):
    import cv2
    matrix = np.eye(2, 3, dtype=np.float32)
    matrix[:, 2] = [start[1], start[0]]
    if initial_matrix is not None:
        matrix = np.asarray(initial_matrix, np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                config['ecc_iterations'], config['ecc_epsilon'])
    try:
        target_mask = tissue if moving_mask is None else moving_mask
        if fixed_mask is None:
            score, matrix = cv2.findTransformECC(a, -b, matrix,
                cv2.MOTION_AFFINE if affine else cv2.MOTION_TRANSLATION,
                criteria, target_mask.astype(np.uint8) * 255, config['ecc_gaussian_size'])
        else:
            score, matrix = cv2.findTransformECCWithMask(a, -b,
                fixed_mask.astype(np.uint8) * 255, target_mask.astype(np.uint8) * 255,
                matrix, cv2.MOTION_AFFINE if affine else cv2.MOTION_TRANSLATION,
                criteria, config['ecc_gaussian_size'])
    except cv2.error as exc:
        return dict(status='nonconverged', reason=str(exc), start=list(start))
    centre = np.array([(a.shape[1] - 1) / 2, (a.shape[0] - 1) / 2, 1])
    delta = matrix @ centre - centre[:2]
    return dict(status='ok', dy=float(delta[1]), dx=float(delta[0]),
                metric=float(score), matrix_xy=matrix.tolist(), start=list(start))


def sitk_mi(a, b, tissue, config, start=(0, 0), model='translation', *,
            fixed_mask=None, moving_mask=None, initial_matrix=None, transform_path=None):
    import SimpleITK as sitk
    fixed, moving = sitk.GetImageFromArray(a), sitk.GetImageFromArray(b)
    source_mask = sitk.GetImageFromArray((tissue if fixed_mask is None else fixed_mask).astype(np.uint8))
    target_mask = sitk.GetImageFromArray((tissue if moving_mask is None else moving_mask).astype(np.uint8))
    if model == 'translation':
        initial = sitk.TranslationTransform(2, [float(start[1]), float(start[0])])
    elif model == 'affine':
        initial = sitk.AffineTransform(2)
        initial.SetCenter([(a.shape[1] - 1) / 2, (a.shape[0] - 1) / 2])
        initial.SetTranslation([float(start[1]), float(start[0])])
        if initial_matrix is not None:
            matrix = np.asarray(initial_matrix)
            centre = np.array(initial.GetCenter())
            initial.SetMatrix(matrix[:, :2].ravel().tolist())
            initial.SetTranslation((matrix[:, 2] + matrix[:, :2] @ centre - centre).tolist())
    elif model == 'bspline':
        initial = sitk.BSplineTransformInitializer(fixed, config['bspline_mesh'], order=3)
    else:
        raise ValueError(f'unknown transform: {model}')
    reg = sitk.ImageRegistrationMethod()
    reg.SetNumberOfThreads(config['threads'])
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=config['mi_bins'])
    reg.SetMetricFixedMask(source_mask)
    reg.SetMetricMovingMask(target_mask)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(config['mi_sampling'], config['seed'])
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(config['mi_shrink'])
    reg.SetSmoothingSigmasPerLevel(config['mi_smoothing'])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if config.get('mi_optimizer', 'gradient') == 'powell':
        reg.SetOptimizerAsPowell(numberOfIterations=config['mi_iterations'],
            maximumLineIterations=50, stepLength=config['mi_learning_rate'],
            stepTolerance=config['mi_min_step'], valueTolerance=1e-7)
    else:
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=config['mi_learning_rate'], minStep=config['mi_min_step'],
            numberOfIterations=config['mi_iterations'], relaxationFactor=.5,
            gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=True)
    base = None
    if model == 'bspline' and initial_matrix is not None:
        matrix = np.asarray(initial_matrix)
        base = sitk.AffineTransform(2)
        base.SetMatrix(matrix[:, :2].ravel().tolist())
        base.SetTranslation(matrix[:, 2].tolist())
        reg.SetMovingInitialTransform(base)
    try:
        transform = reg.Execute(fixed, moving)
        if base is not None:
            transform = sitk.CompositeTransform([base, transform])
        if transform_path is not None:
            sitk.WriteTransform(transform, str(transform_path))
        centre = [(a.shape[1] - 1) / 2, (a.shape[0] - 1) / 2]
        delta = np.asarray(transform.TransformPoint(centre)) - centre
        if not np.isfinite(delta).all():
            raise RuntimeError('nonfinite transform')
        answer = dict(status='ok', dy=float(delta[1]), dx=float(delta[0]),
                      metric=float(reg.GetMetricValue()), start=list(start),
                      stop=reg.GetOptimizerStopConditionDescription(),
                      parameters=list(initial.GetParameters()),
                      fixed_parameters=list(initial.GetFixedParameters()), model=model)
        if model != 'bspline':
            origin = np.array(transform.TransformPoint((0., 0.)))
            x = np.array(transform.TransformPoint((1., 0.))) - origin
            y = np.array(transform.TransformPoint((0., 1.))) - origin
            answer['matrix_xy'] = np.column_stack([x, y, origin]).tolist()
        return answer
    except RuntimeError as exc:
        return dict(status='nonconverged', reason=str(exc), start=list(start), model=model)
