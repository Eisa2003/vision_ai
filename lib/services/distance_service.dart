// FILE: lib/services/distance_service.dart

import 'package:flutter/painting.dart';

/// Estimates the distance to a detected object using a simple heuristic.
///
/// The core idea: larger bounding boxes → object is closer.
/// This is a **placeholder model** designed to be swapped out for a proper
/// depth model (e.g., MiDaS, monocular depth CNN) later.
///
/// Formula:
///   distance ≈ REFERENCE_AREA / boundingBoxArea  (clamped to sane bounds)
///
/// [REFERENCE_AREA] is a tuneable constant. Calibrate it by placing a known
/// object at a known distance and adjusting until the output matches.
class DistanceService {
  // ── Tuneable constants ──────────────────────────────────────────────────
  // Represents the product of (apparent area) × (real distance) for a
  // "typical" object filling ~25% of a 1080p frame at ~2 m.
  static const double _referenceArea = 120000.0;

  static const double _minDistance = 0.3;  // metres
  static const double _maxDistance = 15.0; // metres
  // ────────────────────────────────────────────────────────────────────────

  /// Returns a human-readable distance string, e.g. "1.4 m" or "< 0.3 m".
  ///
  /// [boundingBox]  – the detected object's rect in **image pixel** coords.
  /// [imageSize]    – native resolution of the source image/frame.
  String estimate(Rect boundingBox, Size imageSize) {
    final meters = estimateMeters(boundingBox, imageSize);

    if (meters <= _minDistance) return '< ${_minDistance.toStringAsFixed(1)} m';
    if (meters >= _maxDistance) return '> ${_maxDistance.toStringAsFixed(0)} m';
    return '~${meters.toStringAsFixed(1)} m';
  }

  /// Returns the raw estimated distance in metres.
  double estimateMeters(Rect boundingBox, Size imageSize) {
    final boxArea = boundingBox.width * boundingBox.height;
    if (boxArea <= 0) return _maxDistance;

    // Normalise by image resolution so the constant stays resolution-agnostic
    final imageArea = imageSize.width * imageSize.height;
    final normalised = boxArea / imageArea;   // 0.0 – 1.0

    final distance = (_referenceArea / imageArea) / normalised;
    return distance.clamp(_minDistance, _maxDistance);
  }

  // ── Extension point ──────────────────────────────────────────────────────
  // To replace this with a real depth model:
  // 1. Add your model file to assets/
  // 2. Create a ModelDepthService that implements the same [estimate] signature
  // 3. Swap the instance in your DI / providers
  // ────────────────────────────────────────────────────────────────────────
}
