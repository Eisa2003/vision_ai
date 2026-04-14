// FILE: lib/widgets/bounding_box_painter.dart

import 'package:flutter/material.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';

/// Data class holding everything needed to render one detection result.
class DetectionResult {
  final Rect boundingBox;
  final String label;
  final double confidence;
  final String distance;

  const DetectionResult({
    required this.boundingBox,
    required this.label,
    required this.confidence,
    required this.distance,
  });
}

/// Paints bounding boxes and labels on top of a camera preview or image.
///
/// [results]        – list of detected objects to draw.
/// [imageSize]      – the native resolution of the source (camera frame / image).
/// [widgetSize]     – the on-screen size of the preview/image widget.
/// [isFrontCamera]  – set true to mirror X-axis for selfie camera.
class BoundingBoxPainter extends CustomPainter {
  final List<DetectionResult> results;
  final Size imageSize;
  final Size widgetSize;
  final bool isFrontCamera;

  static const _accentCyan = Color(0xFF00E5FF);
  static const _accentPurple = Color(0xFF7C4DFF);

  BoundingBoxPainter({
    required this.results,
    required this.imageSize,
    required this.widgetSize,
    this.isFrontCamera = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty) return;

    // imageSize is already rotation-corrected (tall in portrait) — no swap needed
    final double frameW = imageSize.width;
    final double frameH = imageSize.height;

    final previewRect = _getPreviewRect(widgetSize, frameW, frameH);

    final scaleX = previewRect.width / frameW;
    final scaleY = previewRect.height / frameH;

    for (int i = 0; i < results.length; i++) {
      final color = i.isEven ? _accentCyan : _accentPurple;
      _drawBox(canvas, results[i], scaleX, scaleY, previewRect.left,
          previewRect.top, color);
    }
  }

  /// Returns the rect of the actual rendered camera image inside the widget.
  /// CameraPreview fits the feed using BoxFit.cover by default — adjust if yours differs.
  Rect _getPreviewRect(Size widget, double frameW, double frameH) {
    final cameraAspect = frameW / frameH;
    final widgetAspect = widget.width / widget.height;

    double renderW, renderH;

    // BoxFit.cover: fill widget, crop excess
    if (cameraAspect > widgetAspect) {
      renderH = widget.height;
      renderW = renderH * cameraAspect;
    } else {
      renderW = widget.width;
      renderH = renderW / cameraAspect;
    }

    final offsetX = (widget.width - renderW) / 2;
    final offsetY = (widget.height - renderH) / 2;

    return Rect.fromLTWH(offsetX, offsetY, renderW, renderH);
  }

  void _drawBox(Canvas canvas, DetectionResult result, double scaleX,
      double scaleY, double offsetX, double offsetY, Color color) {
    final raw = result.boundingBox;

    // Scale + shift into the actual preview rect
    double left = offsetX + raw.left * scaleX;
    double top = offsetY + raw.top * scaleY;
    double right = offsetX + raw.right * scaleX;
    double bottom = offsetY + raw.bottom * scaleY;

    // Mirror for front camera
    if (isFrontCamera) {
      final tmp = widgetSize.width - right;
      right = widgetSize.width - left;
      left = tmp;
    }

    final rect = Rect.fromLTRB(left, top, right, bottom);

    // ── Box stroke ──────────────────────────────────────────────────────────
    final boxPaint = Paint()
      ..color = color.withOpacity(0.85)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(6)),
      boxPaint,
    );

    // ── Corner accents ───────────────────────────────────────────────────────
    final cornerPaint = Paint()
      ..color = color
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    const cLen = 12.0;
    // TL
    canvas.drawLine(Offset(left, top + cLen), Offset(left, top), cornerPaint);
    canvas.drawLine(Offset(left, top), Offset(left + cLen, top), cornerPaint);
    // TR
    canvas.drawLine(Offset(right - cLen, top), Offset(right, top), cornerPaint);
    canvas.drawLine(Offset(right, top), Offset(right, top + cLen), cornerPaint);
    // BR
    canvas.drawLine(
        Offset(right, bottom - cLen), Offset(right, bottom), cornerPaint);
    canvas.drawLine(
        Offset(right, bottom), Offset(right - cLen, bottom), cornerPaint);
    // BL
    canvas.drawLine(
        Offset(left + cLen, bottom), Offset(left, bottom), cornerPaint);
    canvas.drawLine(
        Offset(left, bottom), Offset(left, bottom - cLen), cornerPaint);

    // ── Label background ────────────────────────────────────────────────────
    final labelText =
        '${result.label}  ${(result.confidence * 100).toStringAsFixed(0)}%';
    final distText = result.distance;

    final labelSpan = TextSpan(
      text: labelText,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.5,
      ),
    );
    final distSpan = TextSpan(
      text: distText,
      style: TextStyle(
        color: color,
        fontSize: 10,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
      ),
    );

    final labelPainter = TextPainter(
      text: labelSpan,
      textDirection: TextDirection.ltr,
    )..layout();

    final distPainter = TextPainter(
      text: distSpan,
      textDirection: TextDirection.ltr,
    )..layout();

    const padding = 5.0;
    const vGap = 2.0;
    final bgWidth = [labelPainter.width, distPainter.width]
            .reduce((a, b) => a > b ? a : b) +
        padding * 2;
    final bgHeight =
        labelPainter.height + distPainter.height + vGap + padding * 2;

    // Place tag above box if possible, otherwise below
    final tagTop = top - bgHeight - 4 >= 0 ? top - bgHeight - 4 : bottom + 4;
    final tagLeft = left.clamp(0.0, widgetSize.width - bgWidth);

    final bgRect = Rect.fromLTWH(tagLeft, tagTop, bgWidth, bgHeight);
    final bgPaint = Paint()..color = Colors.black.withOpacity(0.75);

    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
      bgPaint,
    );

    // Left accent stripe
    final stripePaint = Paint()..color = color;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(tagLeft, tagTop, 3, bgHeight),
        const Radius.circular(4),
      ),
      stripePaint,
    );

    labelPainter.paint(
      canvas,
      Offset(tagLeft + padding + 3, tagTop + padding),
    );
    distPainter.paint(
      canvas,
      Offset(
          tagLeft + padding + 3, tagTop + padding + labelPainter.height + vGap),
    );
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) =>
      oldDelegate.results != results ||
      oldDelegate.imageSize != imageSize ||
      oldDelegate.widgetSize != widgetSize;
}
