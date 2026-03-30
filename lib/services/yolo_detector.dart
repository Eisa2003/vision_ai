// FILE: lib/services/yolo_detector.dart

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/painting.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'base_detector.dart';

/// YOLOv8n detector using tflite_flutter (standard Interpreter API).
///
/// Model expected at: assets/models/yolov8n.tflite
/// Labels expected at: assets/models/coco_labels.txt  (one label per line)
///
/// YOLOv8n output tensor layout (ultralytics default export):
///   shape [1, 84, 8400]  — 84 = 4 box coords (cx,cy,w,h) + 80 class scores
///                          8400 anchors
class YoloDetector extends BaseDetector {
  static const _modelPath   = 'assets/models/yolov8n.tflite';
  static const _labelPath   = 'assets/models/coco_labels.txt';

  // YOLOv8n expects 640×640 RGB input
  static const int   _inputSize          = 640;
  static const double _confidenceThreshold = 0.45;
  static const double _iouThreshold        = 0.45;
  static const int   _maxResults          = 10;

  bool         _initialized = false;
  List<String> _labels      = [];
  Interpreter? _interpreter;

  @override
  String get name        => 'YOLOv8n';

  @override
  String get description => 'TFLite · 80 COCO classes · standard interpreter';

  // ─── Lifecycle ────────────────────────────────────────────────────────────

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    // Load COCO labels
    try {
      final raw = await rootBundle.loadString(_labelPath);
      _labels = raw
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();
    } catch (_) {
      _labels = List.generate(80, (i) => 'class_$i');
    }

    // Load TFLite model
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      _modelPath,
      options: options,
    );
    _interpreter!.allocateTensors();

    _initialized = true;
  }

  @override
  Future<void> close() async {
    if (!_initialized) return;
    _interpreter?.close();
    _interpreter  = null;
    _initialized  = false;
  }

  // ─── Public detect entry-points ───────────────────────────────────────────

  /// Generic detect — not used directly; live/file screens call the
  /// specialised methods below.
  @override
  Future<List<Detection>> detect(dynamic image) async => [];

  /// Called by LiveDetectionScreen when YOLO is active.
  /// [bytes]  — raw NV21 plane bytes from CameraImage.planes.first
  /// [imageW/H] — native frame dimensions (before rotation).
  Future<List<Detection>> detectFromBytes({
    required Uint8List bytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) async {
    if (!_initialized || _interpreter == null) return [];

    try {
      // Convert NV21 → RGB image → resize to 640×640
      final rgbImage = _nv21ToRgb(bytes, imageW, imageH, rotation);
      final input    = _imageToFloat32(rgbImage);
      return _runInference(input, imageW.toDouble(), imageH.toDouble());
    } catch (_) {
      return [];
    }
  }

  /// Called by ImageDetectionScreen when YOLO is active.
  Future<List<Detection>> detectFromFile(String filePath) async {
    if (!_initialized || _interpreter == null) return [];

    try {
      final fileBytes = await rootBundle.load(filePath);
      final decoded   = img.decodeImage(fileBytes.buffer.asUint8List());
      if (decoded == null) return [];

      final w      = decoded.width.toDouble();
      final h      = decoded.height.toDouble();
      final resized = img.copyResize(decoded,
          width: _inputSize, height: _inputSize,
          interpolation: img.Interpolation.linear);
      final input  = _imageToFloat32(resized);

      return _runInference(input, w, h);
    } catch (_) {
      return [];
    }
  }

  // ─── Inference ────────────────────────────────────────────────────────────

  /// Runs the interpreter and returns parsed detections.
  /// Input tensor:  [1, 640, 640, 3]  float32
  /// Output tensor: [1, 84, 8400]     float32
  List<Detection> _runInference(
      Float32List input, double origW, double origH) {
    final interpreter = _interpreter!;

    // Build input/output buffers
    final inputTensor = interpreter.getInputTensor(0);
    final inputShape  = inputTensor.shape; // [1, 640, 640, 3]

    // Reshape input list to match tensor shape
    final inputBuffer = input.reshape(inputShape);

    // Output: [1, 84, 8400]
    final outputShape = interpreter.getOutputTensor(0).shape;
    final outputBuffer = List.filled(
        outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);

    interpreter.run(inputBuffer, outputBuffer);

    // outputBuffer[0] is shape [84, 8400]
    final rawOutput = outputBuffer[0] as List;
    return _parseOutput(rawOutput, origW, origH);
  }

  // ─── Output parsing ───────────────────────────────────────────────────────

  /// Parses the raw [84 × 8400] output tensor.
  ///
  /// Layout per anchor:
  ///   row 0-3 : cx, cy, w, h  (normalised 0-1)
  ///   row 4-83: class scores
  List<Detection> _parseOutput(
      List<dynamic> output, double origW, double origH) {
    // output[i] is the i-th row (length 8400)
    final int numAnchors = (output[0] as List).length; // 8400
    final int numRows    = output.length;               // 84

    final List<_RawBox> candidates = [];

    for (int a = 0; a < numAnchors; a++) {
      // Find best class
      double maxScore = 0;
      int    bestClass = 0;
      for (int c = 4; c < numRows; c++) {
        final score = (output[c] as List)[a] as double;
        if (score > maxScore) {
          maxScore  = score;
          bestClass = c - 4;
        }
      }
      if (maxScore < _confidenceThreshold) continue;

      // Box coords (cx, cy, w, h) normalised to [0,1]
      final cx = (output[0] as List)[a] as double;
      final cy = (output[1] as List)[a] as double;
      final bw = (output[2] as List)[a] as double;
      final bh = (output[3] as List)[a] as double;

      // Convert to pixel coords in original image space
      final left   = (cx - bw / 2) * origW;
      final top    = (cy - bh / 2) * origH;
      final width  = bw * origW;
      final height = bh * origH;

      candidates.add(_RawBox(
        rect:       Rect.fromLTWH(left, top, width, height),
        confidence: maxScore,
        classIndex: bestClass,
      ));
    }

    // Non-maximum suppression
    final kept = _nms(candidates);

    return kept
        .take(_maxResults)
        .map((b) => Detection(
              boundingBox: b.rect,
              label:       _labelForIndex(b.classIndex),
              confidence:  b.confidence,
            ))
        .toList();
  }

  // ─── NMS ──────────────────────────────────────────────────────────────────

  List<_RawBox> _nms(List<_RawBox> boxes) {
    // Sort by confidence descending
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));

    final kept = <_RawBox>[];
    final suppressed = List.filled(boxes.length, false);

    for (int i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      kept.add(boxes[i]);
      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(boxes[i].rect, boxes[j].rect) > _iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final inter = a.intersect(b);
    if (inter.isEmpty) return 0;
    final interArea = inter.width  * inter.height;
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea <= 0 ? 0 : interArea / unionArea;
  }

  // ─── Image helpers ────────────────────────────────────────────────────────

  /// Converts raw NV21 bytes to a resized RGB [img.Image].
  img.Image _nv21ToRgb(
      Uint8List nv21, int width, int height, int rotation) {
    // Decode Y, UV planes from NV21 layout
    final yuvImage = img.Image.fromBytes(
      width:  width,
      height: height,
      bytes:  nv21.buffer,
      numChannels: 1, // treated as luma-only; we convert below
    );

    // Use the image package's YUV→RGB conversion
    var rgb = img.decodeImage(nv21) ??
        img.copyResize(yuvImage, width: _inputSize, height: _inputSize);

    // Rotate to correct orientation
    if (rotation != 0) {
      rgb = img.copyRotate(rgb, angle: rotation.toDouble());
    }

    // Resize to model input size
    return img.copyResize(rgb,
        width: _inputSize, height: _inputSize,
        interpolation: img.Interpolation.linear);
  }

  /// Converts an [img.Image] to a normalised Float32List [0,1].
  /// Output layout: HWC (height × width × channels).
  Float32List _imageToFloat32(img.Image image) {
    final pixels = image.width * image.height;
    final buffer = Float32List(pixels * 3);
    int idx = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        buffer[idx++] = pixel.r / 255.0;
        buffer[idx++] = pixel.g / 255.0;
        buffer[idx++] = pixel.b / 255.0;
      }
    }
    return buffer;
  }

  String _labelForIndex(int index) {
    if (index >= 0 && index < _labels.length) return _labels[index];
    return 'object';
  }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

class _RawBox {
  final Rect   rect;
  final double confidence;
  final int    classIndex;
  const _RawBox({
    required this.rect,
    required this.confidence,
    required this.classIndex,
  });
}
